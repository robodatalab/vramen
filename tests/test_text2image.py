import queue
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from vramen import resource_manager
from vramen.models import text2image
from vramen.models.text2image import Text2ImageModel
from vramen.resource_manager import InferenceModelResourceManager


class ThreadQueue(queue.Queue):
    """The queue of a serving process that is really a thread.

    A request reaches the real serving process by pickling; here it is handed
    over as it is, so a test can send one that closes over its own events.
    """

    def close(self) -> None:
        pass


class Picture:
    """A PIL image as far as this package is concerned: something that saves."""

    def __init__(self, data: bytes = b"the png") -> None:
        self.data = data
        self.saved_as: str | None = None

    def save(self, buffer, format: str) -> None:
        self.saved_as = format
        buffer.write(self.data)


def drawing(picture: Picture, on_call=None) -> MagicMock:
    """A pipeline that hands back one picture, and records how it was asked."""

    def pipeline(**kwargs):
        if on_call is not None:
            on_call(kwargs)
        return MagicMock(images=[picture])

    return MagicMock(side_effect=pipeline)


class Drawing(unittest.TestCase):
    """What the pipeline is asked for, and what comes back out of it."""

    def test_the_pipeline_is_asked_for_the_size_steps_and_guidance(self):
        pipe = drawing(Picture())

        text2image._draw("a heron", "blurry", 512, 768, 12, 2.5, None, pipe, None)

        asked = pipe.call_args.kwargs
        self.assertEqual(asked["prompt"], "a heron")
        self.assertEqual(asked["negative_prompt"], "blurry")
        self.assertEqual((asked["width"], asked["height"]), (512, 768))
        self.assertEqual(asked["num_inference_steps"], 12)
        self.assertEqual(asked["guidance_scale"], 2.5)

    def test_the_picture_comes_back_as_the_bytes_of_a_png(self):
        picture = Picture(b"\x89PNG and the rest")
        pipe = drawing(picture)

        png = text2image._draw("a heron", "", 64, 64, 2, 3.0, None, pipe, None)

        self.assertEqual(png, b"\x89PNG and the rest")
        self.assertEqual(picture.saved_as, "PNG")

    def test_a_seed_is_the_seed_the_caller_asked_for(self):
        pipe = drawing(Picture())

        text2image._draw("a heron", "", 64, 64, 2, 3.0, 433, pipe, None)

        generator = pipe.call_args.kwargs["generator"]
        self.assertEqual(generator.initial_seed(), 433)

    def test_without_a_seed_the_pipeline_draws_from_its_own_noise(self):
        """A fresh `torch.Generator` starts from a fixed seed, so passing one
        unseeded would draw the same picture every time. None is the ask for
        the global stream, which moves on between calls."""
        pipe = drawing(Picture())

        text2image._draw("a heron", "", 64, 64, 2, 3.0, None, pipe, None)

        self.assertIsNone(pipe.call_args.kwargs["generator"])

    def test_the_progress_monitor_is_handed_every_step_and_replaces_nothing(self):
        replaced = []

        def denoise(asked):
            monitor = asked["callback_on_step_end"]
            for step in range(asked["num_inference_steps"]):
                # What the pipeline does with the answer: pop its tensors back
                # out of it. Anything but a mapping ends the run here.
                replaced.append(monitor(None, step, 0.0, {"latents": None}))

        pipe = drawing(Picture(), on_call=denoise)

        text2image._draw("a heron", "", 64, 64, 3, 3.0, None, pipe, None)

        self.assertEqual(replaced, [{}, {}, {}])


class Loading(unittest.TestCase):
    """Where the weights land, and what the serving process is handed."""

    def setUp(self):
        self.addCleanup(patch.stopall)
        self.pipe = MagicMock()
        pipeline = MagicMock()
        pipeline.from_pretrained.return_value = self.pipe
        self.pipeline = pipeline
        patch.object(text2image, "_pipeline_class", return_value=pipeline).start()

    def test_the_weights_go_onto_the_gpu_in_bfloat16(self):
        model = Text2ImageModel("test-org/test-image", None, 28.0)

        pipe, tokenizer = model.load()

        self.pipeline.from_pretrained.assert_called_once_with(
            "test-org/test-image", torch_dtype=torch.bfloat16
        )
        self.pipe.to.assert_called_once_with("mps")
        self.pipe.enable_model_cpu_offload.assert_not_called()
        self.assertIs(pipe, self.pipe)
        self.assertIs(tokenizer, self.pipe.tokenizer)

    def test_offloading_leaves_the_placement_to_diffusers(self):
        model = Text2ImageModel("test-org/test-image", None, 28.0, offload=True)

        model.load()

        self.pipe.enable_model_cpu_offload.assert_called_once_with()
        self.pipe.to.assert_not_called()

    def test_the_pipelines_own_progress_bar_is_turned_off(self):
        model = Text2ImageModel("test-org/test-image", None, 28.0)

        model.load()

        self.pipe.set_progress_bar_config.assert_called_once_with(disable=True)


class Serving(unittest.TestCase):
    """The picture across the process boundary, and the failure with it."""

    def setUp(self):
        def spawn(target=None, kwargs=None, daemon=None):
            thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
            process = MagicMock()
            process.start.side_effect = thread.start
            process.is_alive.side_effect = thread.is_alive
            process.join.side_effect = thread.join
            return process

        self.addCleanup(patch.stopall)
        patch.object(resource_manager, "Process", side_effect=spawn).start()
        patch.object(resource_manager, "Queue", ThreadQueue).start()
        pipeline = MagicMock()
        patch.object(text2image, "_pipeline_class", return_value=pipeline).start()

        self.resource_manager = InferenceModelResourceManager(quota_gb=32.0)
        self.addCleanup(self.resource_manager.shutdown)

    def test_the_prompt_is_drawn_and_the_png_comes_back(self):
        drawn = []

        def draw(*args):
            # The last two are the model and the tokenizer, which the serving
            # process adds on its own side.
            drawn.append(args[:-2])
            return b"the png"

        patch.object(text2image, "_draw", side_effect=draw).start()
        model = Text2ImageModel("test-org/test-image", self.resource_manager, 28.0)

        png = model.draw("a heron on a jetty", seed=433)

        self.assertEqual(png, b"the png")
        self.assertEqual(
            drawn, [("a heron on a jetty", "", 1024, 1024, 40, 3.0, 433)]
        )

    def test_the_negative_prompt_stands_until_a_call_overrides_it(self):
        drawn = []
        patch.object(
            text2image, "_draw", side_effect=lambda *args: drawn.append(args[:2])
        ).start()
        model = Text2ImageModel(
            "test-org/test-image",
            self.resource_manager,
            28.0,
            negative_prompt="blurry",
        )

        model.draw("a heron")
        model.draw("a heron", negative_prompt="")

        self.assertEqual(drawn, [("a heron", "blurry"), ("a heron", "")])

    def test_a_failure_in_the_serving_process_is_raised_to_the_caller(self):
        patch.object(
            text2image, "_draw", side_effect=RuntimeError("out of memory")
        ).start()
        model = Text2ImageModel("test-org/test-image", self.resource_manager, 28.0)

        with self.assertRaises(resource_manager.ModelNotAvailable):
            model.draw("a heron")


class WithoutTheExtra(unittest.TestCase):

    def test_a_missing_diffusers_names_the_extra_that_supplies_it(self):
        with patch.dict("sys.modules", {"diffusers": None}):
            with self.assertRaises(resource_manager.ModelNotAvailable) as failure:
                text2image._pipeline_class()

        self.assertIn("vramen[image]", str(failure.exception))


if __name__ == "__main__":
    unittest.main()
