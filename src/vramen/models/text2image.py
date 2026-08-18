from functools import partial
from io import BytesIO
import time
from typing import Any

from vramen import log
from vramen.monitoring import DenoisingProgressMonitor
from vramen.types import Model, Tokenizer
from vramen.resource_manager import (
    InferenceModelResourceManager,
    ModelKind,
    ModelNotAvailable
)
from huggingface_hub import try_to_load_from_cache
import torch

_log = log.logger(__name__)

# What the Chroma card asks for. A denoiser is nothing like a chat model here:
# the steps are the whole run, and the guidance is a dial that ruins the image
# at both ends.
STEPS = 40
GUIDANCE = 3.0

# These pipelines are trained at a size, and drift when taken far off it.
SIZE = 1024

# The file every diffusers repository has, and the one this looks for to tell a
# load from a download.
INDEX_FILE = "model_index.json"


class Text2ImageModel(ModelKind):
    """A diffusers text to image pipeline, resident under the same quota.

    `model_index.json` names the pipeline class a checkpoint wants and
    `DiffusionPipeline` builds it, so Chroma, Flux and the rest arrive here
    without a class each. What they have in common is a prompt in and a picture
    out, and a size that makes the quota worth having: `lodestones/Chroma1-HD`
    is an 8.9B denoiser over a T5 encoder, some 28GB of bfloat16 together.
    """

    def __init__(
            self,
            model_id: str,
            manager: InferenceModelResourceManager,
            mem_required_gb: float,
            negative_prompt: str = "",
            offload: bool = False,
        ) -> None:
        super().__init__(model_id, manager, mem_required_gb)
        self.negative_prompt = negative_prompt
        self.offload = offload

    def load(self) -> tuple[Model, Tokenizer]:
        pipeline = _pipeline_class()
        if not _weights_are_cached(self.model_id):
            _log.info(
                "%s is not in the hub cache, so the load below fetches it first",
                self.model_id,
            )

        pipe = pipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)

        if self.offload:
            # One component on the accelerator at a time, the rest waiting in
            # RAM. That is room bought on a discrete card; on unified memory it
            # buys nothing, because RAM is where they already were.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("mps")

        # The steps are logged, so the bar would only be a second opinion, and
        # a child process is nobody's terminal.
        pipe.set_progress_bar_config(disable=True)

        # The pipeline keeps its own tokenizer, and the caller of a request does
        # not have to know which of them it is.
        return pipe, getattr(pipe, "tokenizer", None)

    def draw(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = SIZE,
        height: int = SIZE,
        steps: int = STEPS,
        guidance: float = GUIDANCE,
        seed: int | None = None,
    ) -> bytes:
        """One picture of the prompt, as the bytes of a PNG.

        A seed draws the same picture again from the same prompt and size.
        Without one the pipeline draws something new every call.
        """
        with self.manager.residency(self) as (requests, replies):
            request = partial(
                _draw,
                prompt,
                self.negative_prompt if negative_prompt is None else negative_prompt,
                width,
                height,
                steps,
                guidance,
                seed,
            )
            requests.put(request)
            error, png = replies.get()
        if error is not None:
            raise ModelNotAvailable(f"{self.model_id} failed to draw: {error}")
        return png


def _pipeline_class() -> Any:
    """`DiffusionPipeline`, imported on the load rather than on the import.

    Drawing is an extra — diffusers, Pillow, and the sentencepiece the T5
    tokenizers are still shipped as — so a package that only ever generates
    text does not carry any of it.
    """
    try:
        from diffusers import DiffusionPipeline  # type: ignore[import-not-found]
    except ImportError as missing:
        raise ModelNotAvailable(
            "drawing needs the image extra: pip install 'vramen[image]'"
        ) from missing
    return DiffusionPipeline


def _weights_are_cached(model_id: str) -> bool:
    """Whether the hub has this repository already, so a load is not a download.

    `from_pretrained` fetches whatever is missing either way. This only decides
    whether to say so first, because the first load of a pipeline this size is
    an hour of network that otherwise reads as a hang. The index file is the
    cheap tell: an interrupted download leaves it behind and still answers yes.
    """
    return isinstance(try_to_load_from_cache(model_id, INDEX_FILE), str)


def _draw(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    pipe,
    _tokenizer,
) -> bytes:
    _log.info(
        "drawing %dx%d in %d steps, guidance %.1f", width, height, steps, guidance
    )
    started = time.monotonic()

    # Seeded on the CPU, as the model card has it: the noise a CPU generator
    # draws is the same noise on any backend, which is what makes a seed worth
    # writing down next to the picture. A generator of its own only when there
    # is a seed for it — a fresh one starts from torch's fixed default seed, so
    # handing the pipeline an unseeded generator would draw the same picture
    # every call. Left to itself it takes the global stream, which moves on.
    generator = torch.Generator("cpu").manual_seed(seed) if seed is not None else None

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        callback_on_step_end=DenoisingProgressMonitor(steps),
    ).images[0]

    elapsed = time.monotonic() - started
    _log.info(
        "drew %dx%d in %.1fs (%.1fs/step)",
        width,
        height,
        elapsed,
        elapsed / steps if steps else 0.0,
    )
    return _png(image)


def _png(image) -> bytes:
    """The picture as PNG bytes, which is the form that leaves the process.

    The reply travels a queue, and so a pickle. Bytes cross that as themselves,
    ask for no PIL on the other side, and are already what a caller writes to a
    file or hands to a browser.
    """
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
