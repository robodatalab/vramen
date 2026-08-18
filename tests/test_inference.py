import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from vramen import resource_manager
from vramen.models import causal
from vramen.models.causal import CausalModel
from vramen.resource_manager import InferenceModelResourceManager
from vramen.utils import qwen_chat_prompt


class ThreadQueue(queue.Queue):
    """The queue of a serving process that is really a thread.

    A request reaches the real serving process by pickling; here it is handed
    over as it is, so a test can send one that closes over its own events.
    """

    def close(self) -> None:
        pass


class ModelCompletionTests(unittest.TestCase):

    def setUp(self):
        self.spawned: list[MagicMock] = []

        def spawn(target=None, kwargs=None, daemon=None):
            thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
            process = MagicMock()
            process.start.side_effect = thread.start
            process.is_alive.side_effect = thread.is_alive
            process.join.side_effect = thread.join
            self.spawned.append(process)
            return process

        self.addCleanup(patch.stopall)
        patch.object(resource_manager, "Process", side_effect=spawn).start()
        patch.object(resource_manager, "Queue", ThreadQueue).start()
        patch.object(causal, "AutoTokenizer").start()
        self.transformer = patch.object(causal, "AutoModelForCausalLM").start()

        # Room for one of the models below, so that a second one has to swap.
        self.resource_manager = InferenceModelResourceManager(quota_gb=8.0)
        self.addCleanup(self.resource_manager.shutdown)

    def test_complete_waits_until_model_is_loaded(self):
        loaded = threading.Event()

        def load(*_args, **_kwargs):
            loaded.wait(timeout=5)
            return MagicMock()

        self.transformer.from_pretrained.side_effect = load
        patch.object(
            causal, "_complete_instruct", return_value="model response 1"
        ).start()
        model = CausalModel(
            "test-org/test-model", qwen_chat_prompt, self.resource_manager, 5.0
        )

        responses = []

        def call():
            responses.append(model.complete("system", "user", max_new_tokens=10))

        caller = threading.Thread(target=call, daemon=True)
        caller.start()

        time.sleep(0.05)
        self.assertEqual(responses, [])

        loaded.set()
        caller.join(timeout=5)
        self.assertSequenceEqual(responses, ["model response 1"])

    def test_model_runs_inference(self):
        patch.object(
            causal, "_complete_instruct", return_value="model response 2"
        ).start()
        model = CausalModel(
            "test-org/test-model", qwen_chat_prompt, self.resource_manager, 5.0
        )

        response = model.complete("system", "user", max_new_tokens=10)

        self.assertEqual(response, "model response 2")

    def test_a_swap_waits_for_a_call_in_flight(self):
        # Otherwise the grace period hides the tear-down behind a long wait, and
        # the assertion below passes for the wrong reason.
        patch.object(resource_manager, "STOP_GRACE_S", 0.05).start()

        generating = threading.Event()
        finish = threading.Event()

        def slow_generate(*_args, **_kwargs):
            generating.set()
            finish.wait(timeout=5)
            return "first response"

        patch.object(causal, "_complete_instruct", side_effect=slow_generate).start()

        first = CausalModel(
            "test-org/first", qwen_chat_prompt, self.resource_manager, 5.0
        )
        second = CausalModel(
            "test-org/second", qwen_chat_prompt, self.resource_manager, 5.0
        )

        responses = []

        def call_first():
            responses.append(first.complete("system", "user", max_new_tokens=10))

        def call_second():
            second.complete("system", "user", max_new_tokens=10)

        caller = threading.Thread(target=call_first, daemon=True)
        caller.start()
        self.assertTrue(generating.wait(timeout=5))

        swapper = threading.Thread(target=call_second, daemon=True)
        swapper.start()
        time.sleep(0.2)

        # The first model is still owed an answer, so its process must be intact.
        self.assertFalse(self.spawned[0].terminate.called)

        finish.set()
        caller.join(timeout=5)
        swapper.join(timeout=5)
        self.assertSequenceEqual(responses, ["first response"])


if __name__ == "__main__":
    unittest.main()
