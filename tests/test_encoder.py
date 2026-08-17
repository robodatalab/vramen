import queue
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch
from transformers import BatchEncoding

from vramen import encoder, resource_manager
from vramen.encoder import EncoderModel
from vramen.resource_manager import InferenceModelResourceManager


class ThreadQueue(queue.Queue):
    """The queue of a serving process that is really a thread.

    A request reaches the real serving process by pickling; here it is handed
    over as it is, so a test can send one that closes over its own events.
    """

    def close(self) -> None:
        pass


class LastToken(unittest.TestCase):
    """Pooling reads the end of a passage, wherever the padding leaves it."""

    def test_left_padding_takes_the_final_column(self) -> None:
        hidden = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        attention_mask = torch.tensor([[0, 1], [1, 1]])

        pooled = encoder._last_token(hidden, attention_mask)

        self.assertEqual(pooled.tolist(), [[2.0], [4.0]])

    def test_right_padding_takes_each_row_at_its_own_length(self) -> None:
        hidden = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
        attention_mask = torch.tensor([[1, 0], [1, 1]])

        pooled = encoder._last_token(hidden, attention_mask)

        self.assertEqual(pooled.tolist(), [[1.0], [4.0]])


class Batching(unittest.TestCase):
    """A forward pass holds its whole batch, so the batch has a ceiling."""

    def test_passages_past_the_batch_size_go_in_several_passes(self):
        rows_per_pass = []

        def tokenize(batch, **_kwargs):
            rows = len(batch)
            return BatchEncoding(
                {
                    "input_ids": torch.ones(rows, 4, dtype=torch.long),
                    "attention_mask": torch.ones(rows, 4, dtype=torch.long),
                }
            )

        def forward(input_ids, attention_mask):
            rows = int(input_ids.shape[0])
            rows_per_pass.append(rows)
            return MagicMock(last_hidden_state=torch.ones(rows, 4, 2))

        tokenizer = MagicMock(side_effect=tokenize)
        model = MagicMock(side_effect=forward, device="cpu")
        passages = tuple(f"passage {n}" for n in range(encoder.BATCH_SIZE + 3))

        vectors = encoder._encode(passages, model, tokenizer)

        self.assertEqual(rows_per_pass, [encoder.BATCH_SIZE, 3])
        self.assertEqual(list(vectors.shape), [len(passages), 2])

    def test_every_vector_comes_back_at_unit_length(self):
        def tokenize(batch, **_kwargs):
            return BatchEncoding(
                {
                    "input_ids": torch.ones(len(batch), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(batch), 4, dtype=torch.long),
                }
            )

        def forward(input_ids, attention_mask):
            rows = int(input_ids.shape[0])
            hidden = torch.arange(1.0, rows * 4 * 2 + 1).reshape(rows, 4, 2)
            return MagicMock(last_hidden_state=hidden)

        tokenizer = MagicMock(side_effect=tokenize)
        model = MagicMock(side_effect=forward, device="cpu")

        vectors = encoder._encode(("one", "two", "three"), model, tokenizer)

        self.assertTrue(
            torch.allclose(vectors.norm(dim=1), torch.ones(3), atol=1e-6)
        )


class Encoding(unittest.TestCase):

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
        patch.object(encoder, "AutoTokenizer").start()
        patch.object(encoder, "AutoModel").start()

        self.resource_manager = InferenceModelResourceManager(quota_gb=8.0)
        self.addCleanup(self.resource_manager.shutdown)

    def test_passages_are_encoded_as_they_stand(self):
        encoded = []

        def encode(texts, model, tokenizer):
            encoded.append(texts)
            return [[0.1, 0.2], [0.3, 0.4]]

        patch.object(encoder, "_encode", side_effect=encode).start()
        model = EncoderModel("test-org/test-encoder", self.resource_manager, 5.0)

        vectors = model.encode(["knocking at the door", "michael goes down"])

        self.assertEqual(encoded, [("knocking at the door", "michael goes down")])
        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])

    def test_a_failure_in_the_serving_process_is_raised_to_the_caller(self):
        patch.object(encoder, "_encode", side_effect=RuntimeError("out of memory")).start()
        model = EncoderModel("test-org/test-encoder", self.resource_manager, 5.0)

        with self.assertRaises(resource_manager.ModelNotAvailable):
            model.encode(["a passage"])


if __name__ == "__main__":
    unittest.main()
