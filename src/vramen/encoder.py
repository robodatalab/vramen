from collections.abc import Sequence
from functools import partial
import time

from vramen import log
from vramen.types import Model, Tokenizer
from vramen.resource_manager import (
    InferenceModelResourceManager,
    ModelKind,
    ModelNotAvailable
)
import torch
from transformers import AutoModel, AutoTokenizer

_log = log.logger(__name__)

# Longer than any passage a manuscript is cut into, and well inside the window.
MAX_TOKENS = 8192

# Encoding is a single forward pass, so the whole set would go at once and take
# the memory of the longest passage times all of them.
BATCH_SIZE = 8


class EncoderModel(ModelKind):
    def __init__(
            self,
            model_id: str,
            manager: InferenceModelResourceManager,
            mem_required_gb: float,
        ) -> None:
        super().__init__(model_id, manager, mem_required_gb)

    def load(self) -> tuple[Model, Tokenizer]:
        model = AutoModel.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="mps"
        )
        # Pooling reads the last token of each row, which left padding lines up
        # in the final column.
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        model.eval()
        return model, tokenizer

    def encode(self, passages: Sequence[str]) -> list[list[float]]:
        """A vector per passage, as the passage stands."""
        return self._vectors(tuple(passages))

    def _vectors(self, texts: tuple[str, ...]) -> list[list[float]]:
        with self.manager.residency(self) as (requests, replies):
            requests.put(partial(_encode, texts))
            error, vectors = replies.get()
        if error is not None:
            raise ModelNotAvailable(f"{str(self)} failed to encode: {error}")
        return vectors


def _encode(texts: tuple[str, ...], model, tokenizer) -> torch.Tensor:
    _log.debug("encoding %d passages", len(texts))
    started = time.monotonic()

    batches: list[torch.Tensor] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[start:start + BATCH_SIZE])
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model(**inputs)

        pooled = _last_token(output.last_hidden_state, inputs["attention_mask"])
        # Unit length, so that a dot product is a cosine and nearness is a sort.
        unit = torch.nn.functional.normalize(pooled, p=2, dim=1)
        # On the CPU in float32: the vectors go back across a process boundary,
        # and a tensor would otherwise carry its device with it.
        batches.append(unit.float().cpu())

    elapsed = time.monotonic() - started
    _log.debug(
        "encoded %d passages in %.1fs (%.1f/s)",
        len(texts),
        elapsed,
        len(texts) / elapsed if elapsed else 0.0,
    )
    return torch.cat(batches) if batches else torch.empty(0)


def _last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """The final real token of each row — what these models pool a passage into.

    Left padding lines those tokens up in the last column, so the whole batch
    reads at once; a tokenizer padding on the right needs each row's own length.
    """
    if bool(attention_mask[:, -1].sum() == attention_mask.shape[0]):
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
