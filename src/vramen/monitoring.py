from multiprocessing.queues import Queue
import time
from typing import Any

from vramen import log
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, TextStreamer


HEARTBEAT_S = 5.0
PROGRESS_EVERY = 64

_log = log.logger(__name__)


class TextStreamerProgressMonitor(TextStreamer):
    """An adapter for monitoring the progress of LLM answer generation."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, budget: int) -> None:
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.budget = budget
        self.tokens = 0
        self.started = time.monotonic()

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        self.tokens += 1
        if self.tokens % PROGRESS_EVERY == 0:
            elapsed = time.monotonic() - self.started
            _log.info(
                "generating %d/%d tokens, %.0fs elapsed, %.1f tok/s",
                self.tokens,
                self.budget,
                elapsed,
                self.tokens / elapsed if elapsed else 0.0,
            )


class DenoisingProgressMonitor:
    """An adapter for monitoring the progress of a diffusion pipeline.

    Diffusers calls this at the end of every step and reads a dictionary of
    replacement tensors back out of it. This one replaces nothing and only says
    where the run has got to, on a heartbeat rather than on every step: a small
    pipeline steps faster than anybody needs telling, and the line that matters
    at the end of a big one is the one the drawing itself writes.
    """

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.started = time.monotonic()
        self.spoken = self.started

    def __call__(
        self, pipe: Any, step: int, timestep: Any, tensors: dict[str, Any]
    ) -> dict[str, Any]:
        done = step + 1
        now = time.monotonic()
        if now - self.spoken >= HEARTBEAT_S and done < self.steps:
            self.spoken = now
            elapsed = now - self.started
            _log.info(
                "denoising %d/%d steps, %.0fs elapsed, %.1fs/step",
                done,
                self.steps,
                elapsed,
                elapsed / done,
            )
        return {}


def reporting_tqdm(signal: Queue[float | str]) -> type:
    class ReportingTqdm(tqdm):  # type: ignore[type-arg]
        def update(self, n: float | None = 1) -> bool | None:
            updated = super().update(n)
            if self.unit != "B" and self.total:
                signal.put(min(self.n / self.total, 1.0))
            return updated

    return ReportingTqdm
