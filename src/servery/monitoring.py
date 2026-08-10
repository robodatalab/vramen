from multiprocessing.queues import Queue
import time

from servery import log
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


def reporting_tqdm(signal: Queue[float | str]) -> type:
    class ReportingTqdm(tqdm):  # type: ignore[type-arg]
        def update(self, n: float | None = 1) -> bool | None:
            updated = super().update(n)
            if self.unit != "B" and self.total:
                signal.put(min(self.n / self.total, 1.0))
            return updated

    return ReportingTqdm
