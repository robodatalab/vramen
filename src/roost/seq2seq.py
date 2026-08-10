from functools import partial
import time

from roost import log
from roost.types import Model, PromptFormatter, Tokenizer
from roost.resource_manager import (
    InferenceModelResourceManager, 
    ModelKind, 
    ModelNotAvailable
)
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_log = log.logger(__name__)


class Seq2SeqModel(ModelKind):
    def __init__(
            self,
            model_id: str,
            prompt: PromptFormatter,
            manager: InferenceModelResourceManager,
            mem_required_gb: float,
        ) -> None:
        super().__init__(model_id, manager, mem_required_gb)
        self.prompt = prompt

    def load(self) -> tuple[Model, Tokenizer]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="mps"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model.eval()
        return model, tokenizer

    def complete(self, system: str, user: str, max_new_tokens: int) -> str:
        with self.manager.residency(self) as (requests, replies):
            request = partial(_complete_seq2seq, self.prompt, system, user, max_new_tokens)
            requests.put(request)
            error, text = replies.get()
        if error is not None:
            raise ModelNotAvailable(f"{str(self)} failed to answer: {error}")
        return text


def _complete_seq2seq(
    prompt_formatter: PromptFormatter, 
    system: str, 
    user: str, 
    max_new_tokens: int,
    model, 
    tokenizer,
) -> str:
    prompt = prompt_formatter(system, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    started = time.monotonic()
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False
    )
    _log.info(
        "edited %d tokens in %.1fs",
        int(output[0].shape[-1]),
        time.monotonic() - started,
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text if isinstance(text, str) else "".join(text)
