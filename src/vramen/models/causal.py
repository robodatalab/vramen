from functools import partial
import time

from vramen import log
from vramen.types import Model, PromptFormatter, Tokenizer
from vramen.monitoring import TextStreamerProgressMonitor
from vramen.resource_manager import (
    InferenceModelResourceManager, 
    ModelKind, 
    ModelNotAvailable
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_log = log.logger(__name__)


class CausalModel(ModelKind):
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
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map="mps"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model.eval()
        return model, tokenizer

    def complete(self, system: str, user: str, max_new_tokens: int) -> str:
        with self.manager.residency(self) as (requests, replies):
            request = partial(_complete_instruct, self.prompt, system, user, max_new_tokens)
            requests.put(request)
            error, text = replies.get()
        if error is not None:
            raise ModelNotAvailable(f"{str(self)} failed to answer: {error}")
        return text


def _complete_instruct(
    prompt_formatter: PromptFormatter, 
    system: str, 
    user: str, 
    max_new_tokens: int, 
    model, 
    tokenizer
) -> str:
    prompt = prompt_formatter(system, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    _log.info(
        "generating: %d prompt tokens, up to %d new", prompt_tokens, max_new_tokens
    )
    started = time.monotonic()

    streamer = TextStreamerProgressMonitor(tokenizer, max_new_tokens)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )

    generated = int(output[0].shape[-1]) - prompt_tokens
    elapsed = time.monotonic() - started
    _log.info(
        "generated %d tokens in %.1fs (%.1f tok/s)%s",
        generated,
        elapsed,
        generated / elapsed if elapsed else 0.0,
        " — hit the budget" if generated >= max_new_tokens else "",
    )

    text = tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)
    return text if isinstance(text, str) else "".join(text)
