"""A local inference server that keeps several models resident side by side.

Everything public is re-exported here, so `from vramen import CausalModel`
reads the same to a type checker as it does at runtime. The submodules stay
importable for the internals: `vramen.encoder.BATCH_SIZE` and the like.
"""

from vramen import log
from vramen.monitoring import TextStreamerProgressMonitor, reporting_tqdm
from vramen.resource_manager import (
    InferenceModelResourceManager,
    MemoryReading,
    ModelKind,
    ModelNotAvailable,
)
from vramen.causal import CausalModel
from vramen.encoder import EncoderModel
from vramen.seq2seq import Seq2SeqModel
from vramen.types import Model, PromptFormatter, Tokenizer
from vramen.utils import (
    coedit_prompt,
    gpu_memory_limit,
    gpu_memory_used,
    gpu_tensors,
    machine_memory,
    process_memory,
    qwen_chat_prompt,
)

__all__ = [
    # Managing the quota
    "InferenceModelResourceManager",
    "MemoryReading",
    "ModelNotAvailable",
    # Model kinds
    "CausalModel",
    "EncoderModel",
    "ModelKind",
    "Seq2SeqModel",
    # Prompts
    "coedit_prompt",
    "qwen_chat_prompt",
    # Memory readings
    "gpu_memory_limit",
    "gpu_memory_used",
    "gpu_tensors",
    "machine_memory",
    "process_memory",
    # Types
    "Model",
    "PromptFormatter",
    "Tokenizer",
    # Logging and progress
    "TextStreamerProgressMonitor",
    "log",
    "reporting_tqdm",
]
