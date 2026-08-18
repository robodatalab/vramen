"""The model kinds, one module per way a checkpoint is loaded and run.

Each subclasses `ModelKind` and holds the `transformers` classes its kind
needs, so a `vramen.models.causal` reads only as the causal case.
"""

from vramen.models.causal import CausalModel
from vramen.models.encoder import EncoderModel
from vramen.models.seq2seq import Seq2SeqModel

__all__ = [
    "CausalModel",
    "EncoderModel",
    "Seq2SeqModel",
]
