"""Raw KGE ``nn.Module`` classes shared across torch-ns and DpRL.

Each model implements the ``KGEModel`` interface (``score_triples``,
``score_all_tails``, ``score_all_heads``, ``compose``). Embedding tables
live inside the model — dataset wiring stays in the consumer repos.
"""
from __future__ import annotations

from .base import KGEModel, det_embedding, det_gather_rows
from .complex import ComplEx
from .conve import ConvE
from .distmult import DistMult
from .factory import build_model, build_training_model
from .mode import ModE
from .rotate import RotatE
from .scorer import kge_default_scorer, recommended_eval_batch_size
from .transe import TransE
from .tucker import TuckER

__all__ = [
    "ComplEx",
    "ConvE",
    "DistMult",
    "KGEModel",
    "ModE",
    "RotatE",
    "TransE",
    "TuckER",
    "build_model",
    "build_training_model",
    "det_embedding",
    "det_gather_rows",
    "kge_default_scorer",
    "recommended_eval_batch_size",
]
