"""Raw KGE ``nn.Module`` classes shared across torch-ns and DpRL.

Each model implements the ``KGEModel`` interface (``score_triples``,
``score_all_tails``, ``score_all_heads``, ``compose``). Embedding tables
live inside the model — dataset wiring stays in the consumer repos.
"""
from __future__ import annotations

from .base import KGEModel
from .complex import ComplEx
from .distmult import DistMult
from .mode import ModE
from .rotate import RotatE
from .transe import TransE
from .tucker import TuckER

__all__ = [
    "ComplEx",
    "DistMult",
    "KGEModel",
    "ModE",
    "RotatE",
    "TransE",
    "TuckER",
]
