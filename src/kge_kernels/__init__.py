"""Shared PyTorch kernels for KGE sampling and scoring."""

from .partial import precompute_partial_scores, score_partial_atoms
from .sampler import Sampler, SamplerConfig, corrupt
from .scoring import (
    KGEBackend,
    score,
)
from .types import CorruptionOutput, ScoreOutput

__all__ = [
    "Sampler",
    "SamplerConfig",
    "KGEBackend",
    "CorruptionOutput",
    "ScoreOutput",
    "corrupt",
    "precompute_partial_scores",
    "score",
    "score_partial_atoms",
]
