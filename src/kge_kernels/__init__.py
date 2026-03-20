"""Shared PyTorch kernels for KGE sampling and scoring."""

from .adapter import (
    apply_masks,
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
)
from .adapter import precompute_partial_scores as precompute_partial_scores
from .partial import score_partial_atoms
from .ranking import ranking_metrics, ranks_from_scores, ranks_from_scores_matrix
from .sampler import Sampler, SamplerConfig, corrupt
from .scoring import (
    KGEBackend,
    score,
)
from .types import CorruptionOutput, ScoreOutput
from .utils import compute_bernoulli_probs

__all__ = [
    "Sampler",
    "SamplerConfig",
    "KGEBackend",
    "CorruptionOutput",
    "ScoreOutput",
    "apply_masks",
    "build_backend",
    "compute_bernoulli_probs",
    "corrupt",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "ranking_metrics",
    "ranks_from_scores",
    "ranks_from_scores_matrix",
    "score",
    "score_partial_atoms",
]
