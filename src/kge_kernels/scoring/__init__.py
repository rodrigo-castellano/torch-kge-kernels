"""Scoring pipeline: types, corruption sampling, backend dispatch, model adaptation, and partial-atom scoring."""

from .adapter import (
    apply_masks,
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
)
from .adapter import precompute_partial_scores as precompute_partial_scores_from_model
from .backend import score
from .partial import LazyPartialScorer, precompute_partial_scores, score_partial_atoms
from .sampler import Sampler, corrupt
from .types import (
    CorruptionOutput,
    KGEBackend,
    LongTensor,
    SamplerConfig,
    ScoreAllHeadsFn,
    ScoreAllTailsFn,
    ScoreOutput,
    ScoreTriplesFn,
    SupportsCorruptWithMask,
)
from .utils import compute_bernoulli_probs

__all__ = [
    # Types
    "CorruptionOutput",
    "KGEBackend",
    "LongTensor",
    "SamplerConfig",
    "ScoreAllHeadsFn",
    "ScoreAllTailsFn",
    "ScoreOutput",
    "ScoreTriplesFn",
    "SupportsCorruptWithMask",
    # Sampler
    "Sampler",
    "corrupt",
    # Backend
    "score",
    # Adapter
    "apply_masks",
    "build_backend",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "precompute_partial_scores_from_model",
    # Partial
    "LazyPartialScorer",
    "score_partial_atoms",
    # Utils
    "compute_bernoulli_probs",
]
