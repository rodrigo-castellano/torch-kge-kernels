"""Evaluation framework: unified evaluator, ranking metrics, corruption pools,
and rank fusion."""

from .evaluator import Evaluator, FusionFn, ScorerFn
from .fusion import rrf, zscore_fusion
from .pool import CandidatePool
from .ranking import (
    StreamingRankingMetrics,
    compute_ranks,
    ranking_metrics,
)
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "EvalResults",
    "Evaluator",
    "FusionFn",
    "ScorerFn",
    "StreamingRankingMetrics",
    "compute_ranks",
    "ranking_metrics",
    "rrf",
    "zscore_fusion",
]
