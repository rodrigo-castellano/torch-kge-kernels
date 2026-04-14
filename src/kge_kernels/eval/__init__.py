"""Evaluation framework: unified evaluator, ranking metrics, corruption pools,
rank fusion, and checkpoint evaluation."""

from .checkpoint import evaluate_checkpoint, evaluate_ranking
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
    "evaluate_checkpoint",
    "evaluate_ranking",
    "ranking_metrics",
    "rrf",
    "zscore_fusion",
]
