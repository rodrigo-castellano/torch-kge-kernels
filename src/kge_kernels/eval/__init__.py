"""Evaluation framework: ranking metrics, corruption pools, rank fusion,
filtered ranking, and metrics."""

from .evaluator import Evaluator, FusionFn, ScorerFn
from .filtered import evaluate_filtered_ranking
from .fusion import rrf, zscore_fusion
from .pool import CandidatePool
from .ranking import (
    StreamingRankingMetrics,
    ranking_metrics,
    ranks_from_labeled_predictions,
    ranks_from_scores,
    ranks_from_scores_matrix,
)
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "EvalResults",
    "Evaluator",
    "FusionFn",
    "ScorerFn",
    "StreamingRankingMetrics",
    "evaluate_filtered_ranking",
    "ranking_metrics",
    "ranks_from_labeled_predictions",
    "ranks_from_scores",
    "ranks_from_scores_matrix",
    "rrf",
    "zscore_fusion",
]
