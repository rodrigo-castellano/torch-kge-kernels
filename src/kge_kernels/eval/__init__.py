"""Evaluation framework: single ``evaluate()`` entry point + ranking primitives."""

from .checkpoint import evaluate_checkpoint, evaluate_ranking
from .eval_hooks import eval_scores, recommended_eval_batch_size
from .evaluate import (
    CandidateProvider,
    Mode,
    ScoresModel,
    clear_eval_cache,
    evaluate,
)
from .pool import CandidatePool
from .ranking import (
    StreamingRankingMetrics,
    compute_ranks,
    ranking_metrics,
    rrf,
    zscore_fusion,
)
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "CandidateProvider",
    "EvalResults",
    "Mode",
    "ScoresModel",
    "StreamingRankingMetrics",
    "clear_eval_cache",
    "compute_ranks",
    "eval_scores",
    "evaluate",
    "evaluate_checkpoint",
    "evaluate_ranking",
    "ranking_metrics",
    "recommended_eval_batch_size",
    "rrf",
    "zscore_fusion",
]
