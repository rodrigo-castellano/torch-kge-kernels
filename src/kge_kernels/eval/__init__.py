"""Evaluation framework: single ``evaluate()`` entry point + ranking primitives."""

from .checkpoint import evaluate_checkpoint
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
    metrics_from_ranks,
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
    "metrics_from_ranks",
    "recommended_eval_batch_size",
    "rrf",
    "zscore_fusion",
]
