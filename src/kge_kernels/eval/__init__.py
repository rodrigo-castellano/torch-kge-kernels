"""Evaluation framework: unified evaluator, ranking metrics, corruption pools,
rank fusion, and checkpoint evaluation."""

from .checkpoint import evaluate_checkpoint, evaluate_ranking
from .evaluator import Evaluator, FusionFn, ScorerFn
from .pool import CandidatePool
from .ranking import (
    StreamingRankingMetrics,
    compute_ranks,
    ranking_metrics,
    rrf,
    zscore_fusion,
)
from .results import EvalResults
from .scoring import eval_scores, recommended_eval_batch_size
from .unified import (
    CandidateProvider,
    Mode,
    ScoresModel,
    clear_eval_cache,
    evaluate,
)

__all__ = [
    "CandidatePool",
    "CandidateProvider",
    "EvalResults",
    "Evaluator",
    "FusionFn",
    "Mode",
    "ScorerFn",
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
