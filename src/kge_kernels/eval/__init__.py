"""Evaluation framework: a single class-based evaluator + ranking primitives."""

from .candidates import CandidateSource, Mode, SamplerCandidates
from .checkpoint import evaluate_checkpoint
from .eval_hooks import kge_default_scorer, recommended_eval_batch_size
from .pool import CandidatePool
from .ranking import (
    StreamingRankingMetrics,
    compute_ranks,
    metrics_from_ranks,
    rrf,
    zscore_fusion,
)
from .ranking_evaluator import RankingEvaluator, RankingResult, ScoreFn
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "CandidateSource",
    "EvalResults",
    "Mode",
    "RankingEvaluator",
    "RankingResult",
    "SamplerCandidates",
    "ScoreFn",
    "StreamingRankingMetrics",
    "compute_ranks",
    "evaluate_checkpoint",
    "kge_default_scorer",
    "metrics_from_ranks",
    "recommended_eval_batch_size",
    "rrf",
    "zscore_fusion",
]
