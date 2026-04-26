"""Evaluation framework: a single class-based evaluator + ranking primitives."""

from .candidates import CandidateSource, Mode, SamplerCandidates
from .checkpoint import evaluate_checkpoint
from .ranking import (
    compute_ranks,
    metrics_from_ranks,
    rrf,
    zscore_fusion,
)
from .ranking_evaluator import RankingEvaluator, RankingResult, ScoreFn
from .results import EvalResults

__all__ = [
    "CandidateSource",
    "EvalResults",
    "Mode",
    "RankingEvaluator",
    "RankingResult",
    "SamplerCandidates",
    "ScoreFn",
    "compute_ranks",
    "evaluate_checkpoint",
    "metrics_from_ranks",
    "rrf",
    "zscore_fusion",
]
