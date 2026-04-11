"""Evaluation framework: corruption pools, rank fusion, filtered ranking,
and metrics."""

from .evaluator import Evaluator, FusionFn, ScorerFn
from .filtered import evaluate_filtered_ranking
from .fusion import rrf, zscore_fusion
from .pool import CandidatePool
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "EvalResults",
    "Evaluator",
    "FusionFn",
    "ScorerFn",
    "evaluate_filtered_ranking",
    "rrf",
    "zscore_fusion",
]
