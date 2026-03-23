"""Evaluation framework: corruption pools, rank fusion, and metrics."""

from .evaluator import Evaluator, FusionFn, ScorerFn
from .fusion import rrf, zscore_fusion
from .pool import CandidatePool
from .results import EvalResults

__all__ = [
    "CandidatePool",
    "EvalResults",
    "Evaluator",
    "FusionFn",
    "ScorerFn",
    "rrf",
    "zscore_fusion",
]
