"""Shared PyTorch kernels for KGE sampling and scoring."""

from .logging import (
    ExperimentSpec,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    RegistryConfig,
    ReportConfig,
    RunContext,
    run_experiment as run_logged_experiment,
)
from .adapter import (
    apply_masks,
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
)
from .adapter import precompute_partial_scores as precompute_partial_scores
from .partial import LazyPartialScorer, score_partial_atoms
from .ranking import ranking_metrics, ranks_from_scores, ranks_from_scores_matrix
from .sampler import Sampler, SamplerConfig, corrupt
from .scoring import (
    KGEBackend,
    score,
)
from .types import CorruptionOutput, ScoreOutput
from .utils import compute_bernoulli_probs
from .eval import CandidatePool, EvalResults, Evaluator, rrf, zscore_fusion

__all__ = [
    "CandidatePool",
    "EvalResults",
    "Evaluator",
    "Sampler",
    "SamplerConfig",
    "KGEBackend",
    "LoggingConfig",
    "ModelConfig",
    "OutputConfig",
    "RegistryConfig",
    "ReportConfig",
    "RunContext",
    "ExperimentSpec",
    "CorruptionOutput",
    "ScoreOutput",
    "apply_masks",
    "build_backend",
    "compute_bernoulli_probs",
    "corrupt",
    "kge_score_all_heads",
    "kge_score_all_tails",
    "kge_score_triples",
    "precompute_partial_scores",
    "ranking_metrics",
    "ranks_from_scores",
    "ranks_from_scores_matrix",
    "run_logged_experiment",
    "rrf",
    "score",
    "LazyPartialScorer",
    "score_partial_atoms",
    "zscore_fusion",
]
