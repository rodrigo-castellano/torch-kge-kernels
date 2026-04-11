"""Shared PyTorch building blocks for proof-based neural-symbolic KGE methods.

See ``README.md`` for an overview of the framework primitives, KGE models,
losses, data utilities, ranking metrics, and evaluation pipeline shipped
by this package.
"""

__version__ = "0.4.3"

# Hot-path imports: tkk's original surface (kernels + framework primitives +
# models + losses + data + ranking + eval + logging + sampler + scoring).
# These are small and stable — consumers that import any tkk submodule
# trigger this __init__ and pay the import cost, so keep it lean.
#
# Phase 9 KGE-training utilities (training/, checkpoints, filtered eval)
# are intentionally NOT eagerly re-exported here: they only matter for
# standalone KGE training pipelines, and eagerly importing them measurably
# slows the first-batch timings of downstream consumers that never touch
# them (e.g. torch-ns SBR/DCR/R2N). Access them explicitly via::
#
#     from kge_kernels.training import TripleDataset, train_kge, ...
#     from kge_kernels.checkpoints import save_checkpoint, ...
#     from kge_kernels.eval import evaluate_filtered_ranking
#
# or via the subpackage attributes ``kge_kernels.training``,
# ``kge_kernels.checkpoints``, ``kge_kernels.eval``.

# Dataset utilities
from . import data as data  # noqa: F401  re-export subpackage

# Framework primitives (atom_repr, state_repr, traj_repr, query_repr, select)
from . import framework as framework  # noqa: F401  re-export subpackage

# Loss functions
from . import losses as losses  # noqa: F401  re-export subpackage

# Raw KGE nn.Module classes
from . import models as models  # noqa: F401  re-export subpackage
from .adapter import (
    apply_masks,
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
)
from .adapter import precompute_partial_scores as precompute_partial_scores
from .data import (
    TripleExample,
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    detect_triple_format,
    encode_split_triples,
    load_triples,
    load_triples_with_mappings,
)
from .eval import (
    CandidatePool,
    EvalResults,
    Evaluator,
    rrf,
    zscore_fusion,
)
from .framework import (
    AtomRepr,
    BeamSelect,
    BestCumulativeTrajRepr,
    ConcatStateRepr,
    ConceptMaxQueryRepr,
    CumulativeLogTrajRepr,
    ExhaustiveSelect,
    GreedySelect,
    KGEBothAtom,
    KGEEmbedAtom,
    KGEScoreAtom,
    LogSumExpQueryRepr,
    MaxQueryRepr,
    MaxStateRepr,
    MeanQueryRepr,
    MeanStateRepr,
    MinStepTrajRepr,
    MLPAtom,
    MLPSumQueryRepr,
    PolicyProductTrajRepr,
    ProofEvidence,
    ProofState,
    QueryRepr,
    Repr,
    ResolutionOp,
    SampleSelect,
    SBRBodyMinTrajRepr,
    ScorerFn,
    Select,
    SelectInfo,
    StateRepr,
    SumQueryRepr,
    SumStateRepr,
    TNormStateRepr,
    TNormTrajRepr,
    TrajRepr,
    build_scorer,
    search_and_score,
)
from .logging import (
    ExperimentSpec,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    RegistryConfig,
    ReportConfig,
    RunContext,
)
from .logging import (
    run_experiment as run_logged_experiment,
)
from .losses import (
    BinaryCrossEntropyRagged,
    BinaryCrossEntropyWithMask,
    CategoricalCrossEntropyRagged,
    HingeLossRagged,
    L2LossRagged,
    NSSALoss,
    PairwiseCrossEntropyRagged,
    WeightedBinaryCrossEntropy,
    build_loss,
)
from .models import (
    ComplEx,
    ConvE,
    DistMult,
    KGEModel,
    ModE,
    RotatE,
    TransE,
    TuckER,
)
from .partial import LazyPartialScorer, score_partial_atoms
from .ranking import (
    StreamingRankingMetrics,
    ranking_metrics,
    ranks_from_labeled_predictions,
    ranks_from_scores,
    ranks_from_scores_matrix,
)
from .sampler import Sampler, SamplerConfig, corrupt
from .scoring import (
    KGEBackend,
    score,
)
from .types import CorruptionOutput, ScoreOutput
from .utils import compute_bernoulli_probs

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
    "ranks_from_labeled_predictions",
    "ranks_from_scores",
    "ranks_from_scores_matrix",
    "run_logged_experiment",
    "rrf",
    "score",
    "LazyPartialScorer",
    "score_partial_atoms",
    "StreamingRankingMetrics",
    "zscore_fusion",
    # Losses
    "BinaryCrossEntropyRagged",
    "BinaryCrossEntropyWithMask",
    "CategoricalCrossEntropyRagged",
    "HingeLossRagged",
    "L2LossRagged",
    "NSSALoss",
    "PairwiseCrossEntropyRagged",
    "WeightedBinaryCrossEntropy",
    "build_loss",
    # Data
    "TripleExample",
    "add_reciprocal_triples",
    "build_filter_maps",
    "build_relation_domains",
    "detect_triple_format",
    "encode_split_triples",
    "load_triples",
    "load_triples_with_mappings",
    # Framework primitives
    "AtomRepr",
    "BeamSelect",
    "BestCumulativeTrajRepr",
    "ConcatStateRepr",
    "ConceptMaxQueryRepr",
    "CumulativeLogTrajRepr",
    "ExhaustiveSelect",
    "GreedySelect",
    "KGEBothAtom",
    "KGEEmbedAtom",
    "KGEScoreAtom",
    "LogSumExpQueryRepr",
    "MLPAtom",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MaxStateRepr",
    "MeanQueryRepr",
    "MeanStateRepr",
    "MinStepTrajRepr",
    "PolicyProductTrajRepr",
    "ProofEvidence",
    "ProofState",
    "QueryRepr",
    "Repr",
    "ResolutionOp",
    "SBRBodyMinTrajRepr",
    "SampleSelect",
    "ScorerFn",
    "Select",
    "SelectInfo",
    "StateRepr",
    "SumQueryRepr",
    "SumStateRepr",
    "TNormStateRepr",
    "TNormTrajRepr",
    "TrajRepr",
    "build_scorer",
    "search_and_score",
    # KGE models
    "ComplEx",
    "ConvE",
    "DistMult",
    "KGEModel",
    "ModE",
    "RotatE",
    "TransE",
    "TuckER",
]
