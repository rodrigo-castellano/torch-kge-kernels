"""Shared PyTorch building blocks for proof-based neural-symbolic KGE methods.

See ``README.md`` for an overview of the framework primitives, KGE models,
losses, data utilities, ranking metrics, and evaluation pipeline shipped
by this package.
"""

__version__ = "0.6.0"

# Hot-path imports: subpackage re-exports that downstream consumers always
# need. These are small and stable — consumers that import any tkk submodule
# trigger this __init__ and pay the import cost, so keep it lean.
#
# Training utilities (training/, training/checkpoints) are intentionally NOT
# eagerly re-exported here: they only matter for standalone KGE training
# pipelines, and eagerly importing them measurably slows the first-batch
# timings of downstream consumers that never touch them (e.g. torch-ns
# SBR/DCR/R2N). Access them explicitly via::
#
#     from kge_kernels.training import TripleDataset, train_kge, ...
#     from kge_kernels.training.checkpoints import save_checkpoint, ...
#
# or via the subpackage attributes ``kge_kernels.training``,
# ``kge_kernels.training.checkpoints``.

# Subpackage re-exports (hot path — imported eagerly)
from . import data as data  # noqa: F401  re-export subpackage
from . import framework as framework  # noqa: F401  re-export subpackage
from . import losses as losses  # noqa: F401  re-export subpackage
from . import models as models  # noqa: F401  re-export subpackage

# Dataset utilities
from .data import (
    TripleExample,
    add_reciprocal_triples,
    build_filter_maps,
    build_relation_domains,
    detect_triple_format,
    encode_split_triples,
    load_dataset_split,
    load_triples,
    load_triples_with_mappings,
    resolve_split_path,
    resolve_train_path,
)

# Evaluation (includes ranking metrics)
from .eval import (
    CandidateSource,
    EvalResults,
    RankingEvaluator,
    RankingResult,
    SamplerCandidates,
    compute_ranks,
    evaluate_checkpoint,
    metrics_from_ranks,
    rrf,
    zscore_fusion,
)

# Framework primitives (atom_repr, state_repr, traj_repr, query_repr, select)
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

# Experiment logging
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

# Loss functions
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

# Raw KGE nn.Module classes + factory
from .models import (
    ComplEx,
    ConvE,
    DistMult,
    KGEModel,
    ModE,
    RotatE,
    TransE,
    TuckER,
    build_model,
    build_training_model,
    kge_default_scorer,
    recommended_eval_batch_size,
)

# Scoring pipeline (sampler, partial, types). KGE scoring itself is the
# model's own ``score(h, r, t)`` method — see kge_kernels.models.
from .scoring import (
    BernoulliSampler,
    LazyPartialScorer,
    PartialScorer,
    Sampler,
    SamplerConfig,
)

__all__ = [
    # Evaluation + ranking
    "CandidateSource",
    "EvalResults",
    "RankingEvaluator",
    "RankingResult",
    "SamplerCandidates",
    "compute_ranks",
    "evaluate_checkpoint",
    "metrics_from_ranks",
    "rrf",
    "zscore_fusion",
    # Scoring pipeline
    "BernoulliSampler",
    "LazyPartialScorer",
    "PartialScorer",
    "Sampler",
    "SamplerConfig",
    # Logging
    "ExperimentSpec",
    "LoggingConfig",
    "ModelConfig",
    "OutputConfig",
    "RegistryConfig",
    "ReportConfig",
    "RunContext",
    "run_logged_experiment",
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
    "load_dataset_split",
    "load_triples",
    "load_triples_with_mappings",
    "resolve_split_path",
    "resolve_train_path",
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
    # KGE models + factory
    "ComplEx",
    "ConvE",
    "DistMult",
    "KGEModel",
    "ModE",
    "RotatE",
    "TransE",
    "TuckER",
    "build_model",
    "build_training_model",
    "kge_default_scorer",
    "recommended_eval_batch_size",
]
