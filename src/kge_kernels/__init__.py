"""Shared PyTorch building blocks for proof-based neural-symbolic KGE methods.

See ``README.md`` for an overview of the framework primitives, KGE models,
losses, data utilities, ranking metrics, and evaluation pipeline shipped
by this package.
"""

__version__ = "0.4.0"

# Dataset utilities
from . import data as data  # noqa: F401  re-export subpackage

# Framework primitives (atom_repr, state_repr, traj_repr, query_repr, select)
from . import framework as framework  # noqa: F401  re-export subpackage

# Loss functions
from . import losses as losses  # noqa: F401  re-export subpackage

# Raw KGE nn.Module classes
from . import models as models  # noqa: F401  re-export subpackage
from . import training as training  # noqa: F401  re-export subpackage
from .adapter import (
    apply_masks,
    build_backend,
    kge_score_all_heads,
    kge_score_all_tails,
    kge_score_triples,
)
from .adapter import precompute_partial_scores as precompute_partial_scores
from .checkpoints import (
    load_checkpoint,
    model_state_dict,
    normalize_loaded_state_dict,
    save_checkpoint,
    save_state_dict,
    unwrap_model,
    write_json_payload,
)
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
    evaluate_filtered_ranking,
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
from .training import (
    KGETrainConfig,
    OnEpochEnd,
    TripleDataset,
    make_cosine_warmup_scheduler,
    set_seed,
    train_kge,
    wrap_model_for_training,
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
    # Training
    "KGETrainConfig",
    "OnEpochEnd",
    "TripleDataset",
    "make_cosine_warmup_scheduler",
    "set_seed",
    "train_kge",
    "wrap_model_for_training",
    # Checkpoints
    "load_checkpoint",
    "model_state_dict",
    "normalize_loaded_state_dict",
    "save_checkpoint",
    "save_state_dict",
    "unwrap_model",
    "write_json_payload",
    # Filtered ranking eval
    "evaluate_filtered_ranking",
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
