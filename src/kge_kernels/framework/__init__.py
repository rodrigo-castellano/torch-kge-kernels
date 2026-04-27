"""Shared framework primitives for proof-based NeSy methods.

Implements the six pluggable slots from ``framework.tex`` (resolve,
atom_repr, state_repr, traj_repr, query_repr, select) plus the ``Repr``
container and a reference ``search_and_score`` composition.

Both ``torch-ns-swarm`` and ``DpRL-KGR-swarm`` import the primitives from
here. ``grounder`` provides the ``ResolutionOp`` (its ``ProofState`` and
``ProofEvidence`` dataclasses satisfy the Protocols here via duck typing).
"""
from __future__ import annotations

from .atom_classification import classify_atoms
from .atom_repr import (
    KGEBothAtom,
    KGEEmbedAtom,
    KGEScoreAtom,
    MLPAtom,
    RemappedKGEScoreAtom,
)
from .protocols import (
    AtomRepr,
    QueryRepr,
    ResolutionOp,
    Select,
    StateRepr,
    TrajRepr,
)
from .query_repr import (
    ALL_TRAJECTORY_SCORE_MODES,
    ConceptMaxQueryRepr,
    LogSumExpQueryRepr,
    MaxQueryRepr,
    MeanQueryRepr,
    MLPSumQueryRepr,
    PolicyRolloutQueryRepr,
    ProofScoreQueryRepr,
    SumQueryRepr,
    TrajectoryScoreQueryRepr,
)
from .repr import Repr
from .scorer import ScorerFn, build_scorer, search_and_score
from .select import (
    BeamSelect,
    ExhaustiveSelect,
    GreedySelect,
    SampleSelect,
    StateFactory,
)
from .state_repr import (
    ConcatStateRepr,
    MaxStateRepr,
    MeanStateRepr,
    PhiPsiStateRepr,
    SumStateRepr,
    TNormStateRepr,
)
from .traj_repr import (
    BestCumulativeTrajRepr,
    BestEverStateScoreTrajRepr,
    BestPrefixAvgTrajRepr,
    CumulativeLogTrajRepr,
    FinalStateScoresTrajRepr,
    FinalStepLogScoreTrajRepr,
    MinStepTrajRepr,
    MultiRepr,
    MultiTrajRepr,
    PolicyProductTrajRepr,
    RuleMLPTrajRepr,
    SBRBodyMinTrajRepr,
    TNormTrajRepr,
)
from .types import ProofEvidence, ProofState, SelectInfo

__all__ = [
    # Containers / types
    "Repr",
    "ProofState",
    "ProofEvidence",
    "SelectInfo",
    # Protocols
    "AtomRepr",
    "QueryRepr",
    "ResolutionOp",
    "Select",
    "StateRepr",
    "TrajRepr",
    # Atom classification + reprs
    "classify_atoms",
    "KGEBothAtom",
    "KGEEmbedAtom",
    "KGEScoreAtom",
    "MLPAtom",
    "RemappedKGEScoreAtom",
    # State reprs
    "ConcatStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "PhiPsiStateRepr",
    "SumStateRepr",
    "TNormStateRepr",
    # Traj reprs
    "BestCumulativeTrajRepr",
    "BestEverStateScoreTrajRepr",
    "BestPrefixAvgTrajRepr",
    "CumulativeLogTrajRepr",
    "FinalStateScoresTrajRepr",
    "FinalStepLogScoreTrajRepr",
    "MinStepTrajRepr",
    "MultiRepr",
    "MultiTrajRepr",
    "PolicyProductTrajRepr",
    "RuleMLPTrajRepr",
    "SBRBodyMinTrajRepr",
    "TNormTrajRepr",
    # Query reprs
    "ConceptMaxQueryRepr",
    "LogSumExpQueryRepr",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MeanQueryRepr",
    "PolicyRolloutQueryRepr",
    "ProofScoreQueryRepr",
    "SumQueryRepr",
    "TrajectoryScoreQueryRepr",
    "ALL_TRAJECTORY_SCORE_MODES",
    # Select
    "BeamSelect",
    "ExhaustiveSelect",
    "GreedySelect",
    "SampleSelect",
    "StateFactory",
    # Scorer
    "ScorerFn",
    "build_scorer",
    "search_and_score",
]
