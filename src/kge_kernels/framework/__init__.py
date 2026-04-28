"""Shared framework primitives for proof-based NeSy methods.

Implements the six pluggable slots from ``framework.tex`` (resolve,
atom_repr, state_repr, traj_repr, query_repr, select) plus the ``Repr``
container.

The canonical 6-tuple scoring loop lives on
:class:`kge_kernels.search.ProofScorer`; subclasses (e.g., DpRL's
PPO/Lookahead proof scorers) override it for specialized rollouts.

Both ``torch-ns-swarm`` and ``DpRL-KGR-swarm`` import the primitives from
here. ``grounder`` provides the ``ResolutionOp`` (its ``ProofState`` and
``ProofEvidence`` dataclasses satisfy the Protocols here via duck typing).
"""
from __future__ import annotations

from .atom_classification import classify_atoms
from .protocols import (
    AtomRepr,
    QueryRepr,
    ResolutionOp,
    Select,
    StateRepr,
    TrajRepr,
)
from .repr import Repr
from .repr_atom import (
    KGEBothAtom,
    KGEEmbedAtom,
    KGEPairAtom,
    KGEScoreAtom,
    MLPAtom,
    RemappedKGEScoreAtom,
)
from .repr_query import (
    ALL_TRAJECTORY_SCORE_MODES,
    ConceptMaxQueryRepr,
    LogSumExpQueryRepr,
    MaxQueryRepr,
    MeanQueryRepr,
    MLPSumQueryRepr,
    SumQueryRepr,
    TrajectoryScoreQueryRepr,
)
from .repr_state import (
    ConcatStateRepr,
    GatedTNormStateRepr,
    MaxStateRepr,
    MeanStateRepr,
    PhiPsiStateRepr,
    RuleWeightedStateRepr,
    SumStateRepr,
    TNormStateRepr,
)
from .repr_traj import (
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
from .select import (
    BeamSelect,
    ExhaustiveSelect,
    GreedySelect,
    SampleSelect,
    StateFactory,
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
    "KGEPairAtom",
    "KGEScoreAtom",
    "MLPAtom",
    "RemappedKGEScoreAtom",
    # State reprs
    "ConcatStateRepr",
    "GatedTNormStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "PhiPsiStateRepr",
    "RuleWeightedStateRepr",
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
    "SumQueryRepr",
    "TrajectoryScoreQueryRepr",
    "ALL_TRAJECTORY_SCORE_MODES",
    # Select
    "BeamSelect",
    "ExhaustiveSelect",
    "GreedySelect",
    "SampleSelect",
    "StateFactory",
]
