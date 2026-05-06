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
    ProofTrajRepr,
    QueryRepr,
    ResolutionOp,
    RuleStateRepr,
    RuleTrajRepr,
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
from .repr_query_pool import LookupAtPool, OutputLayerAtPool
from .repr_state import (
    ClusteredFilterSignStateRepr,
    ConcatStateRepr,
    FilterSignStateRepr,
    GatedTNormStateRepr,
    MaxStateRepr,
    MeanStateRepr,
    PhiPsiStateRepr,
    RuleWeightedStateRepr,
    SumStateRepr,
    TNormStateRepr,
)
from .repr_state_rule import FilterSignRuleState, MinRuleState, RuleMLPState
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
from .repr_traj_rule import DCRPoolLoop, MinPoolLoop, RuleMLPPoolLoop
from .select import (
    BeamSelect,
    ExhaustiveSelect,
    GreedySelect,
    SampleSelect,
    StateFactory,
)
from .types import (
    FiringsTensors,
    ProofEvidence,
    ProofState,
    RuleGroundings,
    SelectInfo,
    build_firings_from_rule_groundings,
)

__all__ = [
    # Containers / types
    "Repr",
    "ProofState",
    "ProofEvidence",
    "RuleGroundings",
    "FiringsTensors",
    "SelectInfo",
    "build_firings_from_rule_groundings",
    # Protocols
    "AtomRepr",
    "ProofTrajRepr",
    "QueryRepr",
    "ResolutionOp",
    "RuleStateRepr",
    "RuleTrajRepr",
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
    # State reprs (proof path — per grounding tree)
    "ClusteredFilterSignStateRepr",
    "ConcatStateRepr",
    "FilterSignStateRepr",
    "GatedTNormStateRepr",
    "MaxStateRepr",
    "MeanStateRepr",
    "PhiPsiStateRepr",
    "RuleWeightedStateRepr",
    "SumStateRepr",
    "TNormStateRepr",
    # State reprs (rule path — per firing)
    "FilterSignRuleState",
    "MinRuleState",
    "RuleMLPState",
    # Traj reprs (proof path — sequential init/step)
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
    # Traj reprs (rule path — K-iter pool loop)
    "DCRPoolLoop",
    "MinPoolLoop",
    "RuleMLPPoolLoop",
    # Query reprs (proof path)
    "ConceptMaxQueryRepr",
    "LogSumExpQueryRepr",
    "MLPSumQueryRepr",
    "MaxQueryRepr",
    "MeanQueryRepr",
    "SumQueryRepr",
    "TrajectoryScoreQueryRepr",
    "ALL_TRAJECTORY_SCORE_MODES",
    # Query reprs (rule path — pool gather)
    "LookupAtPool",
    "OutputLayerAtPool",
    # Select
    "BeamSelect",
    "ExhaustiveSelect",
    "GreedySelect",
    "SampleSelect",
    "StateFactory",
]
