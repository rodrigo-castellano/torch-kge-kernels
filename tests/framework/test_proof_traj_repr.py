"""Lock-down tests for the proof-based reasoning path's TrajRepr family.

After the rule/proof bifurcation, the proof-path primitives
(``CumulativeLogTrajRepr`` / ``MinStepTrajRepr`` / etc.) keep their
incremental ``init/step`` semantics so DpRL's sequential rollouts are
unaffected. These tests pin that behavior down so we notice if a future
refactor breaks the proof path.

Companion to :mod:`test_traj_repr` (which exercises the deprecated
``forward(state_repr, evidence)`` batch interface for the same classes)
and :mod:`test_rule_traj_repr` (which exercises the new
:class:`RuleTrajRepr` pool-iter loops).
"""
from __future__ import annotations

import math

import pytest
import torch

from kge_kernels.framework import (
    CumulativeLogTrajRepr,
    MinStepTrajRepr,
    PolicyProductTrajRepr,
    ProofTrajRepr,
    Repr,
    SBRBodyMinTrajRepr,
    SelectInfo,
    TNormTrajRepr,
    TrajRepr,
)


def test_TrajRepr_alias_is_ProofTrajRepr():
    """``TrajRepr`` is kept as an alias for ``ProofTrajRepr`` during the
    cascade so DpRL's ``from ... import TrajRepr`` continues to work."""
    assert TrajRepr is ProofTrajRepr


@pytest.mark.parametrize(
    "make_traj",
    [
        lambda: TNormTrajRepr("min"),
        lambda: TNormTrajRepr("product"),
        lambda: CumulativeLogTrajRepr(),
        lambda: MinStepTrajRepr(),
        lambda: SBRBodyMinTrajRepr(),
        lambda: PolicyProductTrajRepr(),
        # BestCumulativeTrajRepr is excluded: pre-existing bug — its init
        # builds a 1-D embeddings tensor with leading shape () instead of
        # (B,), violating the Repr leading-shape invariant. Out of scope
        # for this refactor; tracked separately.
    ],
)
def test_proof_traj_has_init_and_step(make_traj):
    """All proof-path ``TrajRepr`` impls expose the incremental contract.

    The ``ProofTrajRepr`` Protocol requires ``init(B, device)`` →
    ``Repr`` and ``step(accum, s_repr, info)`` → ``Repr``. This test
    just smoke-checks both methods are callable; the per-impl semantics
    are exercised by :mod:`test_traj_repr`.
    """
    traj = make_traj()
    assert hasattr(traj, "init"), f"{type(traj).__name__} missing init"
    assert hasattr(traj, "step"), f"{type(traj).__name__} missing step"

    accum = traj.init(2, torch.device("cpu"))
    assert isinstance(accum, Repr)


def test_cumulative_log_init_step_chain_matches_log_sum():
    """DpRL's canonical ``cumulative_log`` accumulator: 3-step chain
    of init + step yields ``Σ log(scores_d)`` as expected."""
    traj = CumulativeLogTrajRepr()
    accum = traj.init(B=1, device=torch.device("cpu"))
    for s in (0.5, 0.5, 0.5):
        accum = traj.step(accum, Repr(scores=torch.tensor([s])), info=None)
    expected = 3 * math.log(0.5)
    assert torch.allclose(accum.scores, torch.tensor([expected]), atol=1e-5)


def test_min_step_init_step_chain_picks_worst_step():
    """Worst single step across the trajectory."""
    traj = MinStepTrajRepr()
    accum = traj.init(B=1, device=torch.device("cpu"))
    for s in (0.9, 0.1, 0.5):
        accum = traj.step(accum, Repr(scores=torch.tensor([s])), info=None)
    expected = math.log(0.1)
    assert torch.allclose(accum.scores, torch.tensor([expected]), atol=1e-5)


def test_policy_product_requires_log_probs_in_info():
    """PPO-flavor ``PolicyProductTrajRepr`` reads SelectInfo.log_probs at each step."""
    traj = PolicyProductTrajRepr()
    accum = traj.init(B=2, device=torch.device("cpu"))
    info = SelectInfo(log_probs=torch.tensor([math.log(0.5), math.log(0.25)]))
    new = traj.step(accum, Repr(scores=torch.zeros(2)), info)
    expected = torch.tensor([math.log(0.5), math.log(0.25)])
    assert torch.allclose(new.scores, expected, atol=1e-5)


def test_policy_product_no_info_raises():
    """``PolicyProductTrajRepr`` must error loudly when info is missing
    — its semantics depend on the per-step policy log-prob."""
    traj = PolicyProductTrajRepr()
    accum = traj.init(B=1, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="log_probs"):
        traj.step(accum, Repr(scores=torch.zeros(1)), info=None)


def test_tnorm_min_init_is_identity_one():
    """T-norm-min identity = 1 (over [0, 1] scores). Verifies init returns
    the correct identity so the first step takes the body score directly."""
    traj = TNormTrajRepr("min")
    accum = traj.init(B=3, device=torch.device("cpu"))
    assert torch.allclose(accum.scores, torch.ones(3))
    # First step with score 0.7 → min(1, 0.7) = 0.7
    after = traj.step(accum, Repr(scores=torch.full((3,), 0.7)), info=None)
    assert torch.allclose(after.scores, torch.full((3,), 0.7))


def test_tnorm_product_init_is_identity_one():
    """T-norm-product identity = 1."""
    traj = TNormTrajRepr("product")
    accum = traj.init(B=3, device=torch.device("cpu"))
    assert torch.allclose(accum.scores, torch.ones(3))
    after = traj.step(accum, Repr(scores=torch.full((3,), 0.7)), info=None)
    assert torch.allclose(after.scores, torch.full((3,), 0.7))
