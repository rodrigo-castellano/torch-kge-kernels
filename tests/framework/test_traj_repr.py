"""TrajRepr forward ≡ init+step*D parity tests + correctness tests."""
from __future__ import annotations

import math

import pytest
import torch

from kge_kernels.framework import (
    BestCumulativeTrajRepr,
    CumulativeLogTrajRepr,
    MinStepTrajRepr,
    PolicyProductTrajRepr,
    Repr,
    SBRBodyMinTrajRepr,
    SelectInfo,
    TNormTrajRepr,
)

from .conftest import make_structured_evidence


@pytest.mark.parametrize(
    "cls,tnorm",
    [
        (lambda: TNormTrajRepr("min"), None),
        (lambda: TNormTrajRepr("product"), None),
        (lambda: CumulativeLogTrajRepr(), None),
        (lambda: MinStepTrajRepr(), None),
        (lambda: SBRBodyMinTrajRepr(), None),
    ],
)
def test_forward_equals_incremental(cls, tnorm):
    """forward(state_repr_full, evidence) ≡ init + step*D for the same data."""
    traj = cls()
    B, C, D = 2, 1, 4
    ev = make_structured_evidence(B=B, C=C, D=D, M=2)
    # Per-state scores [B, C, D]
    scores = torch.tensor(
        [
            [[0.2, 0.3, 0.4, 0.5]],
            [[0.9, 0.8, 0.7, 0.6]],
        ],
        dtype=torch.float32,
    )
    full = traj(Repr(scores=scores), ev)

    # Run incrementally on the b=0, c=0 path. Reduce the [B, C] forward
    # output along C to compare apples-to-apples.
    accum = traj.init(B, scores.device)
    for d in range(D):
        step_scores = scores[:, 0, d]   # [B]
        accum = traj.step(accum, Repr(scores=step_scores), info=None)

    forward_per_path = full.scores[:, 0]  # [B]
    assert torch.allclose(accum.scores, forward_per_path, atol=1e-5), (
        f"forward {forward_per_path} vs incremental {accum.scores}"
    )


def test_cumulative_log_correctness():
    traj = CumulativeLogTrajRepr()
    ev = make_structured_evidence(B=1, C=1, D=3, M=1)
    scores = torch.tensor([[[0.5, 0.5, 0.5]]])
    out = traj(Repr(scores=scores), ev)
    expected = 3 * math.log(0.5)
    assert torch.allclose(out.scores[0, 0], torch.tensor(expected), atol=1e-5)


def test_min_step_correctness():
    traj = MinStepTrajRepr()
    ev = make_structured_evidence(B=1, C=1, D=3, M=1)
    scores = torch.tensor([[[0.9, 0.1, 0.5]]])
    out = traj(Repr(scores=scores), ev)
    assert torch.allclose(out.scores[0, 0], torch.tensor(math.log(0.1)), atol=1e-5)


def test_best_cumulative_picks_best_prefix():
    traj = BestCumulativeTrajRepr()
    ev = make_structured_evidence(B=1, C=1, D=3, M=1)
    # log(0.9) ≈ -0.105, then log(0.1) makes it ~ -2.4. Best prefix is just step 0.
    scores = torch.tensor([[[0.9, 0.1, 0.5]]])
    out = traj(Repr(scores=scores), ev)
    expected = math.log(0.9)
    assert torch.allclose(out.scores[0, 0], torch.tensor(expected), atol=1e-5)


def test_policy_product_traj_requires_log_probs():
    traj = PolicyProductTrajRepr()
    accum = traj.init(2, torch.device("cpu"))
    info = SelectInfo(log_probs=torch.tensor([math.log(0.5), math.log(0.25)]))
    new = traj.step(accum, Repr(scores=torch.zeros(2)), info)
    expected = torch.tensor([math.log(0.5), math.log(0.25)])
    assert torch.allclose(new.scores, expected, atol=1e-5)


def test_policy_product_no_info_raises():
    traj = PolicyProductTrajRepr()
    accum = traj.init(1, torch.device("cpu"))
    with pytest.raises(ValueError, match="log_probs"):
        traj.step(accum, Repr(scores=torch.zeros(1)), info=None)
