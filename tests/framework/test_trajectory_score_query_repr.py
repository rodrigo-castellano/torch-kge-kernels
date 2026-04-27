"""TrajectoryScoreQueryRepr coverage tests.

For every mode in ``ALL_TRAJECTORY_SCORE_MODES``, build a Repr with the
canonical trajectory-summaries dict, run the dispatcher, and assert the
output is the right shape, finite, and on the input device. Plus a few
sanity tests on the Repr.summaries plumbing and validation.

(The legacy A/B parity tests against ``ProofScoreQueryRepr`` /
``PolicyRolloutQueryRepr`` were removed when those classes were
deleted; their bit-exact equivalence was verified at migration time
in commit 86b416b.)
"""
from __future__ import annotations

import pytest
import torch

from kge_kernels.framework import (
    ALL_TRAJECTORY_SCORE_MODES,
    Repr,
    TrajectoryScoreQueryRepr,
)


def _build_summaries(seed: int = 0):
    """Synthetic trajectory summaries with all standard keys populated."""
    g = torch.Generator().manual_seed(seed)
    B = 8
    success = torch.tensor([True, False, True, True, False, True, False, True])
    depths = torch.tensor([3, 5, 1, 4, 2, 7, 6, 1], dtype=torch.long)
    cumulative_log = torch.empty(B).uniform_(-10.0, 0.0, generator=g)
    min_step_log   = torch.empty(B).uniform_(-3.0, -0.1, generator=g)
    final_step_log = torch.empty(B).uniform_(-3.0, -0.1, generator=g)
    best_cumulative = cumulative_log + torch.empty(B).uniform_(0.0, 1.0, generator=g)
    best_prefix_avg = cumulative_log / depths.float().clamp(min=1.0)
    sbr_body_min = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    final_state_score = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    best_ever_state_score = torch.empty(B).uniform_(0.0, 1.0, generator=g)
    endf = torch.tensor([True, True, False, True, False, False, True, False])
    p_end = torch.empty(B).uniform_(0.0, 0.99, generator=g)
    v_pos = torch.empty(B).uniform_(-1.0, 1.0, generator=g)
    v_neg = torch.empty(B).uniform_(-1.0, 1.0, generator=g)
    kge_embed = torch.empty(B).uniform_(-2.0, 2.0, generator=g)
    cum_value = torch.empty(B).uniform_(-5.0, 5.0, generator=g)
    value_steps = torch.tensor([2, 3, 1, 2, 1, 4, 3, 1], dtype=torch.long)
    return {
        "success": success, "depths": depths,
        "cumulative_log": cumulative_log, "min_step_log": min_step_log,
        "best_cumulative": best_cumulative, "sbr_body_min": sbr_body_min,
        "final_step_log": final_step_log, "final_state_score": final_state_score,
        "best_ever_state_score": best_ever_state_score,
        "best_prefix_avg": best_prefix_avg,
        "endf": endf, "p_end": p_end, "v_pos": v_pos, "v_neg": v_neg,
        "kge_embed": kge_embed,
        "cum_value": cum_value, "value_steps": value_steps,
    }


@pytest.mark.parametrize("mode", sorted(ALL_TRAJECTORY_SCORE_MODES))
def test_each_mode_runs_and_returns_finite_shape(mode):
    """Every mode produces a finite [B] tensor on the canonical inputs."""
    d = _build_summaries(seed=42)
    qr = TrajectoryScoreQueryRepr(mode)
    out = qr(Repr(summaries=d), evidence=None).scores
    assert out.shape == (8,)
    assert torch.isfinite(out).all(), f"mode={mode} produced non-finite values"


def test_repr_summaries_alone_is_valid():
    """A Repr with only summaries (no embeddings/scores) must construct."""
    r = Repr(summaries={"cumulative_log": torch.zeros(4),
                         "success": torch.ones(4, dtype=torch.bool),
                         "depths": torch.ones(4, dtype=torch.long)})
    assert r.has_summaries
    assert not r.has_embeddings
    assert not r.has_scores


def test_unified_missing_required_key_raises():
    """Modes that read a key not in summaries must raise loudly."""
    summaries = {"success": torch.tensor([True, False])}
    qr = TrajectoryScoreQueryRepr("logprob")
    with pytest.raises(KeyError):
        qr(Repr(summaries=summaries), evidence=None)


def test_unified_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown trajectory-scoring mode"):
        TrajectoryScoreQueryRepr("does_not_exist")


def test_unified_missing_summaries_raises():
    qr = TrajectoryScoreQueryRepr("logprob")
    with pytest.raises(ValueError, match="requires traj_repr.summaries"):
        qr(Repr(scores=torch.zeros(4)), evidence=None)
