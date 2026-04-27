"""A/B parity tests for PolicyRolloutQueryRepr modes.

Each test reconstructs the original DpRL formula inline (verbatim from
the deleted ``ppo/scoring.py:_RL_SCORERS`` and ``_STEP0_SCORERS`` lookup
tables, plus the special-case branches in ``_score_candidates``) and
asserts the new framework primitive produces identical output.

Catches future formula drift if anyone edits PolicyRolloutQueryRepr.
"""
from __future__ import annotations

import torch

from kge_kernels.framework import PolicyRolloutQueryRepr


def _make_inputs(N: int = 16, seed: int = 0):
    """Fixed random rollout-output tensors."""
    g = torch.Generator().manual_seed(seed)
    success = torch.rand(N, generator=g) > 0.5
    logprobs = torch.randn(N, generator=g) * 2.0 - 5.0     # in [-7, -3] roughly
    endf = torch.rand(N, generator=g) > 0.7
    depths = torch.randint(0, 8, (N,), generator=g)
    p_end = torch.sigmoid(torch.randn(N, generator=g))
    v_pos = torch.randn(N, generator=g)
    v_neg = torch.randn(N, generator=g)
    kge_embed = torch.randn(N, generator=g) * 3.0
    return {
        "success": success, "logprobs": logprobs, "endf": endf, "depths": depths,
        "p_end": p_end, "v_pos": v_pos, "v_neg": v_neg, "kge_embed": kge_embed,
    }


# ───────────────────────────────────────────────────────────────────────
# Pure-RL modes
# ───────────────────────────────────────────────────────────────────────


def test_logprob_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("logprob", fail_penalty=1000.0)
    expected = torch.where(x["success"], x["logprobs"], x["logprobs"] - 1000.0)
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_logprob_endf_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("logprob_endf", fail_penalty=1000.0, endf_penalty=200.0)
    failed = ~x["success"]
    expected = x["logprobs"].clone()
    expected[failed & x["endf"]] -= 200.0
    expected[failed & ~x["endf"]] -= 1000.0
    actual = qr(success=x["success"], logprobs=x["logprobs"], endf=x["endf"])
    assert torch.allclose(actual, expected)


def test_proof_binary_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("proof_binary", alpha=5.0)
    expected = torch.where(
        x["success"],
        torch.full_like(x["logprobs"], 5.0),
        torch.full_like(x["logprobs"], -5.0),
    )
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_proof_bonus_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("proof_bonus", alpha=5.0, fail_penalty=1000.0)
    expected = torch.where(x["success"], x["logprobs"] + 5.0, x["logprobs"] - 1000.0)
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_depth_weighted_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("depth_weighted", beta=2.0, fail_penalty=1000.0)
    d_clamped = x["depths"].float().clamp(min=1.0)
    expected = torch.where(
        x["success"],
        x["logprobs"] + 2.0 * d_clamped,
        x["logprobs"] - 1000.0,
    )
    actual = qr(success=x["success"], logprobs=x["logprobs"], depths=x["depths"])
    assert torch.allclose(actual, expected)


def test_logprob_clipped_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("logprob_clipped", alpha=5.0)
    expected = torch.where(
        x["success"],
        x["logprobs"].clamp(min=-5.0),
        torch.full_like(x["logprobs"], -100.0),
    )
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_proof_rank_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("proof_rank")
    expected = torch.where(x["success"], x["logprobs"] + 1000.0, x["logprobs"])
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_success_only_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("success_only")
    expected = torch.where(x["success"], torch.zeros_like(x["logprobs"]), -torch.ones_like(x["logprobs"]))
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


def test_logprob_scaled_penalty_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("logprob_scaled_penalty", alpha=5.0)
    expected = torch.where(x["success"], x["logprobs"], x["logprobs"] - 5.0)
    assert torch.allclose(qr(success=x["success"], logprobs=x["logprobs"]), expected)


# ───────────────────────────────────────────────────────────────────────
# Step-0 signal modes
# ───────────────────────────────────────────────────────────────────────


def test_value_pos_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("value_pos")
    expected = x["v_pos"]
    actual = qr(success=x["success"], logprobs=x["logprobs"], v_pos=x["v_pos"])
    assert torch.allclose(actual, expected)


def test_end_prob_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("end_prob")
    expected = torch.log1p(-x["p_end"])
    actual = qr(success=x["success"], logprobs=x["logprobs"], p_end=x["p_end"])
    assert torch.allclose(actual, expected)


def test_hybrid_value_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("hybrid_value", beta=2.0, fail_penalty=1000.0)
    expected = torch.where(
        x["success"], x["logprobs"] + 2.0 * x["v_pos"], x["logprobs"] - 1000.0,
    )
    actual = qr(success=x["success"], logprobs=x["logprobs"], v_pos=x["v_pos"])
    assert torch.allclose(actual, expected)


def test_value_diff_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("value_diff")
    expected = x["v_pos"] - x["v_neg"]
    actual = qr(success=x["success"], logprobs=x["logprobs"], v_pos=x["v_pos"], v_neg=x["v_neg"])
    assert torch.allclose(actual, expected)


def test_hybrid_end_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("hybrid_end", alpha=5.0, fail_penalty=1000.0)
    end_bonus = 5.0 * torch.log1p(-x["p_end"])
    expected = torch.where(
        x["success"], x["logprobs"] + end_bonus, x["logprobs"] - 1000.0 + end_bonus,
    )
    actual = qr(success=x["success"], logprobs=x["logprobs"], p_end=x["p_end"])
    assert torch.allclose(actual, expected)


def test_combined_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("combined", alpha=5.0, beta=2.0, fail_penalty=1000.0)
    end_bonus = 5.0 * torch.log1p(-x["p_end"])
    expected = torch.where(
        x["success"],
        x["logprobs"] + end_bonus + 2.0 * x["v_pos"],
        x["logprobs"] - 1000.0 + end_bonus,
    )
    actual = qr(
        success=x["success"], logprobs=x["logprobs"],
        p_end=x["p_end"], v_pos=x["v_pos"],
    )
    assert torch.allclose(actual, expected)


def test_endf_kge_embed_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("endf_kge_embed", alpha=5.0, beta=2.0)
    expected = 5.0 * torch.log1p(-x["p_end"]) + 2.0 * x["kge_embed"]
    actual = qr(
        success=x["success"], logprobs=x["logprobs"],
        p_end=x["p_end"], kge_embed=x["kge_embed"],
    )
    assert torch.allclose(actual, expected)


# ───────────────────────────────────────────────────────────────────────
# KGE-embed modes
# ───────────────────────────────────────────────────────────────────────


def test_kge_embed_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("kge_embed")
    expected = x["kge_embed"]
    actual = qr(success=x["success"], logprobs=x["logprobs"], kge_embed=x["kge_embed"])
    assert torch.allclose(actual, expected)


def test_kge_embed_hybrid_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("kge_embed_hybrid", alpha=5.0, beta=2.0)
    expected = 5.0 * x["success"].float() + 2.0 * x["kge_embed"]
    actual = qr(success=x["success"], logprobs=x["logprobs"], kge_embed=x["kge_embed"])
    assert torch.allclose(actual, expected)


def test_logprob_kge_embed_parity():
    x = _make_inputs()
    qr = PolicyRolloutQueryRepr("logprob_kge_embed", beta=2.0, fail_penalty=1000.0)
    expected = torch.where(
        x["success"],
        x["logprobs"] + 2.0 * x["kge_embed"],
        -1000.0 + 2.0 * x["kge_embed"],
    )
    actual = qr(success=x["success"], logprobs=x["logprobs"], kge_embed=x["kge_embed"])
    assert torch.allclose(actual, expected)


# ───────────────────────────────────────────────────────────────────────
# Validation
# ───────────────────────────────────────────────────────────────────────


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown policy-rollout scoring mode"):
        PolicyRolloutQueryRepr("not_a_mode")


def test_required_input_missing_raises():
    import pytest
    qr = PolicyRolloutQueryRepr("hybrid_value")
    with pytest.raises(ValueError, match="hybrid_value requires v_pos"):
        qr(success=torch.tensor([True]), logprobs=torch.tensor([0.0]))
