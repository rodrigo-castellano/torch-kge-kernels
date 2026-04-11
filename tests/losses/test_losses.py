"""Tests for kge_kernels.losses."""
from __future__ import annotations

import math

import pytest
import torch

from kge_kernels.losses import (
    BinaryCrossEntropyRagged,
    BinaryCrossEntropyWithMask,
    CategoricalCrossEntropyRagged,
    HingeLossRagged,
    L2LossRagged,
    PairwiseCrossEntropyRagged,
    WeightedBinaryCrossEntropy,
    build_loss,
)

# ═══════════════════════════════════════════════════════════════════════
# BinaryCrossEntropyWithMask
# ═══════════════════════════════════════════════════════════════════════


def test_bce_with_mask_ignores_padding():
    loss = BinaryCrossEntropyWithMask()
    y_pred = torch.tensor([[0.1, 0.9, 0.5]])
    y_true = torch.tensor([[0.0, 1.0, -1.0]])  # last slot padded
    out = loss(y_pred, y_true)
    # Should equal mean BCE over the two unpadded entries
    expected = 0.5 * (-math.log(1 - 0.1) + -math.log(0.9))
    assert torch.allclose(out, torch.tensor(expected), atol=1e-5)


def test_bce_with_mask_all_padded_is_zero():
    loss = BinaryCrossEntropyWithMask()
    y_pred = torch.tensor([[0.5, 0.5]])
    y_true = torch.tensor([[-1.0, -1.0]])
    out = loss(y_pred, y_true)
    assert torch.allclose(out, torch.tensor(0.0))


def test_bce_with_mask_from_logits():
    loss = BinaryCrossEntropyWithMask(from_logits=True)
    logits = torch.tensor([[0.0, 2.0]])
    labels = torch.tensor([[0.0, 1.0]])
    out = loss(logits, labels)
    # sanity: finite, positive
    assert out.isfinite().item() and out.item() > 0


# ═══════════════════════════════════════════════════════════════════════
# WeightedBinaryCrossEntropy
# ═══════════════════════════════════════════════════════════════════════


def test_weighted_bce_equal_weights_matches_unweighted():
    weighted = WeightedBinaryCrossEntropy(weight_0=1.0, weight_1=1.0)
    y_pred = torch.tensor([0.2, 0.8, 0.5, 0.3])
    y_true = torch.tensor([0.0, 1.0, 1.0, 0.0])
    out = weighted(y_pred, y_true)
    baseline = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    assert torch.allclose(out, baseline, atol=1e-5)


def test_weighted_bce_heavier_positive_weight_increases_loss_on_missed_positive():
    weighted = WeightedBinaryCrossEntropy(weight_0=1.0, weight_1=10.0)
    y_pred = torch.tensor([0.1])
    y_true = torch.tensor([1.0])
    heavy_out = weighted(y_pred, y_true)
    baseline = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    assert heavy_out.item() > baseline.item()


# ═══════════════════════════════════════════════════════════════════════
# BinaryCrossEntropyRagged
# ═══════════════════════════════════════════════════════════════════════


def test_bce_ragged_unbalanced():
    loss = BinaryCrossEntropyRagged(balance_negatives=False)
    y_pred = torch.tensor([[0.2, 0.9], [0.1, 0.7]])
    y_true = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    out = loss(y_pred, y_true)
    baseline = torch.nn.functional.binary_cross_entropy(y_pred, y_true)
    assert torch.allclose(out, baseline, atol=1e-5)


def test_bce_ragged_balanced_equalizes_classes():
    loss = BinaryCrossEntropyRagged(balance_negatives=True)
    y_pred = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    y_true = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = loss(y_pred, y_true)
    assert out.isfinite().item() and out.item() > 0


# ═══════════════════════════════════════════════════════════════════════
# PairwiseCrossEntropyRagged
# ═══════════════════════════════════════════════════════════════════════


def test_pairwise_ce_positive_score_minimizes_loss():
    loss = PairwiseCrossEntropyRagged(balance_negatives=False)
    y_true = torch.tensor([[1.0, 0.0, 0.0]])
    # Case A: positive has high logit, negatives low → small loss
    y_pred_good = torch.tensor([[5.0, -5.0, -5.0]])
    good = loss(y_pred_good, y_true)
    # Case B: positive low, negatives high → large loss
    y_pred_bad = torch.tensor([[-5.0, 5.0, 5.0]])
    bad = loss(y_pred_bad, y_true)
    assert bad.item() > good.item() * 10


def test_pairwise_ce_balanced_variant_runs():
    loss = PairwiseCrossEntropyRagged(balance_negatives=True)
    y_pred = torch.tensor([[1.0, -0.5, -0.2]])
    y_true = torch.tensor([[1.0, 0.0, 0.0]])
    out = loss(y_pred, y_true)
    assert out.isfinite().item()


# ═══════════════════════════════════════════════════════════════════════
# HingeLossRagged
# ═══════════════════════════════════════════════════════════════════════


def test_hinge_zero_when_margin_satisfied():
    loss = HingeLossRagged(gamma=1.0)
    y_pred = torch.tensor([[5.0, 0.0, 0.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0]])
    out = loss(y_pred, y_true)
    # pos_mean = 5, neg_max = 0, margin = 1; loss = relu(1 + 0 - 5) = 0
    assert torch.allclose(out, torch.tensor(0.0))


def test_hinge_positive_when_margin_violated():
    loss = HingeLossRagged(gamma=1.0)
    y_pred = torch.tensor([[0.0, 2.0, 0.0]])
    y_true = torch.tensor([[1.0, 0.0, 0.0]])
    out = loss(y_pred, y_true)
    # pos_mean = 0, neg_max = 2, margin = 1; loss = relu(1 + 2 - 0) = 3
    assert torch.allclose(out, torch.tensor(3.0))


# ═══════════════════════════════════════════════════════════════════════
# L2 / Categorical
# ═══════════════════════════════════════════════════════════════════════


def test_l2_loss_matches_mse():
    loss = L2LossRagged()
    y_pred = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([1.0, 2.0, 5.0])
    out = loss(y_pred, y_true)
    assert torch.allclose(out, torch.tensor(4.0 / 3.0))


def test_categorical_ce_one_hot():
    loss = CategoricalCrossEntropyRagged(from_logits=True)
    logits = torch.tensor([[2.0, 0.1, 0.1]])
    target = torch.tensor([[1.0, 0.0, 0.0]])
    out = loss(logits, target)
    assert out.isfinite().item() and out.item() < 1.0


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


def test_build_loss_canonical_names():
    cases = {
        "bce_masked": BinaryCrossEntropyWithMask,
        "bce_weighted": WeightedBinaryCrossEntropy,
        "bce_ragged": BinaryCrossEntropyRagged,
        "pairwise": PairwiseCrossEntropyRagged,
        "categorical": CategoricalCrossEntropyRagged,
        "hinge": HingeLossRagged,
        "l2": L2LossRagged,
    }
    for name, cls in cases.items():
        obj = build_loss(name)
        assert isinstance(obj, cls), f"{name} → {type(obj).__name__}"


def test_build_loss_legacy_aliases():
    assert isinstance(build_loss("binary_crossentropy"), BinaryCrossEntropyRagged)
    bal = build_loss("balanced_binary_crossentropy")
    assert isinstance(bal, BinaryCrossEntropyRagged) and bal.balance_negatives
    pair_bal = build_loss("balanced_pairwise_crossentropy")
    assert isinstance(pair_bal, PairwiseCrossEntropyRagged)
    assert pair_bal.balance_negatives


def test_build_loss_bce_balanced_shortcut():
    obj = build_loss("bce_balanced")
    assert isinstance(obj, BinaryCrossEntropyRagged)
    assert obj.balance_negatives


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss("quadratic_sparkle")


# ═══════════════════════════════════════════════════════════════════════
# NSSALoss (self-adversarial negative sampling)
# ═══════════════════════════════════════════════════════════════════════


def test_nssa_loss_factory():
    from kge_kernels.losses import NSSALoss

    obj = build_loss("nssa", adv_temp=1.0, neg_ratio=5)
    assert isinstance(obj, NSSALoss)
    assert obj.adv_temp == 1.0
    assert obj.neg_ratio == 5


def test_nssa_loss_runs_zero_temp():
    from kge_kernels.losses import NSSALoss

    loss = NSSALoss(adv_temp=0.0, neg_ratio=3)
    pos = torch.tensor([2.0, 1.0, 0.5])          # [B=3]
    neg = torch.randn(9)                          # [B*neg_ratio=9]
    out = loss(pos, neg)
    assert out.isfinite().item()
    assert out.item() > 0


def test_nssa_loss_advtemp_weights_hard_negatives():
    """With adv_temp > 0, harder (higher-score) negatives should contribute more."""
    from kge_kernels.losses import NSSALoss

    loss_hot = NSSALoss(adv_temp=2.0, neg_ratio=3)
    loss_cold = NSSALoss(adv_temp=0.0, neg_ratio=3)

    pos = torch.tensor([1.0])
    # One very hard negative mixed with two easy ones
    neg = torch.tensor([5.0, -5.0, -5.0])

    hot = loss_hot(pos, neg).item()
    cold = loss_cold(pos, neg).item()
    # Self-adversarial reweighting makes the hard negative dominate,
    # which should produce a larger loss than plain averaging.
    assert hot > cold
