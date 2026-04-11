"""Ranking and regression losses: pairwise CE, hinge, L2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PairwiseCrossEntropyRagged(nn.Module):
    """Pairwise cross-entropy loss for ranking.

    Positives are encouraged to have high logits (``logsigmoid(y_pred)``) and
    negatives to have low logits (``logsigmoid(-y_pred)``). ``balance_negatives``
    rescales each side by row-wise counts.
    """

    def __init__(
        self, balance_negatives: bool = False, from_logits: bool = True
    ) -> None:
        super().__init__()
        self.balance_negatives = balance_negatives
        # ``from_logits`` is accepted for interface compatibility; the
        # implementation always assumes logits (logsigmoid).
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pos_loss = -torch.sum(
            torch.where(
                y_true == 1, F.logsigmoid(y_pred), torch.zeros_like(y_pred)
            ),
            dim=-1,
            keepdim=True,
        )
        neg_loss = -torch.sum(
            torch.where(
                y_true == 0, F.logsigmoid(-y_pred), torch.zeros_like(y_pred)
            ),
            dim=-1,
            keepdim=True,
        )
        if self.balance_negatives:
            num_pos = (y_true == 1).sum(dim=-1, keepdim=True).to(y_pred.dtype)
            num_neg = (y_true == 0).sum(dim=-1, keepdim=True).to(y_pred.dtype)
            pos_loss = pos_loss / (num_pos + 1e-8)
            neg_loss = neg_loss / (num_neg + 1e-8)
        return (pos_loss + neg_loss).squeeze(-1).mean()


class HingeLossRagged(nn.Module):
    """Margin hinge loss: ``max(0, gamma + max(neg) - mean(pos))``."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_scores = torch.where(pos_mask, y_pred, torch.zeros_like(y_pred))
        neg_scores = torch.where(
            neg_mask, y_pred, torch.full_like(y_pred, float("-inf"))
        )

        pos_mean = pos_scores.sum(dim=-1) / (
            pos_mask.sum(dim=-1).to(y_pred.dtype) + 1e-8
        )
        neg_max = neg_scores.max(dim=-1).values

        loss = F.relu(self.gamma + neg_max - pos_mean)
        return loss.mean()


class L2LossRagged(nn.Module):
    """Mean-squared-error wrapper (kept for factory-name compatibility)."""

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true)


class NSSALoss(nn.Module):
    """Self-adversarial negative-sampling loss (Sun et al., 2019 — RotatE).

    Maximizes the log-sigmoid of positive scores and the log-sigmoid of
    negated negative scores. When ``adv_temp > 0`` the negatives are
    weighted by a stop-gradient softmax over ``adv_temp * neg_scores``,
    which self-adversarially upweights hard negatives.

    This is the loss used by RotatE / ComplEx / TransE in DpRL's
    standalone KGE training pipeline. Input format: ``pos_scores`` has
    shape ``[B]`` and ``neg_scores`` has shape ``[B * neg_ratio]`` (it
    is reshaped internally to ``[B, neg_ratio]`` for the softmax step).

    Args:
        adv_temp: Self-adversarial temperature. ``0`` recovers plain
            uniform averaging over negatives. Typical values: 0.5–2.0.
        neg_ratio: Number of negatives per positive. Must match what the
            sampler produces; the loss uses it only to reshape.
    """

    def __init__(self, adv_temp: float = 0.0, neg_ratio: int = 1) -> None:
        super().__init__()
        self.adv_temp = adv_temp
        self.neg_ratio = neg_ratio

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        pos_loss = -F.logsigmoid(pos_scores).mean()
        neg_flat = neg_scores.view(-1, self.neg_ratio)

        if self.adv_temp > 0:
            with torch.no_grad():
                weights = torch.softmax(self.adv_temp * neg_flat, dim=1)
            neg_loss = (weights * (-F.logsigmoid(-neg_flat))).sum(dim=1).mean()
        else:
            neg_loss = -F.logsigmoid(-neg_flat).mean()

        return pos_loss + neg_loss


__all__ = [
    "HingeLossRagged",
    "L2LossRagged",
    "NSSALoss",
    "PairwiseCrossEntropyRagged",
]
