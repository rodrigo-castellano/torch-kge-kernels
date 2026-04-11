"""Ranking and regression losses: pairwise CE, hinge, L2."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


__all__ = [
    "HingeLossRagged",
    "L2LossRagged",
    "PairwiseCrossEntropyRagged",
]
