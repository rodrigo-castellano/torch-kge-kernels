"""Classification losses: BCE variants and categorical cross-entropy."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BinaryCrossEntropyWithMask(nn.Module):
    """Binary cross entropy with a ``y_true == -1`` padding mask.

    Padded entries are excluded from both the numerator and denominator.
    """

    def __init__(self, from_logits: bool = False) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        mask = y_true >= 0
        y_true_masked = torch.where(mask, y_true, torch.zeros_like(y_true))

        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, y_true_masked, reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(y_pred, y_true_masked, reduction="none")

        masked_loss = loss * mask.to(loss.dtype)
        return masked_loss.sum() / mask.sum().clamp(min=1)


class WeightedBinaryCrossEntropy(nn.Module):
    """BCE with separate scalar weights for positive and negative samples."""

    def __init__(self, weight_0: float = 1.0, weight_1: float = 1.0) -> None:
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")
        weights = torch.where(
            y_true > 0.5,
            torch.full_like(y_true, self.weight_1),
            torch.full_like(y_true, self.weight_0),
        )
        return (weights * bce).mean()


class BinaryCrossEntropyRagged(nn.Module):
    """BCE over variable-length rows, optionally balancing positives/negatives.

    When ``balance_negatives`` is ``True``, each row's positive and negative
    contributions are rescaled by the reciprocal of their counts, so that
    heavy negative sampling does not drown out the positive signal.
    """

    def __init__(
        self, balance_negatives: bool = False, from_logits: bool = False
    ) -> None:
        super().__init__()
        self.balance_negatives = balance_negatives
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(
                y_pred, y_true, reduction="none"
            )
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")

        if self.balance_negatives:
            num_pos = (y_true == 1).sum(dim=-1, keepdim=True).to(loss.dtype)
            num_neg = (y_true == 0).sum(dim=-1, keepdim=True).to(loss.dtype)
            loss_pos = torch.where(
                y_true == 1, loss / (num_pos + 1e-8), torch.zeros_like(loss)
            )
            loss_neg = torch.where(
                y_true == 0, loss / (num_neg + 1e-8), torch.zeros_like(loss)
            )
            loss = loss_pos + loss_neg

        return loss.mean()


class CategoricalCrossEntropyRagged(nn.Module):
    """Categorical cross-entropy over ``[B, N, ...]`` predictions.

    ``y_true`` is expected to be one-hot encoded along the last dim; the
    class takes ``argmax`` to get integer targets.
    """

    def __init__(self, from_logits: bool = False) -> None:
        super().__init__()
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        targets = y_true.argmax(dim=-1)
        if self.from_logits:
            return F.cross_entropy(y_pred, targets)
        return F.nll_loss(torch.log(y_pred + 1e-8), targets)


__all__ = [
    "BinaryCrossEntropyRagged",
    "BinaryCrossEntropyWithMask",
    "CategoricalCrossEntropyRagged",
    "WeightedBinaryCrossEntropy",
]
