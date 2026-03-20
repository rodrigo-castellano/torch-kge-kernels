"""Ranking and metrics kernels for KGE evaluation."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor


def ranks_from_scores(
    pos_scores: Tensor,
    neg_scores: Tensor,
    valid_mask: Optional[torch.BoolTensor] = None,
    tie_handling: Literal["average", "random"] = "average",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Compute rank of positive among K negatives.

    Args:
        pos_scores: ``[B]`` or ``[B, 1]`` positive scores.
        neg_scores: ``[B, K]`` negative scores.
        valid_mask: ``[B, K]`` — ``False`` entries are ignored (treated as
            ``-inf``).
        tie_handling: ``"average"`` gives ``1 + #better + 0.5 * #tied``;
            ``"random"`` breaks ties with a coin flip.
        generator: RNG for ``"random"`` tie handling.

    Returns:
        ``[B]`` float ranks (1-based).
    """
    if pos_scores.dim() == 2:
        pos_scores = pos_scores.squeeze(1)
    pos = pos_scores.unsqueeze(1)  # [B, 1]

    if valid_mask is not None:
        neg_scores = neg_scores.masked_fill(~valid_mask, float("-inf"))

    better = (neg_scores > pos).sum(dim=1, dtype=torch.float32)

    if tie_handling == "average":
        equal = (neg_scores == pos).sum(dim=1, dtype=torch.float32)
        return better + 1.0 + 0.5 * equal
    else:
        tied = neg_scores == pos
        rnd_neg = torch.rand(neg_scores.shape, generator=generator, device=neg_scores.device)
        rnd_pos = torch.rand(pos_scores.shape, generator=generator, device=pos_scores.device)
        coin = rnd_neg > rnd_pos.unsqueeze(1)
        tied_won = (tied & coin).sum(dim=1, dtype=torch.float32)
        return better + 1.0 + tied_won


def ranks_from_scores_matrix(
    scores: Tensor,
    true_indices: torch.LongTensor,
    tie_handling: Literal["average", "random"] = "average",
) -> Tensor:
    """Rank true entity among all E candidates.

    Args:
        scores: ``[B, E]`` score matrix over all entities.
        true_indices: ``[B]`` indices of the true entities.
        tie_handling: ``"average"`` or ``"random"``.

    Returns:
        ``[B]`` float ranks (1-based).
    """
    batch_idx = torch.arange(scores.shape[0], device=scores.device)
    target = scores[batch_idx, true_indices]
    greater = (scores > target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)
    equal = (scores == target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)
    if tie_handling == "average":
        return greater + 1.0 + 0.5 * (equal - 1.0).clamp(min=0)
    else:
        rnd = torch.rand_like(scores)
        rnd_target = rnd[batch_idx, true_indices]
        tied = scores == target.unsqueeze(1)
        tied[batch_idx, true_indices] = False
        coin = rnd > rnd_target.unsqueeze(1)
        tied_won = (tied & coin).sum(dim=1, dtype=torch.float32)
        return greater + 1.0 + tied_won


def ranking_metrics(ranks: Tensor) -> dict[str, float]:
    """Compute standard ranking metrics from a rank tensor.

    Args:
        ranks: Float tensor of 1-based ranks.

    Returns:
        Dict with ``MRR``, ``Hits@1``, ``Hits@3``, ``Hits@10``.
    """
    return {
        "MRR": (1.0 / ranks.double()).mean().item(),
        "Hits@1": (ranks <= 1).double().mean().item(),
        "Hits@3": (ranks <= 3).double().mean().item(),
        "Hits@10": (ranks <= 10).double().mean().item(),
    }


__all__ = [
    "ranks_from_scores",
    "ranks_from_scores_matrix",
    "ranking_metrics",
]
