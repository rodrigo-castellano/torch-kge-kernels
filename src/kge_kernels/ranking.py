"""Ranking and metrics kernels for KGE evaluation."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

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


def ranking_metrics(
    ranks: Tensor,
    ks: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, float]:
    """Compute MRR and Hits@k from a rank tensor.

    Args:
        ranks: Float tensor of 1-based ranks (shape ``[N]``).
        ks: Cutoffs for Hits@k. Default ``(1, 3, 10)`` matches the legacy
            output keys ``Hits@1``, ``Hits@3``, ``Hits@10``.

    Returns:
        Dict with key ``MRR`` and one ``Hits@{k}`` entry per ``k``.
    """
    if ranks.numel() == 0:
        out: Dict[str, float] = {"MRR": 0.0}
        for k in ks:
            out[f"Hits@{k}"] = 0.0
        return out
    inv = 1.0 / ranks.double()
    results: Dict[str, float] = {"MRR": inv.mean().item()}
    for k in ks:
        results[f"Hits@{k}"] = (ranks <= k).double().mean().item()
    return results


def ranks_from_labeled_predictions(
    y_pred: Tensor,
    y_true: Tensor,
    pad_value: int = -1,
    positive_value: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Compute per-row ranks from ``[B, N]`` score/label pairs.

    ``y_true`` uses the convention: ``positive_value`` for correct answers,
    anything else non-``pad_value`` for distractors, ``pad_value`` for
    padded slots to ignore.

    When a row has multiple positives, the row's rank is computed against
    the best-scored positive (standard filtered ranking convention).

    Uses average tie handling: ``rank = 1 + #strictly_greater + 0.5 * #tied``.

    Args:
        y_pred: ``[B, N]`` scores.
        y_true: ``[B, N]`` labels.
        pad_value: Label value marking padded/ignored entries.
        positive_value: Label value marking positive (correct) entries.

    Returns:
        ``(ranks, valid)`` both shape ``[B]``. ``ranks`` is float32,
        ``valid`` is bool — ``False`` rows had no usable positive and
        should be excluded from metric aggregation.
    """
    mask = y_true != pad_value
    is_positive = y_true == positive_value
    valid = mask.any(dim=1) & is_positive.any(dim=1)

    pos_scores = (
        y_pred.masked_fill(~is_positive, float("-inf")).max(dim=1).values
    )
    n_greater = ((y_pred > pos_scores.unsqueeze(1)) & mask).sum(dim=1).to(
        torch.float32
    )
    n_equal = ((y_pred == pos_scores.unsqueeze(1)) & mask).sum(dim=1).to(
        torch.float32
    )
    ranks = n_greater + (1.0 + n_equal) * 0.5
    return ranks, valid


class StreamingRankingMetrics:
    """Streaming MRR + Hits@k accumulator for labeled-prediction batches.

    Designed for evaluation loops that want to accumulate per-batch ranks
    on GPU without tensor allocations or GPU→CPU syncs until ``compute()``.
    Internally stores one GPU scalar per metric and uses ``Tensor.add_``
    for in-place accumulation.

    Usage::

        metric = StreamingRankingMetrics(ks=(1, 3, 10))
        for batch in loader:
            y_pred, y_true = batch
            metric.update(y_pred, y_true)
        out = metric.compute()  # {"MRR": ..., "Hits@1": ..., ...}

    Args:
        ks: Hits cutoffs. Default ``(1, 3, 10)``.
        pad_value: Label value to ignore (default ``-1``).
        positive_value: Label value marking positives (default ``1``).
        metric_key: Legacy key name for the MRR output. Set to
            ``"mrrmetric"`` for the lowercase keys used by some codebases;
            default ``"MRR"`` matches :func:`ranking_metrics`.
    """

    def __init__(
        self,
        ks: Tuple[int, ...] = (1, 3, 10),
        pad_value: int = -1,
        positive_value: int = 1,
        metric_key: str = "MRR",
    ) -> None:
        self.ks = tuple(ks)
        self.pad_value = pad_value
        self.positive_value = positive_value
        self.metric_key = metric_key
        self._initialized = False
        self._mrr_sum: Optional[Tensor] = None
        self._count: Optional[Tensor] = None
        self._hits_sums: Dict[int, Tensor] = {}

    def reset(self) -> None:
        """Zero the accumulators (reusing allocated tensors if any)."""
        if self._initialized and self._mrr_sum is not None and self._count is not None:
            self._mrr_sum.zero_()
            self._count.zero_()
            for k in self.ks:
                self._hits_sums[k].zero_()
        else:
            self._mrr_sum = None
            self._count = None
            self._hits_sums = {}
            self._initialized = False

    def _init(self, device: torch.device) -> None:
        self._mrr_sum = torch.zeros((), device=device, dtype=torch.float64)
        self._count = torch.zeros((), device=device, dtype=torch.long)
        for k in self.ks:
            self._hits_sums[k] = torch.zeros((), device=device, dtype=torch.long)
        self._initialized = True

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        """Accumulate one batch without any GPU→CPU sync."""
        if not self._initialized:
            self._init(y_pred.device)
        assert self._mrr_sum is not None and self._count is not None

        ranks, valid = ranks_from_labeled_predictions(
            y_pred,
            y_true,
            pad_value=self.pad_value,
            positive_value=self.positive_value,
        )
        inv = (1.0 / ranks.double()).masked_fill(~valid, 0.0)
        self._mrr_sum.add_(inv.sum())
        self._count.add_(valid.sum())
        for k in self.ks:
            self._hits_sums[k].add_((ranks.le(k) & valid).sum())

    def compute(self) -> Dict[str, float]:
        """Return ``{metric_key: mrr, "Hits@{k}": hits@k}`` as Python floats."""
        hits_prefix = "Hits"
        if self.metric_key == "mrrmetric":
            hits_prefix = "hits"
        if not self._initialized or self._count is None or self._mrr_sum is None:
            out: Dict[str, float] = {self.metric_key: 0.0}
            for k in self.ks:
                out[f"{hits_prefix}@{k}"] = 0.0
            return out
        count = self._count.clamp(min=1).double()
        results: Dict[str, float] = {
            self.metric_key: (self._mrr_sum / count).item(),
        }
        for k in self.ks:
            results[f"{hits_prefix}@{k}"] = (
                self._hits_sums[k].double() / count
            ).item()
        return results


__all__ = [
    "StreamingRankingMetrics",
    "ranking_metrics",
    "ranks_from_labeled_predictions",
    "ranks_from_scores",
    "ranks_from_scores_matrix",
]
