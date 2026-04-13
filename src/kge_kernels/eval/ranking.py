"""Ranking and metrics kernels for KGE evaluation."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor


def compute_ranks(
    scores: Tensor,
    true_idx: Tensor,
    valid_mask: Optional[torch.BoolTensor] = None,
    tie_handling: Literal["average", "random"] = "average",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Compute 1-based rank of the true item among candidates.

    This is the single rank-computation function used by all evaluation
    paths: exhaustive filtered ranking, sampled corruption ranking, and
    streaming mini-batch ranking.

    Args:
        scores: ``[B, N]`` score matrix — all candidate scores per query.
        true_idx: ``[B]`` index of the true item in each row.
        valid_mask: ``[B, N]`` boolean mask — ``False`` entries are treated
            as ``-inf`` (ignored). ``None`` means all entries are valid.
        tie_handling: ``"average"`` gives ``1 + #better + 0.5 * #tied``
            (excluding the true item from the tie count);
            ``"random"`` breaks ties with a seeded coin flip.
        generator: RNG for ``"random"`` tie handling.

    Returns:
        ``[B]`` float tensor of 1-based ranks.
    """
    B, N = scores.shape
    device = scores.device

    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))

    batch_idx = torch.arange(B, device=device)
    target = scores[batch_idx, true_idx]  # [B]

    greater = (scores > target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)

    if tie_handling == "average":
        equal = (scores == target.unsqueeze(1)).sum(dim=1, dtype=torch.float32)
        # Subtract 1 for the true item itself (it ties with itself)
        return greater + 1.0 + 0.5 * (equal - 1.0).clamp(min=0)
    else:
        tied = scores == target.unsqueeze(1)
        tied[batch_idx, true_idx] = False  # exclude true item from tie-breaking
        rnd = torch.rand(scores.shape, generator=generator, device=device)
        rnd_target = rnd[batch_idx, true_idx]
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


def _ranks_from_labeled_predictions(
    y_pred: Tensor,
    y_true: Tensor,
    pad_value: int = -1,
    positive_value: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Compute per-row ranks from ``[B, N]`` score/label pairs.

    Finds the best-scored positive per row and delegates to
    :func:`compute_ranks`. Rows without a positive are flagged as invalid.

    Args:
        y_pred: ``[B, N]`` scores.
        y_true: ``[B, N]`` labels (``positive_value`` = correct,
            ``pad_value`` = ignored, anything else = distractor).

    Returns:
        ``(ranks, valid)`` both ``[B]``. Invalid rows should be excluded.
    """
    mask = y_true != pad_value
    is_positive = y_true == positive_value
    valid = mask.any(dim=1) & is_positive.any(dim=1)

    # Find index of the best-scored positive per row
    best_pos_idx = y_pred.masked_fill(~is_positive, float("-inf")).argmax(dim=1)

    ranks = compute_ranks(y_pred, best_pos_idx, valid_mask=mask)
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
        # Float32 accumulators match torch-ns's pre-consolidation
        # FusedRankingMetrics byte-for-byte. float64 is overkill for MRR /
        # Hits@k summations (ranks are at most ~vocab_size, 1/rank ≥ 1e-5
        # has plenty of float32 precision) and the extra float32→float64
        # cast on the hot path measurably slowed torch-ns test_train_speed
        # via the per-validation-batch overhead in evaluate_chunked.
        self._mrr_sum = torch.zeros((), device=device, dtype=torch.float32)
        self._count = torch.zeros((), device=device, dtype=torch.long)
        for k in self.ks:
            self._hits_sums[k] = torch.zeros((), device=device, dtype=torch.long)
        self._initialized = True

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        """Accumulate one batch without any GPU→CPU sync.

        All operations run in ``y_pred.dtype`` (typically float32) with
        no dtype promotion. The in-place ``masked_fill_`` avoids
        allocating a fresh tensor per batch.
        """
        if not self._initialized:
            self._init(y_pred.device)
        assert self._mrr_sum is not None and self._count is not None

        ranks, valid = _ranks_from_labeled_predictions(
            y_pred,
            y_true,
            pad_value=self.pad_value,
            positive_value=self.positive_value,
        )
        inv_ranks = (1.0 / ranks).masked_fill_(~valid, 0.0)
        self._mrr_sum.add_(inv_ranks.sum())
        self._count.add_(valid.sum())
        for k in self.ks:
            self._hits_sums[k].add_((ranks.le(k) & valid).sum())

    def compute(self) -> Dict[str, float]:
        """Return ``{metric_key: mrr, "Hits@{k}": hits@k}`` as Python floats.

        The final division uses float32 to match torch-ns legacy behavior.
        For the ``mrrmetric`` / ``hits@k`` lowercase convention, set
        ``metric_key="mrrmetric"`` at construction time.
        """
        hits_prefix = "Hits"
        if self.metric_key == "mrrmetric":
            hits_prefix = "hits"
        if not self._initialized or self._count is None or self._mrr_sum is None:
            out: Dict[str, float] = {self.metric_key: 0.0}
            for k in self.ks:
                out[f"{hits_prefix}@{k}"] = 0.0
            return out
        count = self._count.clamp(min=1).float()
        results: Dict[str, float] = {
            self.metric_key: (self._mrr_sum / count).item(),
        }
        for k in self.ks:
            results[f"{hits_prefix}@{k}"] = (
                self._hits_sums[k].float() / count
            ).item()
        return results


__all__ = [
    "StreamingRankingMetrics",
    "compute_ranks",
    "ranking_metrics",
]
