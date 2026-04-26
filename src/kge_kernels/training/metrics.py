"""Streaming metric utilities for training-time observability.

These accumulate metrics over many per-batch ``(y_pred, y_true)``
labeled-prediction tensors during training — distinct from the eval
flow, which goes through :class:`kge_kernels.eval.RankingEvaluator`
and produces a single :class:`~kge_kernels.eval.RankingResult` per call.

When you have a per-batch labeled-prediction stream and want a running
MRR / Hits@k (typical training-loop instrumentation), use
:class:`StreamingRankingMetrics`. When you have a held-out triple set
and a corruption sampler and want the standard filtered MRR, use
:func:`kge_kernels.eval.RankingEvaluator`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..eval.ranking import compute_ranks


class StreamingRankingMetrics:
    """Streaming MRR + Hits@k accumulator for labeled-prediction batches.

    Designed for training loops that want to accumulate per-batch ranks
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
        metric_key: Output key for MRR. Set to ``"mrrmetric"`` for the
            lowercase convention used by some codebases; default
            ``"MRR"`` matches :func:`kge_kernels.eval.metrics_from_ranks`.
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
        # via per-validation-batch overhead.
        self._mrr_sum = torch.zeros((), device=device, dtype=torch.float32)
        self._count = torch.zeros((), device=device, dtype=torch.long)
        for k in self.ks:
            self._hits_sums[k] = torch.zeros((), device=device, dtype=torch.long)
        self._initialized = True

    def update(self, y_pred: Tensor, y_true: Tensor) -> None:
        """Accumulate one batch without any GPU→CPU sync.

        ``y_true`` is a per-cell label tensor: ``positive_value`` marks
        positives, ``pad_value`` marks ignored cells, anything else is a
        distractor. We pick the best-scored positive per row and rank it.

        All operations run in ``y_pred.dtype`` (typically float32) with
        no dtype promotion. The in-place ``masked_fill_`` avoids
        allocating a fresh tensor per batch.
        """
        if not self._initialized:
            self._init(y_pred.device)
        assert self._mrr_sum is not None and self._count is not None

        # Pick the highest-scored positive per row, rank it among valid cells.
        valid_cells = y_true != self.pad_value
        is_positive = y_true == self.positive_value
        valid = valid_cells.any(dim=1) & is_positive.any(dim=1)
        best_pos_idx = y_pred.masked_fill(~is_positive, float("-inf")).argmax(dim=1)
        ranks = compute_ranks(y_pred, best_pos_idx, valid_mask=valid_cells)

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


__all__ = ["StreamingRankingMetrics"]
