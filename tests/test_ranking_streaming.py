"""Tests for metrics_from_ranks (ks parameter) and StreamingRankingMetrics."""
from __future__ import annotations

import math

import torch

from kge_kernels.eval import (
    StreamingRankingMetrics,
    metrics_from_ranks,
)

# ═══════════════════════════════════════════════════════════════════════
# metrics_from_ranks with ks parameter
# ═══════════════════════════════════════════════════════════════════════


def test_metrics_from_ranks_default_ks():
    ranks = torch.tensor([1.0, 2.0, 5.0, 11.0])
    out = metrics_from_ranks(ranks)
    assert "MRR" in out
    for k in (1, 3, 10):
        assert f"Hits@{k}" in out


def test_metrics_from_ranks_custom_ks():
    ranks = torch.tensor([1.0, 2.0, 5.0, 11.0, 20.0])
    out = metrics_from_ranks(ranks, ks=(1, 5, 20))
    assert set(out.keys()) == {"MRR", "Hits@1", "Hits@5", "Hits@20"}
    assert out["Hits@1"] == 0.2   # only rank 1 qualifies
    assert out["Hits@5"] == 0.6   # ranks 1, 2, 5
    assert out["Hits@20"] == 1.0  # all


def test_metrics_from_ranks_mrr_value():
    ranks = torch.tensor([1.0, 2.0, 4.0])
    out = metrics_from_ranks(ranks)
    expected_mrr = (1.0 + 0.5 + 0.25) / 3
    assert math.isclose(out["MRR"], expected_mrr, rel_tol=1e-6)


def test_metrics_from_ranks_empty_tensor():
    out = metrics_from_ranks(torch.tensor([], dtype=torch.float32), ks=(1, 10))
    assert out == {"MRR": 0.0, "Hits@1": 0.0, "Hits@10": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# StreamingRankingMetrics — integration tests cover the inlined
# label→best-positive→rank pipeline previously exposed as
# _ranks_from_labeled_predictions.
# ═══════════════════════════════════════════════════════════════════════


def test_streaming_metrics_perfect_predictions():
    metric = StreamingRankingMetrics(ks=(1, 3, 10))
    y_pred = torch.tensor([[0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])
    y_true = torch.tensor([[1, 0, 0], [0, 1, 0]])
    metric.update(y_pred, y_true)
    out = metric.compute()
    assert out["MRR"] == 1.0
    assert out["Hits@1"] == 1.0
    assert out["Hits@3"] == 1.0


def test_streaming_metrics_worst_predictions():
    metric = StreamingRankingMetrics(ks=(1, 3, 10))
    y_pred = torch.tensor([[0.1, 0.9, 0.5]])
    y_true = torch.tensor([[1, 0, 0]])
    metric.update(y_pred, y_true)
    out = metric.compute()
    # positive scored 0.1, both others better → rank 3
    assert math.isclose(out["MRR"], 1 / 3, rel_tol=1e-6)
    assert out["Hits@1"] == 0.0
    assert out["Hits@3"] == 1.0


def test_streaming_metrics_best_positive_wins_when_multiple():
    """When a row has multiple positives, the highest-scored one is ranked."""
    metric = StreamingRankingMetrics(ks=(1,))
    y_pred = torch.tensor([[0.9, 0.5, 0.1]])
    y_true = torch.tensor([[1, 1, 0]])  # two positives
    metric.update(y_pred, y_true)
    out = metric.compute()
    # Best positive is at idx 0 (score 0.9) → rank 1
    assert out["MRR"] == 1.0


def test_streaming_metrics_padding_ignored():
    metric = StreamingRankingMetrics(ks=(1,))
    y_pred = torch.tensor([[0.99, 0.5]])  # would be best but is padding
    y_true = torch.tensor([[-1, 1]])
    metric.update(y_pred, y_true)
    out = metric.compute()
    # Only valid cell is the positive at idx 1 → rank 1 among valid
    assert out["MRR"] == 1.0


def test_streaming_metrics_average_tie_handling():
    metric = StreamingRankingMetrics(ks=(1,))
    y_pred = torch.tensor([[0.5, 0.5, 0.5]])
    y_true = torch.tensor([[1, 0, 0]])
    metric.update(y_pred, y_true)
    out = metric.compute()
    # 0 strictly greater, 3 equal (incl. self) → rank = 1 + 0.5*(3-1) = 2.0
    assert math.isclose(out["MRR"], 0.5, rel_tol=1e-6)


def test_streaming_metrics_accumulates_across_batches():
    metric = StreamingRankingMetrics(ks=(1, 3, 10))
    # First batch: perfect (rank 1)
    metric.update(
        torch.tensor([[0.9, 0.1]]), torch.tensor([[1, 0]])
    )
    # Second batch: worst (rank 2)
    metric.update(
        torch.tensor([[0.1, 0.9]]), torch.tensor([[1, 0]])
    )
    out = metric.compute()
    expected_mrr = (1.0 + 0.5) / 2
    assert math.isclose(out["MRR"], expected_mrr, rel_tol=1e-6)
    assert out["Hits@1"] == 0.5
    assert out["Hits@3"] == 1.0


def test_streaming_metrics_reset_clears_state():
    metric = StreamingRankingMetrics(ks=(1,))
    metric.update(torch.tensor([[0.9, 0.1]]), torch.tensor([[1, 0]]))
    metric.reset()
    # After reset, compute should show zeros (accumulators allocated but zeroed)
    out = metric.compute()
    assert out["MRR"] == 0.0
    assert out["Hits@1"] == 0.0


def test_streaming_metrics_lowercase_key_compat():
    """torch-ns uses 'mrrmetric' / 'hits@k' lowercase keys."""
    metric = StreamingRankingMetrics(ks=(1, 3), metric_key="mrrmetric")
    metric.update(
        torch.tensor([[0.9, 0.1, 0.5]]), torch.tensor([[1, 0, 0]])
    )
    out = metric.compute()
    assert "mrrmetric" in out
    assert "hits@1" in out
    assert "hits@3" in out


def test_streaming_metrics_skips_all_padded_row():
    metric = StreamingRankingMetrics(ks=(1,))
    y_pred = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    y_true = torch.tensor([[1, 0], [-1, -1]])  # second row all padded
    metric.update(y_pred, y_true)
    out = metric.compute()
    assert out["MRR"] == 1.0  # only valid row was perfect


def test_streaming_metrics_no_positive_marked_invalid():
    """Rows with no positive are excluded from the metrics."""
    metric = StreamingRankingMetrics(ks=(1,))
    y_pred = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    y_true = torch.tensor([[1, 0], [0, 0]])  # second row has no positive
    metric.update(y_pred, y_true)
    out = metric.compute()
    # Only the first row contributes: rank 1 → MRR 1.0
    assert out["MRR"] == 1.0
