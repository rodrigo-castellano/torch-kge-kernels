"""Tests for the new ranking surfaces: ks parameter, ranks_from_labeled_predictions,
StreamingRankingMetrics."""
from __future__ import annotations

import math

import torch

from kge_kernels.ranking import (
    StreamingRankingMetrics,
    ranking_metrics,
    ranks_from_labeled_predictions,
)

# ═══════════════════════════════════════════════════════════════════════
# ranking_metrics with ks parameter
# ═══════════════════════════════════════════════════════════════════════


def test_ranking_metrics_default_ks():
    ranks = torch.tensor([1.0, 2.0, 5.0, 11.0])
    out = ranking_metrics(ranks)
    assert "MRR" in out
    for k in (1, 3, 10):
        assert f"Hits@{k}" in out


def test_ranking_metrics_custom_ks():
    ranks = torch.tensor([1.0, 2.0, 5.0, 11.0, 20.0])
    out = ranking_metrics(ranks, ks=(1, 5, 20))
    assert set(out.keys()) == {"MRR", "Hits@1", "Hits@5", "Hits@20"}
    assert out["Hits@1"] == 0.2   # only rank 1 qualifies
    assert out["Hits@5"] == 0.6   # ranks 1, 2, 5
    assert out["Hits@20"] == 1.0  # all


def test_ranking_metrics_mrr_value():
    ranks = torch.tensor([1.0, 2.0, 4.0])
    out = ranking_metrics(ranks)
    expected_mrr = (1.0 + 0.5 + 0.25) / 3
    assert math.isclose(out["MRR"], expected_mrr, rel_tol=1e-6)


def test_ranking_metrics_empty_tensor():
    out = ranking_metrics(torch.tensor([], dtype=torch.float32), ks=(1, 10))
    assert out == {"MRR": 0.0, "Hits@1": 0.0, "Hits@10": 0.0}


# ═══════════════════════════════════════════════════════════════════════
# ranks_from_labeled_predictions
# ═══════════════════════════════════════════════════════════════════════


def test_ranks_single_positive_first():
    y_pred = torch.tensor([[0.9, 0.1, 0.2]])
    y_true = torch.tensor([[1, 0, 0]])
    ranks, valid = ranks_from_labeled_predictions(y_pred, y_true)
    assert ranks.item() == 1.0
    assert valid.item()


def test_ranks_single_positive_middle():
    y_pred = torch.tensor([[0.9, 0.5, 0.1]])
    y_true = torch.tensor([[0, 1, 0]])
    ranks, valid = ranks_from_labeled_predictions(y_pred, y_true)
    assert ranks.item() == 2.0


def test_ranks_ignores_padding():
    y_pred = torch.tensor([[0.99, 0.5]])    # padding shouldn't be counted
    y_true = torch.tensor([[-1, 1]])
    ranks, valid = ranks_from_labeled_predictions(y_pred, y_true)
    assert ranks.item() == 1.0
    assert valid.item()


def test_ranks_multiple_positives_best_wins():
    y_pred = torch.tensor([[0.9, 0.5, 0.1]])
    y_true = torch.tensor([[1, 1, 0]])
    ranks, valid = ranks_from_labeled_predictions(y_pred, y_true)
    # Best positive score = 0.9 → rank should be 1
    assert ranks.item() == 1.0


def test_ranks_average_tie_handling():
    y_pred = torch.tensor([[0.5, 0.5, 0.5]])
    y_true = torch.tensor([[1, 0, 0]])
    ranks, _ = ranks_from_labeled_predictions(y_pred, y_true)
    # 0 strictly greater, 3 equal (incl. self) → rank = 0 + (1 + 3)/2 = 2
    assert ranks.item() == 2.0


def test_ranks_no_positive_marked_invalid():
    y_pred = torch.tensor([[0.5, 0.5]])
    y_true = torch.tensor([[0, 0]])
    _, valid = ranks_from_labeled_predictions(y_pred, y_true)
    assert not valid.item()


def test_ranks_batched():
    y_pred = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    y_true = torch.tensor([[1, 0], [1, 0]])
    ranks, valid = ranks_from_labeled_predictions(y_pred, y_true)
    assert ranks.tolist() == [1.0, 2.0]
    assert valid.tolist() == [True, True]


# ═══════════════════════════════════════════════════════════════════════
# StreamingRankingMetrics
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
