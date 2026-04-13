import torch

from kge_kernels.eval import compute_ranks, ranking_metrics


def test_compute_ranks_average():
    # scores matrix: true item at index 0 per row
    # Row 0: true=3.0, candidates=[3.0, 1.0, 2.0, 4.0] -> 1 better (4), 0 tied (excl self) -> rank 2.0
    # Row 1: true=2.0, candidates=[2.0, 1.0, 2.0, 3.0] -> 1 better (3), 1 tied -> rank 2.5
    scores = torch.tensor([[3.0, 1.0, 2.0, 4.0], [2.0, 1.0, 2.0, 3.0]])
    true_idx = torch.tensor([0, 0])
    ranks = compute_ranks(scores, true_idx, tie_handling="average")
    assert ranks[0].item() == 2.0
    assert ranks[1].item() == 2.5


def test_compute_ranks_with_valid_mask():
    scores = torch.tensor([[3.0, 1.0, 5.0, 4.0]])
    true_idx = torch.tensor([0])
    mask = torch.tensor([[True, True, False, True]])  # mask out 5.0
    ranks = compute_ranks(scores, true_idx, valid_mask=mask, tie_handling="average")
    # After mask: true=3.0, valid candidates: [3.0, 1.0, -inf, 4.0] -> 1 better (4) -> rank 2.0
    assert ranks[0].item() == 2.0


def test_compute_ranks_random():
    scores = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    true_idx = torch.tensor([0])
    gen = torch.Generator().manual_seed(42)
    ranks = compute_ranks(scores, true_idx, tie_handling="random", generator=gen)
    # All tied; random coin flips determine rank
    assert ranks[0].item() >= 1.0
    assert ranks[0].item() <= 4.0


def test_compute_ranks_matrix_average():
    scores = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    true_idx = torch.tensor([2, 1])
    ranks = compute_ranks(scores, true_idx, tie_handling="average")
    # Entity 2 score=3.0: 1 better (4.0) -> rank 2.0
    # Entity 1 score=3.0: 1 better (4.0) -> rank 2.0
    assert ranks[0].item() == 2.0
    assert ranks[1].item() == 2.0


def test_compute_ranks_matrix_with_ties():
    scores = torch.tensor([[3.0, 3.0, 3.0, 1.0]])
    true_idx = torch.tensor([0])
    ranks = compute_ranks(scores, true_idx, tie_handling="average")
    # target=3.0, greater=0, equal=3 (incl self) -> 1.0 + 0.5*(3-1) = 2.0
    assert ranks[0].item() == 2.0


def test_ranking_metrics():
    ranks = torch.tensor([1.0, 2.0, 5.0, 11.0])
    metrics = ranking_metrics(ranks)
    expected_mrr = (1 / 1.0 + 1 / 2.0 + 1 / 5.0 + 1 / 11.0) / 4
    assert abs(metrics["MRR"] - expected_mrr) < 1e-6
    assert metrics["Hits@1"] == 0.25
    assert metrics["Hits@3"] == 0.5
    assert metrics["Hits@10"] == 0.75
