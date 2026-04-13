import torch

from kge_kernels.eval import ranking_metrics, ranks_from_scores, ranks_from_scores_matrix


def test_ranks_from_scores_average():
    pos = torch.tensor([3.0, 2.0])
    neg = torch.tensor([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
    ranks = ranks_from_scores(pos, neg, tie_handling="average")
    # pos=3: 1 better (4), 0 tied -> rank 2.0
    # pos=2: 1 better (3), 1 tied (2) -> rank 2.5
    assert ranks[0].item() == 2.0
    assert ranks[1].item() == 2.5


def test_ranks_from_scores_with_valid_mask():
    pos = torch.tensor([3.0])
    neg = torch.tensor([[1.0, 5.0, 4.0]])
    mask = torch.tensor([[True, False, True]])
    ranks = ranks_from_scores(pos, neg, valid_mask=mask, tie_handling="average")
    # neg=5 is masked out; only 1 and 4 remain -> 1 better (4) -> rank 2.0
    assert ranks[0].item() == 2.0


def test_ranks_from_scores_random():
    pos = torch.tensor([2.0])
    neg = torch.tensor([[2.0, 2.0, 2.0]])
    gen = torch.Generator().manual_seed(42)
    ranks = ranks_from_scores(pos, neg, tie_handling="random", generator=gen)
    # All tied; random coin flips determine rank
    assert ranks[0].item() >= 1.0
    assert ranks[0].item() <= 4.0


def test_ranks_from_scores_matrix_average():
    scores = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
    ])
    true_indices = torch.tensor([2, 1])
    ranks = ranks_from_scores_matrix(scores, true_indices, tie_handling="average")
    # Entity 2 score=3.0: 1 better (4.0) -> rank 2.0
    # Entity 1 score=3.0: 1 better (4.0) -> rank 2.0
    assert ranks[0].item() == 2.0
    assert ranks[1].item() == 2.0


def test_ranks_from_scores_matrix_with_ties():
    scores = torch.tensor([[3.0, 3.0, 3.0, 1.0]])
    true_indices = torch.tensor([0])
    ranks = ranks_from_scores_matrix(scores, true_indices, tie_handling="average")
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
