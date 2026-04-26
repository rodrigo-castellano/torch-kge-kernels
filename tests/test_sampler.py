import torch

from kge_kernels.scoring import BernoulliSampler, Sampler


def test_corrupt_supports_zero_ids():
    known = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=3,
        num_relations=2,
        device=torch.device("cpu"),
        min_entity_idx=0,
    )

    pos = torch.tensor([[0, 0, 0]], dtype=torch.long)
    neg, valid = sampler.corrupt(pos, num_negatives=None, mode="tail", return_mask=True)

    rows = [tuple(triple.tolist()) for triple, keep in zip(neg[0], valid[0]) if keep]
    assert (0, 0, 0) not in rows
    assert (0, 0, 1) not in rows
    assert rows == [(0, 0, 2)]


def test_domain_sampling_stays_in_domain():
    known = torch.tensor([[0, 0, 1]], dtype=torch.long)
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=4,
        num_relations=1,
        device=torch.device("cpu"),
        domain2idx={"a": [0, 2], "b": [1, 3]},
        entity2domain={0: "a", 1: "b", 2: "a", 3: "b"},
        min_entity_idx=0,
    )

    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)
    neg, valid = sampler.corrupt(pos, num_negatives=None, mode="head", return_mask=True)
    rows = [tuple(triple.tolist()) for triple, keep in zip(neg[0], valid[0]) if keep]
    assert rows == [(0, 2, 1)]


def test_exhaustive_domain_sampling_does_not_leak_padding_entity_zero():
    known = torch.tensor([[0, 1, 2]], dtype=torch.long)
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=5,
        num_relations=1,
        device=torch.device("cpu"),
        domain2idx={"small": [2, 3], "large": [0, 1, 4]},
        entity2domain={0: "large", 1: "large", 2: "small", 3: "small", 4: "large"},
        min_entity_idx=0,
    )

    pos = torch.tensor([[0, 1, 2]], dtype=torch.long)
    neg, valid = sampler.corrupt(pos, num_negatives=None, mode="tail", return_mask=True)
    rows = [tuple(triple.tolist()) for triple, keep in zip(neg[0], valid[0]) if keep]
    assert rows == [(0, 1, 3)]


def test_corrupt_default_returns_negs_only():
    """Default ``return_mask=False`` returns just the negatives tensor."""
    known = torch.tensor([[0, 0, 1]], dtype=torch.long)
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=4,
        num_relations=1,
        device=torch.device("cpu"),
        min_entity_idx=0,
    )
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)
    out = sampler.corrupt(pos, num_negatives=2, mode="tail")
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 2, 3)


def test_bernoulli_sampler_produces_valid_corruptions():
    """BernoulliSampler.corrupt yields per-triple coin-flipped negatives."""
    known = torch.tensor([
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 0],
    ], dtype=torch.long)
    bern_probs = BernoulliSampler.compute_probs(known, num_relations=2)
    sampler = BernoulliSampler.from_data(
        all_known_triples_idx=known,
        num_entities=3,
        num_relations=2,
        device=torch.device("cpu"),
        min_entity_idx=0,
        bern_probs=bern_probs,
    )

    pos = torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long)
    neg, valid = sampler.corrupt(
        pos, num_negatives=5, filter=False, unique=False, return_mask=True,
    )
    assert neg.shape == (2, 5, 3)
    assert valid.shape == (2, 5)
    # Relations must be preserved (column 0)
    for i in range(2):
        for j in range(5):
            assert neg[i, j, 0].item() == pos[i, 0].item()


def test_compute_probs():
    """compute_probs returns probabilities in valid range."""
    triples = torch.tensor([
        [0, 0, 1],  # r=0, h=0, t=1
        [0, 0, 2],  # r=0, h=0, t=2
        [0, 1, 2],  # r=0, h=1, t=2
        [1, 0, 1],  # r=1, h=0, t=1
    ], dtype=torch.long)
    probs = BernoulliSampler.compute_probs(triples, num_relations=2)
    assert probs.shape == (2,)
    assert (probs >= 0.05).all()
    assert (probs <= 0.95).all()
