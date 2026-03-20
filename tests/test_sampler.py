import torch

from kge_kernels.sampler import Sampler, corrupt
from kge_kernels.utils import compute_bernoulli_probs


def test_corrupt_with_mask_supports_zero_ids():
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
    out = corrupt(sampler, pos, num_corruptions=None, mode="tail")
    neg, valid = out.negatives, out.valid_mask

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
    out = corrupt(sampler, pos, num_corruptions=None, mode="head")
    neg, valid = out.negatives, out.valid_mask
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
    out = corrupt(sampler, pos, num_corruptions=None, mode="tail")
    neg, valid = out.negatives, out.valid_mask
    rows = [tuple(triple.tolist()) for triple, keep in zip(neg[0], valid[0]) if keep]
    assert rows == [(0, 1, 3)]


def test_bernoulli_corruption_mode():
    """Bernoulli mode produces valid corruptions and respects the shape contract."""
    known = torch.tensor([
        [0, 0, 1],
        [0, 1, 2],
        [1, 2, 0],
    ], dtype=torch.long)
    bern_probs = compute_bernoulli_probs(known, num_relations=2)
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=3,
        num_relations=2,
        device=torch.device("cpu"),
        min_entity_idx=0,
        bern_probs=bern_probs,
    )

    pos = torch.tensor([[0, 0, 1], [1, 2, 0]], dtype=torch.long)
    neg, valid = sampler.corrupt_with_mask(
        pos, num_negatives=5, mode="bernoulli",
        filter=False, unique=False,
    )
    assert neg.shape == (2, 5, 3)
    assert valid.shape == (2, 5)
    # Relations must be preserved (column 0)
    for i in range(2):
        for j in range(5):
            assert neg[i, j, 0].item() == pos[i, 0].item()


def test_bernoulli_without_probs_raises():
    """Bernoulli mode without bern_probs raises a clear error."""
    known = torch.tensor([[0, 0, 1]], dtype=torch.long)
    sampler = Sampler.from_data(
        all_known_triples_idx=known,
        num_entities=3,
        num_relations=1,
        device=torch.device("cpu"),
        min_entity_idx=0,
    )
    pos = torch.tensor([[0, 0, 1]], dtype=torch.long)
    try:
        sampler.corrupt_with_mask(pos, num_negatives=2, mode="bernoulli")
        assert False, "Expected RuntimeError"
    except RuntimeError:
        pass


def test_compute_bernoulli_probs():
    """compute_bernoulli_probs returns probabilities in valid range."""
    triples = torch.tensor([
        [0, 0, 1],  # r=0, h=0, t=1
        [0, 0, 2],  # r=0, h=0, t=2
        [0, 1, 2],  # r=0, h=1, t=2
        [1, 0, 1],  # r=1, h=0, t=1
    ], dtype=torch.long)
    probs = compute_bernoulli_probs(triples, num_relations=2)
    assert probs.shape == (2,)
    assert (probs >= 0.05).all()
    assert (probs <= 0.95).all()
    # Relation 0: 2 unique heads (0,1), 2 unique tails (1,2), 3 triples
    # tph = 3/2 = 1.5, hpt = 3/2 = 1.5 -> prob = 0.5
    assert abs(probs[0].item() - 0.5) < 1e-5
