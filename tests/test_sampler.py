import torch

from kge_kernels.sampler import Sampler


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
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=None, mode="tail")

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
    neg, valid = sampler.corrupt_with_mask(pos, num_negatives=None, mode="head")
    rows = [tuple(triple.tolist()) for triple, keep in zip(neg[0], valid[0]) if keep]
    assert rows == [(0, 2, 1)]
