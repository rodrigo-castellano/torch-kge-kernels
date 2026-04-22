"""Tests for kge_kernels.training.batching.iterate_epoch_batches."""
from __future__ import annotations

import torch

from kge_kernels.scoring import Sampler
from kge_kernels.training import iterate_epoch_batches


def _make_sampler(num_entities: int = 10, num_relations: int = 3, seed: int = 0):
    return Sampler.from_data(
        all_known_triples_idx=torch.empty((0, 3), dtype=torch.long),
        num_entities=num_entities,
        num_relations=num_relations,
        device=torch.device("cpu"),
        default_mode="tail",
        seed=seed,
        min_entity_idx=0,
    )


def test_iterate_yields_expected_batches():
    """Yields ceil(N/B) batches; per-batch shapes match args."""
    sampler = _make_sampler()
    # 10 triples: r, h, t
    triples = torch.tensor(
        [[0, 1, 2], [0, 3, 4], [1, 5, 6], [2, 7, 8], [0, 9, 0],
         [1, 2, 3], [2, 4, 5], [0, 6, 7], [1, 8, 9], [2, 0, 1]],
        dtype=torch.long,
    )
    batches = list(iterate_epoch_batches(
        triples, sampler, batch_size=3, num_negatives=2,
        corrupt_modes=["tail"], filter=False,
    ))
    # 10 triples, batch_size=3 → 4 batches of sizes 3,3,3,1
    assert len(batches) == 4
    sizes = [b[0].shape[0] for b in batches]
    assert sizes == [3, 3, 3, 1]
    # Negatives: [B, K_total=2, 3]
    for pos, neg, valid in batches:
        assert pos.ndim == 2 and pos.shape[-1] == 3
        assert neg.ndim == 3 and neg.shape[-1] == 3
        assert neg.shape[1] == 2     # K_total
        assert valid.shape == neg.shape[:-1]


def test_iterate_multi_mode_concatenates():
    """Two modes → K_total doubles along the K axis."""
    sampler = _make_sampler()
    triples = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.long)
    batches = list(iterate_epoch_batches(
        triples, sampler, batch_size=2, num_negatives=3,
        corrupt_modes=["head", "tail"], filter=False,
    ))
    assert len(batches) == 1
    pos, neg, valid = batches[0]
    assert neg.shape == (2, 6, 3)   # K_total = 2 modes × 3 = 6
    assert valid.shape == (2, 6)


def test_iterate_reproducible_with_generator():
    """Same generator seed → same batch order."""
    sampler = _make_sampler()
    triples = torch.arange(18, dtype=torch.long).reshape(6, 3)

    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)
    b1 = list(iterate_epoch_batches(
        triples, sampler, batch_size=2, num_negatives=1,
        corrupt_modes=["tail"], generator=g1, filter=False,
    ))
    b2 = list(iterate_epoch_batches(
        triples, sampler, batch_size=2, num_negatives=1,
        corrupt_modes=["tail"], generator=g2, filter=False,
    ))
    assert len(b1) == len(b2)
    for (pos1, _, _), (pos2, _, _) in zip(b1, b2):
        assert torch.equal(pos1, pos2)


def test_iterate_covers_every_positive_exactly_once():
    """Union of yielded positive tensors equals the input set."""
    sampler = _make_sampler()
    triples = torch.tensor(
        [[0, i, i + 1] for i in range(7)], dtype=torch.long,
    )
    pos_batches = [
        pos for pos, _, _ in iterate_epoch_batches(
            triples, sampler, batch_size=3, num_negatives=1,
            corrupt_modes=["tail"], filter=False,
        )
    ]
    all_pos = torch.cat(pos_batches, dim=0)
    assert all_pos.shape == triples.shape
    # Check it is a permutation of the input (as a set of rows)
    orig = {tuple(row.tolist()) for row in triples}
    got = {tuple(row.tolist()) for row in all_pos}
    assert orig == got
