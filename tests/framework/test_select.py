"""Select primitive tests: shape, determinism, and seed control."""
from __future__ import annotations

import torch

from kge_kernels.framework import (
    BeamSelect,
    ExhaustiveSelect,
    GreedySelect,
    Repr,
    SampleSelect,
)


def test_exhaustive_select_returns_none_state():
    state, info = ExhaustiveSelect()(evidence=None, s_repr=Repr(scores=torch.zeros(1)))
    assert state is None
    assert info is None


def test_greedy_select_picks_argmax():
    s = Repr(scores=torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.4]]))
    _, info = GreedySelect()(evidence=None, s_repr=s)
    assert info is not None
    assert info.chosen_indices.shape == (2, 1)
    assert torch.equal(info.chosen_indices.squeeze(-1), torch.tensor([1, 0]))
    assert torch.allclose(info.chosen_scores.squeeze(-1), torch.tensor([0.9, 0.7]))


def test_beam_select_top_k():
    s = Repr(scores=torch.tensor([[0.1, 0.9, 0.5, 0.7], [0.7, 0.2, 0.4, 0.1]]))
    _, info = BeamSelect(k=2)(evidence=None, s_repr=s)
    assert info.chosen_indices.shape == (2, 2)
    # Row 0: top-2 = (0.9, 0.7) at indices (1, 3)
    assert set(info.chosen_indices[0].tolist()) == {1, 3}


def test_beam_select_clamps_to_available():
    s = Repr(scores=torch.tensor([[0.1, 0.9]]))
    _, info = BeamSelect(k=10)(evidence=None, s_repr=s)
    assert info.chosen_indices.shape == (1, 2)


def test_sample_select_seed_determinism():
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    s = Repr(scores=torch.randn(3, 5))
    _, info1 = SampleSelect(n=2, generator=g1)(evidence=None, s_repr=s)
    _, info2 = SampleSelect(n=2, generator=g2)(evidence=None, s_repr=s)
    assert torch.equal(info1.chosen_indices, info2.chosen_indices)
    assert info1.log_probs is not None
    assert info1.log_probs.shape == (3, 2)
