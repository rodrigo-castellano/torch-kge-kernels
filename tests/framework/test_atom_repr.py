"""AtomRepr tests with a tiny TransE-style backend."""
from __future__ import annotations

import torch

from kge_kernels.framework import (
    KGEBothAtom,
    KGEEmbedAtom,
    KGEScoreAtom,
    MLPAtom,
    RemappedKGEScoreAtom,
)
from kge_kernels.models import TransE


def _build_model(num_preds: int = 5, num_ents: int = 7, dim: int = 8) -> TransE:
    torch.manual_seed(0)
    return TransE(num_entities=num_ents, num_relations=num_preds, dim=dim)


def _indices(B: int = 2, C: int = 3, D: int = 2, M: int = 2):
    g = torch.Generator().manual_seed(1)
    preds = torch.randint(5, (B, C, D, M), generator=g)
    subjs = torch.randint(7, (B, C, D, M), generator=g)
    objs = torch.randint(7, (B, C, D, M), generator=g)
    return preds, subjs, objs


def test_kge_score_atom_shape_and_range():
    model = _build_model()
    preds, subjs, objs = _indices()
    out = KGEScoreAtom(normalize=True)(preds, subjs, objs, model)
    assert out.has_scores and not out.has_embeddings
    assert out.scores.shape == preds.shape
    assert ((out.scores >= 0) & (out.scores <= 1)).all()


def test_kge_score_atom_unnormalized():
    model = _build_model()
    preds, subjs, objs = _indices()
    out = KGEScoreAtom(normalize=False)(preds, subjs, objs, model)
    # TransE returns -|| h+r-t || which is <= 0
    assert (out.scores <= 1e-6).all()


def test_kge_embed_atom_shape():
    model = _build_model()
    preds, subjs, objs = _indices()
    out = KGEEmbedAtom()(preds, subjs, objs, model)
    assert out.has_embeddings and not out.has_scores
    assert out.embeddings.shape == (*preds.shape, model.dim)


def test_kge_both_atom_shapes():
    model = _build_model()
    preds, subjs, objs = _indices()
    out = KGEBothAtom()(preds, subjs, objs, model)
    assert out.has_embeddings and out.has_scores
    assert out.embeddings.shape == (*preds.shape, model.dim)
    assert out.scores.shape == preds.shape


def test_mlp_atom_shape():
    model = _build_model(dim=8)
    preds, subjs, objs = _indices()
    out = MLPAtom(embed_dim=model.dim)(preds, subjs, objs, model)
    assert out.has_embeddings
    assert out.embeddings.shape == (*preds.shape, model.dim)


def test_remapped_kge_score_atom_log_default():
    """Default log=True returns log-scores; OOV → log(0.5)."""
    model = _build_model()
    # Identity remap: caller idx == engine idx for all valid entries.
    const_remap = torch.arange(7)
    pred_remap = torch.arange(5)
    atom = RemappedKGEScoreAtom(const_remap, pred_remap)
    preds = torch.tensor([1, 2])
    subjs = torch.tensor([0, 1])
    objs = torch.tensor([2, 3])
    out = atom(preds, subjs, objs, model)
    assert out.scores.shape == (2,)
    # Log scores are <= 0 (since sigmoid output is in [0, 1]).
    assert (out.scores <= 0).all()


def test_remapped_kge_score_atom_log_false_returns_linear():
    """log=False returns linear sigmoid scores in [0, 1]."""
    model = _build_model()
    const_remap = torch.arange(7)
    pred_remap = torch.arange(5)
    atom = RemappedKGEScoreAtom(const_remap, pred_remap, log=False)
    preds = torch.tensor([1, 2])
    subjs = torch.tensor([0, 1])
    objs = torch.tensor([2, 3])
    out = atom(preds, subjs, objs, model)
    assert out.scores.shape == (2,)
    assert ((out.scores >= 0.0) & (out.scores <= 1.0)).all()


def test_remapped_kge_score_atom_oov_handling():
    """OOV (-1 in remap) → 0.5 (linear) or log(0.5) (log-space)."""
    model = _build_model()
    # Caller idx 0 is OOV in pred_remap.
    pred_remap = torch.tensor([-1, 1, 2, 3, 4])
    const_remap = torch.arange(7)
    atom_lin = RemappedKGEScoreAtom(const_remap, pred_remap, log=False)
    atom_log = RemappedKGEScoreAtom(const_remap, pred_remap, log=True)
    preds = torch.tensor([0, 1])  # 0 is OOV
    subjs = torch.tensor([0, 1])
    objs = torch.tensor([2, 3])
    out_lin = atom_lin(preds, subjs, objs, model)
    out_log = atom_log(preds, subjs, objs, model)
    assert torch.allclose(out_lin.scores[0], torch.tensor(0.5))
    assert torch.allclose(out_log.scores[0], torch.log(torch.tensor(0.5)))
