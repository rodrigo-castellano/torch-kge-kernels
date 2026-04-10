"""AtomRepr implementations: atoms → per-atom Repr.

Each implementation takes ``preds``, ``subjs``, ``objs`` index tensors with
matching shape (typically ``[B, C, D, M]``) and a model object. The model
object can be either a ``KGEModel`` (from ``kge_kernels.models``) or a
``KGEBackend`` (from ``kge_kernels.scoring``); the implementations dispatch
based on the available methods.

Returns a ``Repr`` whose leading shape matches the inputs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .repr import Repr


def _flatten_for_lookup(preds: Tensor, subjs: Tensor, objs: Tensor):
    """Flatten matching index tensors to 1D for embedding lookups."""
    if preds.shape != subjs.shape or preds.shape != objs.shape:
        raise ValueError(
            f"AtomRepr expects matching index shapes; got "
            f"preds={tuple(preds.shape)} subjs={tuple(subjs.shape)} objs={tuple(objs.shape)}"
        )
    leading = tuple(preds.shape)
    return preds.reshape(-1), subjs.reshape(-1), objs.reshape(-1), leading


def _score_triples(model, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Dispatch to whichever scoring method the model exposes."""
    if hasattr(model, "score_triples"):
        return model.score_triples(h, r, t)
    if hasattr(model, "score"):
        return model.score(h, r, t)
    raise AttributeError(f"AtomRepr model {type(model).__name__} has no score_triples/score method")


def _compose(model, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Dispatch to model.compose for fused atom embeddings."""
    if hasattr(model, "compose"):
        return model.compose(h, r, t)
    raise AttributeError(
        f"KGEEmbedAtom requires model.compose; {type(model).__name__} does not provide it"
    )


class KGEScoreAtom(nn.Module):
    """Per-atom score from a KGE backend; produces Repr(scores=…)."""

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def forward(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr:
        r, h, t, leading = _flatten_for_lookup(preds, subjs, objs)
        raw = _score_triples(model, h, r, t)
        if self.normalize:
            raw = torch.sigmoid(raw)
        return Repr(scores=raw.reshape(leading))


class KGEEmbedAtom(nn.Module):
    """Per-atom fused embedding from a KGE model; produces Repr(embeddings=…)."""

    def forward(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr:
        r, h, t, leading = _flatten_for_lookup(preds, subjs, objs)
        emb = _compose(model, h, r, t)            # [N, E]
        return Repr(embeddings=emb.reshape(*leading, emb.shape[-1]))


class KGEBothAtom(nn.Module):
    """Both fused embedding AND scalar score from a KGE model."""

    def __init__(self, normalize_score: bool = True) -> None:
        super().__init__()
        self.normalize_score = normalize_score

    def forward(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr:
        r, h, t, leading = _flatten_for_lookup(preds, subjs, objs)
        emb = _compose(model, h, r, t)            # [N, E]
        sc = _score_triples(model, h, r, t)       # [N]
        if self.normalize_score:
            sc = torch.sigmoid(sc)
        return Repr(
            embeddings=emb.reshape(*leading, emb.shape[-1]),
            scores=sc.reshape(leading),
        )


class MLPAtom(nn.Module):
    """MLP atom embedder over concat(h_emb, r_emb, t_emb).

    The model must expose entity and relation embedding tables (any model
    in ``kge_kernels.models`` qualifies). Returns Repr(embeddings=…).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3 * embed_dim, 2 * embed_dim)
        self.fc2 = nn.Linear(2 * embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _entity_table(model) -> Tensor:
        for name in ("entity_embeddings", "ent"):
            mod = getattr(model, name, None)
            if isinstance(mod, nn.Embedding):
                return mod.weight
        raise AttributeError("MLPAtom requires model.entity_embeddings or model.ent")

    @staticmethod
    def _relation_table(model) -> Tensor:
        for name in ("relation_embeddings", "rel"):
            mod = getattr(model, name, None)
            if isinstance(mod, nn.Embedding):
                return mod.weight
        raise AttributeError("MLPAtom requires model.relation_embeddings or model.rel")

    def forward(self, preds: Tensor, subjs: Tensor, objs: Tensor, model) -> Repr:
        r, h, t, leading = _flatten_for_lookup(preds, subjs, objs)
        ent = self._entity_table(model)
        rel = self._relation_table(model)
        h_emb = F.embedding(h, ent)
        t_emb = F.embedding(t, ent)
        r_emb = F.embedding(r, rel)
        x = torch.cat([h_emb, r_emb, t_emb], dim=-1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.norm(self.fc2(x))
        return Repr(embeddings=x.reshape(*leading, x.shape[-1]))


__all__ = ["KGEBothAtom", "KGEEmbedAtom", "KGEScoreAtom", "MLPAtom"]
