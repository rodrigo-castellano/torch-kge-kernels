"""DirectSearcher: no proof search; score triples directly via AtomRepr.

The baseline against which proof-based methods are compared. Treats
each query ``[N, 3]`` as a single atom and applies ``atom_repr``
directly. If the atom_repr returns scores, those are the output. If
it returns embeddings, they're reduced to scalars via L2 norm by
default.

Used internally by DpRL's "direct_kge" reasoning_mode to bypass proof
search.
"""
from __future__ import annotations

from typing import Any, Dict, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..framework import AtomRepr


class DirectSearcher(nn.Module):
    """Score triples directly via an :class:`AtomRepr` — no search."""

    def __init__(
        self,
        *,
        atom_repr: AtomRepr,
        model: Any = None,
        name: str = "direct",
        embedding_reduce: Literal["norm", "sum"] = "norm",
    ) -> None:
        super().__init__()
        self.atom_repr = atom_repr
        self.model = model
        self.name = name
        self.embedding_reduce = embedding_reduce

    @torch.no_grad()
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        # queries: [N, 3] in (pred, subj, obj) format.
        preds, subjs, objs = queries[:, 0], queries[:, 1], queries[:, 2]
        a_repr = self.atom_repr(preds, subjs, objs, self.model)
        if a_repr.has_scores:
            return {self.name: a_repr.scores}
        # Embedding-only AtomRepr — reduce to scalar.
        emb = a_repr.embeddings
        if self.embedding_reduce == "norm":
            return {self.name: -emb.norm(dim=-1)}
        return {self.name: emb.sum(dim=-1)}


__all__ = ["DirectSearcher"]
