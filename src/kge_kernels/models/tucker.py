"""TuckER: tensor factorization Score(h,r,t) = W ×_1 e_h ×_2 r ×_3 e_t."""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .base import KGEModel


class TuckER(KGEModel):
    """TuckER tensor factorization model."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        entity_dim: int,
        relation_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_e = entity_dim
        d_r = relation_dim if relation_dim is not None else entity_dim
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = d_e
        self.entity_dim = d_e
        self.relation_dim = d_r
        self.ent = nn.Embedding(num_entities, d_e)
        self.rel = nn.Embedding(num_relations, d_r)
        self.W = nn.Parameter(torch.empty(d_r, d_e, d_e))
        self.dropout_e = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_r = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.xavier_uniform_(self.W)

    def _project(self, h: Tensor, r: Tensor) -> Tensor:
        e_r = self.dropout_r(self.rel(r))
        e_h = self.dropout_e(self.ent(h))
        Wr = torch.tensordot(e_r, self.W, dims=([1], [0]))   # [B, d_e, d_e]
        return torch.bmm(e_h.unsqueeze(1), Wr).squeeze(1)    # [B, d_e]

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        x = self._project(h, r)
        e_t = self.dropout_e(self.ent(t))
        return (x * e_t).sum(dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        e_r = self.dropout_r(self.rel(r))
        e_h = self.ent(h)
        Wr = torch.tensordot(e_r, self.W, dims=([1], [0]))   # [B, d_e, d_e]
        x = torch.bmm(e_h.unsqueeze(1), Wr).squeeze(1)       # [B, d_e]
        return x @ self.ent.weight.T

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        e_r = self.dropout_r(self.rel(r))
        e_t = self.ent(t)
        Wr = torch.tensordot(e_r, self.W, dims=([1], [0]))   # [B, d_e, d_e]
        y = torch.bmm(Wr, e_t.unsqueeze(-1)).squeeze(-1)     # [B, d_e]
        return y @ self.ent.weight.T

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused TuckER feature: projected h then element-wise with t."""
        x = self._project(h, r)
        return x * self.ent(t)


__all__ = ["TuckER"]
