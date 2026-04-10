"""ModE: modular embeddings.

Score = -|| r * h - t ||
Compose = r * h - t
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .base import KGEModel


class ModE(KGEModel):
    """ModE knowledge graph embedding (modular)."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        p_norm: int = 2,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.p = p_norm
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        diff = self.relation_embeddings(r) * self.entity_embeddings(h) - self.entity_embeddings(t)
        return -torch.norm(diff, p=self.p, dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        rh = self.relation_embeddings(r) * self.entity_embeddings(h)         # [B, dim]
        diff = rh.unsqueeze(1) - self.entity_embeddings.weight.unsqueeze(0)  # [B, E, dim]
        return -torch.norm(diff, p=self.p, dim=-1)

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        # r * h_i for all h_i, then subtract t
        r_emb = self.relation_embeddings(r)                       # [B, dim]
        all_h = self.entity_embeddings.weight.unsqueeze(0)        # [1, E, dim]
        rh = r_emb.unsqueeze(1) * all_h                           # [B, E, dim]
        t_emb = self.entity_embeddings(t).unsqueeze(1)            # [B, 1, dim]
        diff = rh - t_emb
        return -torch.norm(diff, p=self.p, dim=-1)

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused ModE feature: r * h - t."""
        return self.relation_embeddings(r) * self.entity_embeddings(h) - self.entity_embeddings(t)


__all__ = ["ModE"]
