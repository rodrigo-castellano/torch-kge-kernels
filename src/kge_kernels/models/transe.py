"""TransE: score = -|| h + r - t ||_p, compose = h + r - t."""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .base import KGEModel


class TransE(KGEModel):
    """TransE knowledge graph embedding."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        p_norm: int = 1,
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
        bound = 6 / math.sqrt(self.dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)
        self._project_entities()

    @torch.no_grad()
    def _project_entities(self) -> None:
        norm = self.entity_embeddings.weight.data.norm(p=2, dim=-1, keepdim=True)
        self.entity_embeddings.weight.data = (
            self.entity_embeddings.weight.data / torch.clamp(norm, min=1e-6)
        )

    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor],
        *,
        d_chunk: Optional[int] = None,
    ) -> Tensor:
        r_emb = self.relation_embeddings(r)
        if h is not None and t is not None:
            diff = self.entity_embeddings(h) + r_emb - self.entity_embeddings(t)
            return -torch.norm(diff, p=self.p, dim=-1)
        if t is None:
            hr = self.entity_embeddings(h) + r_emb                              # [B, dim]
            diff = hr.unsqueeze(1) - self.entity_embeddings.weight.unsqueeze(0)  # [B, E, dim]
            return -torch.norm(diff, p=self.p, dim=-1)
        # h is None: rank all heads
        rt = self.entity_embeddings(t) - r_emb                                  # [B, dim]
        diff = self.entity_embeddings.weight.unsqueeze(0) - rt.unsqueeze(1)     # [B, E, dim]
        return -torch.norm(diff, p=self.p, dim=-1)

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused TransE embedding: h + r - t."""
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)


__all__ = ["TransE"]
