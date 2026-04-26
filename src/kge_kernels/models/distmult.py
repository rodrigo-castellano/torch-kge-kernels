"""DistMult: score = < h, r, t > (trilinear product)."""
from __future__ import annotations

import math
from typing import Optional

from torch import Tensor, nn

from .base import KGEModel


class DistMult(KGEModel):
    """DistMult knowledge graph embedding."""

    def __init__(self, num_entities: int, num_relations: int, dim: int) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)

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
            return (self.entity_embeddings(h) * r_emb * self.entity_embeddings(t)).sum(dim=-1)
        if t is None:
            # all-tails matmul fast path
            return (self.entity_embeddings(h) * r_emb) @ self.entity_embeddings.weight.T
        # h is None: all-heads (DistMult is symmetric, same matmul)
        return (self.entity_embeddings(t) * r_emb) @ self.entity_embeddings.weight.T

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused DistMult embedding: h * r * t (element-wise)."""
        return (
            self.entity_embeddings(h)
            * self.relation_embeddings(r)
            * self.entity_embeddings(t)
        )


__all__ = ["DistMult"]
