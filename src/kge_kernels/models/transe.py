"""TransE: score = -|| h + r - t ||_p, compose = h + r - t."""
from __future__ import annotations

import math

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

    # ----- KGEModel interface -----

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        diff = self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
        return -torch.norm(diff, p=self.p, dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        hr = self.entity_embeddings(h) + self.relation_embeddings(r)        # [B, dim]
        diff = hr.unsqueeze(1) - self.entity_embeddings.weight.unsqueeze(0)  # [B, E, dim]
        return -torch.norm(diff, p=self.p, dim=-1)

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        rt = self.entity_embeddings(t) - self.relation_embeddings(r)         # [B, dim]
        diff = self.entity_embeddings.weight.unsqueeze(0) - rt.unsqueeze(1)  # [B, E, dim]
        return -torch.norm(diff, p=self.p, dim=-1)

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused TransE embedding: h + r - t."""
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)


__all__ = ["TransE"]
