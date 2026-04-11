"""ConvE: 2D convolution over reshaped ``(entity, relation)`` embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base import KGEModel


class ConvE(KGEModel):
    """ConvE knowledge graph embedding with a 2D conv projection."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        input_dropout: float = 0.2,
        feature_map_dropout: float = 0.2,
        hidden_dropout: float = 0.3,
        embedding_height: int = 10,
        embedding_width: int = 20,
    ) -> None:
        super().__init__()
        if dim != embedding_height * embedding_width:
            raise ValueError(
                f"ConvE dim ({dim}) must equal embedding_height * embedding_width "
                f"({embedding_height} * {embedding_width} = {embedding_height * embedding_width})"
            )
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)
        self.conv1 = nn.Conv2d(1, 32, (3, 3), stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        conv_out_height = 2 * embedding_height - 3 + 1
        conv_out_width = embedding_width - 3 + 1
        self.fc = nn.Linear(32 * conv_out_height * conv_out_width, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.entity_embeddings.weight)
        nn.init.xavier_normal_(self.relation_embeddings.weight)

    def _conv_project(self, h: Tensor, r: Tensor) -> Tensor:
        h_emb = self.entity_embeddings(h).view(-1, 1, self.embedding_height, self.embedding_width)
        r_emb = self.relation_embeddings(r).view(-1, 1, self.embedding_height, self.embedding_width)
        stacked = self.bn0(torch.cat([h_emb, r_emb], dim=2))
        stacked = self.inp_drop(stacked)
        x = F.relu(self.bn1(self.conv1(stacked)))
        x = self.feature_map_drop(x)
        x = F.relu(self.hidden_drop(self.fc(x.view(x.shape[0], -1))))
        return x

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        x = self._conv_project(h, r)
        return (x * self.entity_embeddings(t)).sum(dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        x = self._conv_project(h, r)
        return x @ self.entity_embeddings.weight.T

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        # Brute-force fallback — ConvE's projection is asymmetric in h.
        batch_size = t.shape[0]
        num_ent = self.entity_embeddings.num_embeddings
        device = t.device
        all_h = torch.arange(num_ent, device=device).unsqueeze(0).expand(batch_size, -1)
        h_exp = all_h.reshape(-1)
        r_exp = (
            r.expand(batch_size * num_ent)
            if r.dim() == 0
            else r.unsqueeze(1).expand_as(all_h).reshape(-1)
        )
        t_exp = t.unsqueeze(1).expand_as(all_h).reshape(-1)
        return self.score_triples(h_exp, r_exp, t_exp).view(batch_size, num_ent)

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused ConvE feature: projected (h, r) combined with t element-wise."""
        x = self._conv_project(h, r)
        return x * self.entity_embeddings(t)


__all__ = ["ConvE"]
