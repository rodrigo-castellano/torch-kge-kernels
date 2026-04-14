"""ComplEx: bilinear model in complex space.

Score = Re(< h, r, conj(t) >)
Compose = real-part of the per-component product (returns a real vector
of size ``dim/2`` so that downstream MLPs see a fixed-width feature).

Embeddings are stored as single interleaved ``[re | im]`` tensors
(matching torch-ns's proven layout) so that Adam's per-parameter
adaptive statistics cover the full complex vector.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import KGEModel


class ComplEx(KGEModel):
    """Complex-valued bilinear KGE model."""

    def __init__(self, num_entities: int, num_relations: int, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("ComplEx requires even dim (real + imaginary halves)")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.half_dim = dim // 2
        self.entity_embeddings = nn.Embedding(num_entities, dim)
        self.relation_embeddings = nn.Embedding(num_relations, dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.entity_embeddings.weight, -0.05, 0.05)
        nn.init.uniform_(self.relation_embeddings.weight, -0.05, 0.05)

    def _split_ent(self, idx: Tensor):
        emb = self.entity_embeddings(idx)
        return emb[..., :self.half_dim], emb[..., self.half_dim:]

    def _split_rel(self, idx: Tensor):
        emb = self.relation_embeddings(idx)
        return emb[..., :self.half_dim], emb[..., self.half_dim:]

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_re, h_im = self._split_ent(h)
        r_re, r_im = self._split_rel(r)
        t_re, t_im = self._split_ent(t)
        s = h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re
        return s.sum(dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        h_re, h_im = self._split_ent(h)
        r_re, r_im = self._split_rel(r)
        all_re = self.entity_embeddings.weight[:, :self.half_dim]
        all_im = self.entity_embeddings.weight[:, self.half_dim:]
        hr_re_re = h_re * r_re
        hr_im_re = h_im * r_re
        hr_re_im = h_re * r_im
        hr_im_im = h_im * r_im
        return (
            hr_re_re @ all_re.T
            + hr_im_re @ all_im.T
            + hr_re_im @ all_im.T
            - hr_im_im @ all_re.T
        )

    def score_all_heads(self, r: Tensor, t: Tensor) -> Tensor:
        t_re, t_im = self._split_ent(t)
        r_re, r_im = self._split_rel(r)
        all_re = self.entity_embeddings.weight[:, :self.half_dim]
        all_im = self.entity_embeddings.weight[:, self.half_dim:]
        rt_re_re = r_re * t_re
        rt_re_im = r_re * t_im
        rt_im_im = r_im * t_im
        rt_im_re = r_im * t_re
        return (
            rt_re_re @ all_re.T
            + rt_re_im @ all_im.T
            + rt_im_im @ all_re.T
            - rt_im_re @ all_im.T
        )

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused real-component ComplEx feature."""
        h_re, h_im = self._split_ent(h)
        r_re, r_im = self._split_rel(r)
        t_re, t_im = self._split_ent(t)
        real = h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re
        imag = h_re * r_re * t_im - h_im * r_re * t_re + h_re * r_im * t_re + h_im * r_im * t_im
        return torch.cat([real, imag], dim=-1)


__all__ = ["ComplEx"]
