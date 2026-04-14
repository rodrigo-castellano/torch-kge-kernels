"""ComplEx: bilinear model in complex space.

Score = Re(< h, r, conj(t) >)
Compose = real-part of the per-component product (returns a real vector
of size ``dim/2`` so that downstream MLPs see a fixed-width feature).
"""
from __future__ import annotations

import math

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
        self.ent_re = nn.Embedding(num_entities, self.half_dim)
        self.ent_im = nn.Embedding(num_entities, self.half_dim)
        self.rel_re = nn.Embedding(num_relations, self.half_dim)
        self.rel_im = nn.Embedding(num_relations, self.half_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in (self.ent_re, self.ent_im, self.rel_re, self.rel_im):
            nn.init.uniform_(emb.weight, -0.05, 0.05)

    def score_triples(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        s = h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re
        return s.sum(dim=-1)

    def score_all_tails(self, h: Tensor, r: Tensor) -> Tensor:
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        all_re, all_im = self.ent_re.weight, self.ent_im.weight
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
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        all_re, all_im = self.ent_re.weight, self.ent_im.weight
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
        """Fused real-component ComplEx feature.

        Returns the per-component product (length = ``dim``, the
        concatenation of real and imaginary halves) so callers can feed
        a fixed-width feature into downstream MLPs.
        """
        h_re, h_im = self.ent_re(h), self.ent_im(h)
        r_re, r_im = self.rel_re(r), self.rel_im(r)
        t_re, t_im = self.ent_re(t), self.ent_im(t)
        real = h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re
        imag = h_re * r_re * t_im - h_im * r_re * t_re + h_re * r_im * t_re + h_im * r_im * t_im
        return torch.cat([real, imag], dim=-1)


__all__ = ["ComplEx"]
