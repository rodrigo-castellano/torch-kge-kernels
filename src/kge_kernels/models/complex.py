"""ComplEx: bilinear model in complex space.

Score = Re(< h, r, conj(t) >)
Compose = real-part of the per-component product (returns a real vector
of size ``dim/2`` so that downstream MLPs see a fixed-width feature).

Embeddings are stored as single interleaved ``[re | im]`` tensors
(matching torch-ns's proven layout) so that Adam's per-parameter
adaptive statistics cover the full complex vector.

The arithmetic lives in :mod:`kge_kernels.models.ops` and is shared
with DpRL's pre-embedded atom embedders — keeping one source of truth
for the complex Hermitian product.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from . import ops
from .base import KGEModel, det_embedding


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

    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor],
        *,
        d_chunk: Optional[int] = None,
    ) -> Tensor:
        if h is not None and t is not None:
            return ops.complex_hermitian_real_vec(
                det_embedding(self.entity_embeddings, h),
                det_embedding(self.relation_embeddings, r),
                det_embedding(self.entity_embeddings, t),
            ).sum(dim=-1)

        r_re, r_im = ops.complex_split(det_embedding(self.relation_embeddings, r))
        all_re = self.entity_embeddings.weight[:, :self.half_dim]
        all_im = self.entity_embeddings.weight[:, self.half_dim:]
        if t is None:
            # all-tails matmul fast path
            h_re, h_im = ops.complex_split(det_embedding(self.entity_embeddings, h))
            return (
                (h_re * r_re) @ all_re.T
                + (h_im * r_re) @ all_im.T
                + (h_re * r_im) @ all_im.T
                - (h_im * r_im) @ all_re.T
            )
        # h is None: all-heads matmul fast path (different sign pattern from conj on t)
        t_re, t_im = ops.complex_split(det_embedding(self.entity_embeddings, t))
        return (
            (r_re * t_re) @ all_re.T
            + (r_re * t_im) @ all_im.T
            + (r_im * t_im) @ all_re.T
            - (r_im * t_re) @ all_im.T
        )

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused real+imag ComplEx feature (concatenated)."""
        h_emb = det_embedding(self.entity_embeddings, h)
        r_emb = det_embedding(self.relation_embeddings, r)
        t_emb = det_embedding(self.entity_embeddings, t)
        return torch.cat(
            [
                ops.complex_hermitian_real_vec(h_emb, r_emb, t_emb),
                ops.complex_hermitian_imag_vec(h_emb, r_emb, t_emb),
            ],
            dim=-1,
        )


__all__ = ["ComplEx"]
