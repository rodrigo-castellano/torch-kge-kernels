"""RotatE: complex rotation, score = gamma - || (h . r) - t ||_p.

Two variants share this file:

* :class:`RotatE` — the original tkk implementation. Relations are stored
  as wrapped phase angles in ``rel_phase`` (init uniform in ``±π``);
  entities init in ``±6 / √half_dim``. All existing checkpoints under
  this name continue to load unchanged.

* :class:`RotatENS` — ns-old-aligned variant. Relations are stored as
  raw values in ``relation_embeddings`` and multiplied by ``norm_factor =
  π / embedding_range`` in forward; entities init in ``±6 / √half_dim``,
  relations init in ``±embedding_range`` with ``embedding_range =
  (γ + ε) / half_dim``. The ``norm_factor`` scaling acts as a
  per-parameter lr multiplier for relations — critical for RotatE
  training on large KGs (wn18rr).

The two variants share a common base class :class:`_RotateBase` for the
parts that are identical (entity setup, single ``score`` dispatcher,
``compose``). They diverge on relation storage, ``_hr`` phase source,
``_dist`` numerics, the all-heads phase source, and the d-chunked L2
path (RotatE does per-component sqrt then sum; RotatENS accumulates
squared diffs and applies a single final sqrt).

Embeddings are stored as single interleaved ``[re | im]`` tensors
(matching torch-ns's proven layout) so that Adam's per-parameter
adaptive statistics cover the full complex vector.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from . import ops
from .base import KGEModel


class _RotateMemoryMixin:
    """Memory-aware eval batch size for RotatE-family models.

    All-tails / all-heads broadcast to a ``[B, |E|, half_dim]``
    intermediate — one dimension larger than the default matmul-style
    ``[B, |E|]`` assumed by the base class. Override the budget to
    include that factor so callers in the unified eval path (ns, tkk)
    don't OOM on rotate-style models with many entities.

    On family (|E| ≈ 3k, half_dim = 100, budget 2 GB) this returns
    B ≤ ~1700; on wn18rr (|E| ≈ 40k) it returns B ≤ ~130.
    """

    def recommended_eval_batch_size(
        self, num_entities: int, budget_gb: float = 2.0,
    ) -> int:
        half_dim = getattr(self, "half_dim", None) or (self.dim // 2)
        bytes_per_B = 4 * int(num_entities) * int(half_dim)  # [B, E, H] float32
        budget_b = budget_gb * (1 << 30)
        return max(8, min(2048, int(budget_b / max(bytes_per_B, 1))))


class _RotateLossMixin:
    """Train-step hook override for RotatE variants: sigmoid+BCE instead
    of ``binary_cross_entropy_with_logits``. The raw RotatE score is
    ``gamma - ||h rot r - t||`` which can be large in magnitude; fusing
    sigmoid inside BCE saturates and zeroes the gradient for confident
    negatives. Explicit sigmoid + BCE matches ns-old's training dynamics
    and tkk's pipeline ``_bce_fn`` behavior for RotatE/RotatENS.
    """
    _train_loss_is_from_logits = False


class _RotateBase(_RotateMemoryMixin, _RotateLossMixin, KGEModel):
    """Shared RotatE plumbing.

    Subclasses must set ``self.relation_embeddings`` (or ``rel_phase``)
    in their own ``__init__`` and implement:
      - ``_hr(h, r) -> (re, im)``: rotated head embedding.
      - ``_phase(r) -> phase`` for the all-heads path (same scaling
        convention as ``_hr`` uses internally).
      - ``_dist(a_re, a_im, b_re, b_im) -> Tensor``: distance reduction.
      - ``_dist_sq_to_dist(dist_sq) -> Tensor``: optional terminal
        sqrt for the chunked L2 path. Default = identity (RotatE-style
        per-chunk sqrt-then-sum). RotatENS overrides to do a single
        final sqrt over accumulated squared diffs.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma: float = 12.0,
        p_norm: int = 1,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"{type(self).__name__} requires even dim (real + imaginary halves)")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.half_dim = dim // 2
        self.gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.get_default_dtype()), requires_grad=False
        )
        self.p = p_norm
        self.entity_embeddings = nn.Embedding(num_entities, dim)

    # ---- subclass hooks ----------------------------------------------------

    def _hr(self, h: Tensor, r: Tensor):
        raise NotImplementedError

    def _phase(self, r: Tensor) -> Tensor:
        raise NotImplementedError

    def _dist(self, a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tensor:
        raise NotImplementedError

    # Default: per-chunk sqrt-then-sum (matches RotatE legacy). RotatENS
    # overrides to accumulate squared diffs and apply one terminal sqrt.
    _accumulate_squared_in_chunk: bool = False

    # ---- shared utilities --------------------------------------------------

    @torch.no_grad()
    def _clamp_entity_modulus(self) -> None:
        """Clamp entity embeddings to the unit disk in complex space.

        Called once at initialisation to match torch-ns's init.
        """
        w = self.entity_embeddings.weight.data
        re, im = w[:, :self.half_dim], w[:, self.half_dim:]
        mod = torch.clamp(torch.sqrt(re * re + im * im), min=1e-6)
        factor = torch.clamp(1.0 / mod, max=1.0)
        re.mul_(factor)
        im.mul_(factor)

    def _split_ent(self, idx: Tensor):
        return ops.complex_split(self.entity_embeddings(idx))

    # ---- unified scoring ---------------------------------------------------

    def score(
        self,
        h: Optional[Tensor],
        r: Tensor,
        t: Optional[Tensor],
        *,
        d_chunk: Optional[int] = None,
    ) -> Tensor:
        if h is not None and t is not None:
            hr_re, hr_im = self._hr(h, r)
            t_re, t_im = self._split_ent(t)
            return self.gamma - self._dist(hr_re, hr_im, t_re, t_im)

        if t is None:
            # rank all tails for (h, r)
            if d_chunk is not None:
                return self._score_all_tails_dchunked(h, r, d_chunk)
            hr_re, hr_im = self._hr(h, r)
            all_re = self.entity_embeddings.weight[:, :self.half_dim]
            all_im = self.entity_embeddings.weight[:, self.half_dim:]
            return self.gamma - self._dist(
                hr_re.unsqueeze(1), hr_im.unsqueeze(1),
                all_re.unsqueeze(0), all_im.unsqueeze(0),
            )

        # h is None: rank all heads for (r, t)
        if d_chunk is not None:
            return self._score_all_heads_dchunked(r, t, d_chunk)
        t_re, t_im = self._split_ent(t)
        phase = self._phase(r)
        c, s = torch.cos(phase).unsqueeze(1), torch.sin(phase).unsqueeze(1)
        all_re = self.entity_embeddings.weight[:, :self.half_dim].unsqueeze(0)
        all_im = self.entity_embeddings.weight[:, self.half_dim:].unsqueeze(0)
        all_hr_re = all_re * c - all_im * s
        all_hr_im = all_re * s + all_im * c
        return self.gamma - self._dist(
            all_hr_re, all_hr_im, t_re.unsqueeze(1), t_im.unsqueeze(1),
        )

    def _score_all_tails_dchunked(self, h: Tensor, r: Tensor, d_chunk: int) -> Tensor:
        """Chunk the half-dim axis to cap peak memory in all-tails mode.

        Peak intermediate is ``[K, E, d_chunk]`` rather than
        ``[K, E, half_dim]``. Exact result because ``_dist`` is a sum
        over the D axis (L1: elementwise abs + sum; L2: see
        ``_accumulate_squared_in_chunk``).
        """
        hr_re, hr_im = self._hr(h, r)
        all_re_full = self.entity_embeddings.weight[:, :self.half_dim]
        all_im_full = self.entity_embeddings.weight[:, self.half_dim:]
        K = hr_re.shape[0]
        E = all_re_full.shape[0]
        H = self.half_dim
        dist = torch.zeros(K, E, device=hr_re.device, dtype=hr_re.dtype)
        for d_start in range(0, H, d_chunk):
            d_end = min(d_start + d_chunk, H)
            diff_re = hr_re[:, d_start:d_end].unsqueeze(1) - all_re_full[:, d_start:d_end].unsqueeze(0)
            diff_im = hr_im[:, d_start:d_end].unsqueeze(1) - all_im_full[:, d_start:d_end].unsqueeze(0)
            if self.p == 1:
                dist = dist + diff_re.abs().sum(dim=-1) + diff_im.abs().sum(dim=-1)
            elif self._accumulate_squared_in_chunk:
                dist = dist + (diff_re * diff_re + diff_im * diff_im).sum(dim=-1)
            else:
                dist = dist + torch.sqrt(diff_re * diff_re + diff_im * diff_im + 1e-9).sum(dim=-1)
        if self.p != 1 and self._accumulate_squared_in_chunk:
            dist = torch.sqrt(torch.clamp(dist, min=1e-9))
        return self.gamma - dist

    def _score_all_heads_dchunked(self, r: Tensor, t: Tensor, d_chunk: int) -> Tensor:
        """Chunk the half-dim axis in all-heads mode. See
        :meth:`_score_all_tails_dchunked` for the rationale."""
        t_re, t_im = self._split_ent(t)
        phase = self._phase(r)
        all_re_full = self.entity_embeddings.weight[:, :self.half_dim]
        all_im_full = self.entity_embeddings.weight[:, self.half_dim:]
        K = t_re.shape[0]
        E = all_re_full.shape[0]
        H = self.half_dim
        dist = torch.zeros(K, E, device=t_re.device, dtype=t_re.dtype)
        for d_start in range(0, H, d_chunk):
            d_end = min(d_start + d_chunk, H)
            c = torch.cos(phase[:, d_start:d_end]).unsqueeze(1)
            s = torch.sin(phase[:, d_start:d_end]).unsqueeze(1)
            all_re_c = all_re_full[:, d_start:d_end].unsqueeze(0)
            all_im_c = all_im_full[:, d_start:d_end].unsqueeze(0)
            hr_re_c = all_re_c * c - all_im_c * s
            hr_im_c = all_re_c * s + all_im_c * c
            diff_re = hr_re_c - t_re[:, d_start:d_end].unsqueeze(1)
            diff_im = hr_im_c - t_im[:, d_start:d_end].unsqueeze(1)
            if self.p == 1:
                dist = dist + diff_re.abs().sum(dim=-1) + diff_im.abs().sum(dim=-1)
            elif self._accumulate_squared_in_chunk:
                dist = dist + (diff_re * diff_re + diff_im * diff_im).sum(dim=-1)
            else:
                dist = dist + torch.sqrt(diff_re * diff_re + diff_im * diff_im + 1e-9).sum(dim=-1)
        if self.p != 1 and self._accumulate_squared_in_chunk:
            dist = torch.sqrt(torch.clamp(dist, min=1e-9))
        return self.gamma - dist

    def compose(self, h: Tensor, r: Tensor, t: Tensor) -> Tensor:
        """Fused RotatE feature: concatenated (rotated h - t) real/imag."""
        hr_re, hr_im = self._hr(h, r)
        t_re, t_im = self._split_ent(t)
        return torch.cat([hr_re - t_re, hr_im - t_im], dim=-1)


class RotatE(_RotateBase):
    """RotatE knowledge graph embedding (complex rotation).

    Relations stored as wrapped phase angles (``rel_phase``); distance is
    the legacy per-component ``Σ sqrt(re² + im²)`` for L2 — preserved
    for checkpoint compatibility with the original tkk training runs.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma: float = 12.0,
        p_norm: int = 1,
    ) -> None:
        super().__init__(num_entities, num_relations, dim, gamma=gamma, p_norm=p_norm)
        self.rel_phase = nn.Embedding(num_relations, self.half_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 6 / math.sqrt(self.half_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.rel_phase.weight, -math.pi, math.pi)
        self._clamp_entity_modulus()

    def _phase(self, r: Tensor) -> Tensor:
        # Legacy RotatE wraps stored phases (in ``[-π, π]``) to
        # ``[0, 2π]`` before rotation so cos/sin match the pre-wrap
        # convention existing checkpoints trained under.
        return torch.remainder(self.rel_phase(r), 2 * math.pi)

    def _hr(self, h: Tensor, r: Tensor):
        return ops.rotate_apply(self.entity_embeddings(h), self._phase(r), norm_factor=1.0)

    def _dist(self, a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tensor:
        """Legacy-RotatE distance: per-component-sum-after-per-pair-sqrt.

        Kept as-is (rather than switching to :func:`ops.complex_dist`)
        because this variant is the one used by existing tkk checkpoints.
        ``RotatENS`` uses the Euclidean variant via ``ops.complex_dist``.
        """
        if self.p == 1:
            return ((a_re - b_re).abs() + (a_im - b_im).abs()).sum(dim=-1)
        return torch.sqrt(((a_re - b_re) ** 2 + (a_im - b_im) ** 2) + 1e-9).sum(dim=-1)


class RotatENS(_RotateBase):
    """ns-old-aligned RotatE variant.

    Relations stored as raw values (``relation_embeddings``), scaled by
    ``norm_factor = π / embedding_range`` in forward. The scaling acts as a
    per-parameter lr multiplier for relations — critical for RotatE training
    on large KGs (wn18rr). Distance is true Euclidean via ``ops.complex_dist``,
    and the chunked-D L2 path accumulates squared diffs with one terminal
    sqrt (rather than per-chunk sqrt-then-sum).
    """

    _accumulate_squared_in_chunk = True

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        dim: int,
        gamma: float = 12.0,
        p_norm: int = 1,
    ) -> None:
        super().__init__(num_entities, num_relations, dim, gamma=gamma, p_norm=p_norm)
        self.relation_embeddings = nn.Embedding(num_relations, self.half_dim)
        epsilon = 0.5
        self.embedding_range = (float(gamma) + epsilon) / self.half_dim
        self.norm_factor = math.pi / self.embedding_range
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Entity init follows the standard RotatE paper / ns-old convention
        # (``±6/√half_dim``) rather than ``±embedding_range``. This matches
        # torch-ns's ``_reinit_rotate_entity_`` override exactly — so
        # ``ns --kge rotate`` and ``tkk model=rotate_ns`` produce
        # bit-identical weights at the same seed.
        entity_bound = 6.0 / math.sqrt(self.half_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -entity_bound, entity_bound)
        nn.init.uniform_(self.relation_embeddings.weight,
                         -self.embedding_range, self.embedding_range)
        self._clamp_entity_modulus()

    def _phase(self, r: Tensor) -> Tensor:
        return self.relation_embeddings(r) * self.norm_factor

    def _hr(self, h: Tensor, r: Tensor):
        return ops.rotate_apply(
            self.entity_embeddings(h),
            self.relation_embeddings(r),
            norm_factor=self.norm_factor,
        )

    def _dist(self, a_re: Tensor, a_im: Tensor, b_re: Tensor, b_im: Tensor) -> Tensor:
        return ops.complex_dist(a_re - b_re, a_im - b_im, p=self.p)


__all__ = ["RotatE", "RotatENS"]
