"""Fully vectorized corruption sampler shared across projects.

Public surface:

- :class:`Sampler` — head / tail / both corruption with optional filter,
  domain pools, and validity-mask return.
- :class:`BernoulliSampler` — per-triple coin-flip between head and
  tail corruption (Bordes et al. trick); uses per-relation
  head-corruption probabilities. Has a ``compute_probs`` staticmethod
  that builds the probabilities from a known-triple set.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor as _Tensor

from .types import LongTensor, SamplerConfig, Tensor


# ============================================================================
# Internal hashing utilities
# ============================================================================


def _mix_hash(triples: LongTensor, b_e: int, b_r: int) -> LongTensor:
    """Pack ``(r, h, t)`` triples into sortable 64-bit hash values.

    The arithmetic bases are provided by the caller so the mapping stays
    unique for that caller's entity and relation ranges without relying
    on fixed bit widths.
    """
    h = triples[..., 1].to(dtype=triples.dtype)
    r = triples[..., 0].to(dtype=triples.dtype)
    t = triples[..., 2].to(dtype=triples.dtype)
    return ((h * b_r) + r) * b_e + t


class Sampler:
    """Fully vectorized head / tail / both corruption sampler.

    Public tensors are always in ``(relation, head, tail)`` format.

    For Bordes-style per-triple Bernoulli coin-flip corruption, use
    :class:`BernoulliSampler` instead.
    """

    def __init__(self, cfg: SamplerConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.num_entities = cfg.num_entities
        self.num_relations = cfg.num_relations
        self.default_mode = cfg.default_mode
        self.order_negatives = cfg.order_negatives
        self.min_entity_idx = cfg.min_entity_idx
        self._filter_hashes_sorted: Optional[LongTensor] = None
        self.b_e = max(2 * self.num_entities + 1, 1024)
        self.b_r = max(2 * self.num_relations + 1, 128)
        self.domain_padded: Optional[Tensor] = None
        self.domain_len: Optional[Tensor] = None
        self.ent2dom: Optional[LongTensor] = None
        self.pos_in_dom: Optional[LongTensor] = None
        self.num_domains = 0
        self.max_pool_len = 0

    @classmethod
    def from_data(
        cls,
        all_known_triples_idx: Optional[LongTensor],
        num_entities: int,
        num_relations: int,
        device: torch.device,
        default_mode: Literal["head", "tail", "both"] = "both",
        seed: int = 0,
        domain2idx: Optional[Dict[str, List[int]]] = None,
        entity2domain: Optional[Dict[int, str]] = None,
        order_negatives: bool = False,
        min_entity_idx: int = 1,
    ) -> "Sampler":
        """Construct a sampler from known triples and optional domain pools.

        Args:
            all_known_triples_idx: Known positive triples in ``(r, h, t)``
                format. Used for filtered corruption generation.
            num_entities: Number of entity ids available to corrupt with.
            num_relations: Number of relation ids.
            device: Device where the internal lookup tensors will live.
            default_mode: Default corruption side used by ``corrupt``.
            seed: Stored seed for downstream callers; runtime sampling remains
                purely tensor-based.
            domain2idx: Optional domain-to-entity mapping for typed corruption.
            entity2domain: Optional inverse mapping used with ``domain2idx``.
            order_negatives: Whether to sort valid negatives deterministically.
            min_entity_idx: Smallest valid entity id. Use ``0`` when entity 0
                is real data rather than padding.
        """
        cfg = SamplerConfig(
            num_entities, num_relations, device, default_mode, seed,
            order_negatives, min_entity_idx,
        )
        self = cls(cfg)
        if all_known_triples_idx is not None and all_known_triples_idx.numel() > 0:
            hashes = _mix_hash(all_known_triples_idx.detach().to(device=self.device, dtype=torch.long), self.b_e, self.b_r)
            self._filter_hashes_sorted = torch.sort(torch.unique(hashes)).values
        else:
            self._filter_hashes_sorted = torch.empty((0,), dtype=torch.long, device=self.device)
        if domain2idx and entity2domain:
            self._build_domain_structures(domain2idx, entity2domain, device)
        return self

    def _build_domain_structures(
        self,
        domain2idx: Dict[str, List[int]],
        entity2domain: Dict[int, str],
        device: torch.device,
    ) -> None:
        """Materialize padded domain pools for compiled-friendly sampling."""
        domain_names = sorted(domain2idx.keys())
        domain_lists = [torch.tensor(domain2idx[n], dtype=torch.int32, device=device) for n in domain_names]
        self.num_domains = len(domain_lists)
        self.max_pool_len = max((t.numel() for t in domain_lists), default=0)
        self.domain_padded = torch.zeros((self.num_domains, self.max_pool_len), dtype=torch.int32, device=device)
        self.domain_len = torch.zeros((self.num_domains,), dtype=torch.int32, device=device)
        for i, tensor in enumerate(domain_lists):
            self.domain_padded[i, : tensor.numel()] = tensor
            self.domain_len[i] = tensor.numel()
        max_ent = max(entity2domain.keys(), default=0)
        self.ent2dom = torch.full((max_ent + 1,), -1, dtype=torch.int32, device=device)
        self.pos_in_dom = torch.zeros((max_ent + 1,), dtype=torch.int32, device=device)
        for dom_idx, name in enumerate(domain_names):
            row = self.domain_padded[dom_idx, : self.domain_len[dom_idx]]
            if row.numel() > 0:
                self.ent2dom[row.long()] = dom_idx
                self.pos_in_dom[row.long()] = torch.arange(row.numel(), device=device, dtype=torch.int32)

    def _has_domain_info(self) -> bool:
        """Return ``True`` when typed/domain-aware corruption is available."""
        return self.domain_padded is not None and self.num_domains > 0

    # ---- per-column corruption primitives (dispatched from .corrupt) -----

    def _corrupt_relation_col(
        self,
        pos_hrt: LongTensor,
        count: int,
        is_exhaustive: bool,
        device: torch.device,
    ) -> LongTensor:
        """Generate relation corruptions in internal ``(h, r, t)`` layout."""
        batch_size = pos_hrt.shape[0]
        if is_exhaustive:
            all_rels = torch.arange(self.num_relations, device=device, dtype=pos_hrt.dtype)
            all_rels = all_rels.unsqueeze(0).expand(batch_size, -1)
            orig_rel = pos_hrt[:, 1:2]
            mask = all_rels != orig_rel
            indices = mask.long().cumsum(dim=1) - 1
            indices = indices.clamp(min=0, max=count - 1)
            return torch.gather(all_rels, 1, indices[:, :count])
        orig = pos_hrt[:, 1:2].expand(-1, count)
        rnd = torch.randint(0, self.num_relations - 1, (batch_size, count), device=device)
        return rnd + (rnd >= orig)

    def _corrupt_entity_global(
        self,
        pos_hrt: LongTensor,
        col: int,
        count: int,
        is_exhaustive: bool,
        device: torch.device,
    ) -> LongTensor:
        """Generate entity corruptions from the global entity pool."""
        batch_size = pos_hrt.shape[0]
        pool_size = self.num_entities
        lo = self.min_entity_idx
        orig_ent = pos_hrt[:, col : col + 1]

        if is_exhaustive:
            all_ents = torch.arange(lo, lo + pool_size, device=device, dtype=pos_hrt.dtype)
            all_ents = all_ents.unsqueeze(0).expand(batch_size, -1)
            mask = all_ents != orig_ent
            sort_keys = (~mask).long()
            _, perm = torch.sort(sort_keys, dim=1, stable=True)
            sorted_ents = torch.gather(all_ents, 1, perm)
            return sorted_ents[:, :count]

        rnd = torch.randint(lo, lo + pool_size - 1, (batch_size, count), device=device)
        return rnd + ((rnd >= orig_ent) & (orig_ent >= lo)).long()

    def _corrupt_entity_domain(
        self,
        pos_hrt: LongTensor,
        col: int,
        count: int,
        is_exhaustive: bool,
        orig_slice: LongTensor,
        device: torch.device,
    ) -> LongTensor:
        """Generate entity corruptions restricted to each example's domain."""
        orig_ent = pos_hrt[:, col]
        d_ids = self.ent2dom[orig_ent].long()
        pools = self.domain_padded[d_ids]
        lo = self.min_entity_idx
        lengths = self.domain_len[d_ids].long()
        positions = torch.arange(pools.shape[1], device=device).unsqueeze(0)
        in_domain = positions < lengths.unsqueeze(1)
        valid_mask = in_domain & (pools != orig_ent.unsqueeze(1)) & (pools >= lo)

        if is_exhaustive:
            sort_keys = (~valid_mask).long()
            _, perm = torch.sort(sort_keys, dim=1, stable=True)
            sorted_pools = torch.gather(pools, 1, perm)
            sorted_valid = torch.gather(valid_mask, 1, perm)
            result = sorted_pools[:, :count]
            result.masked_fill_(~sorted_valid[:, :count], -1)
            return result

        orig_flat = orig_slice.reshape(-1)
        valid = orig_flat >= lo
        result_flat = orig_flat.clone()
        safe_orig = torch.where(valid, orig_flat, torch.full_like(orig_flat, lo))
        d_flat = self.ent2dom[safe_orig].long()
        lengths = self.domain_len[d_flat]
        positions = self.pos_in_dom[safe_orig].long()
        has_alternatives = lengths > 1

        pool_len_m1 = (lengths - 1).float().clamp(min=0)
        rnd = torch.floor(torch.rand(orig_flat.shape, device=device) * pool_len_m1).long()
        shifted_idx = rnd + (rnd >= positions)

        sampled_ents = self.domain_padded[d_flat, shifted_idx].long()
        mask = valid & has_alternatives
        result_flat = torch.where(mask, sampled_ents, result_flat)
        return result_flat.reshape(orig_slice.shape)

    # ---- filter / postprocess --------------------------------------------

    def _filter_mask_batched(self, triples: LongTensor) -> torch.BoolTensor:
        """Return a keep-mask for batched triples against known positives."""
        if self._filter_hashes_sorted is None or self._filter_hashes_sorted.numel() == 0:
            return torch.ones(triples.shape[:-1], dtype=torch.bool, device=triples.device)
        batch_size, k = triples.shape[:2]
        flat = triples.reshape(-1, 3)
        hashes = _mix_hash(flat, self.b_e, self.b_r)

        target = self._filter_hashes_sorted
        pos = torch.searchsorted(target, hashes)
        in_range = pos < target.numel()
        safe_pos = pos.clamp(min=0, max=target.numel() - 1)
        eq = in_range & (target[safe_pos] == hashes)
        return (~eq).reshape(batch_size, k)

    def _postprocess(
        self,
        neg: LongTensor,
        batch_size: int,
        k: int,
        do_filter: bool,
        do_unique: bool,
        device: torch.device,
    ) -> Tuple[LongTensor, torch.BoolTensor]:
        """Convert internal triples to public ``(r, h, t)`` and compact them."""
        neg_rht = torch.stack([neg[:, :, 1], neg[:, :, 0], neg[:, :, 2]], dim=-1)
        lo = self.min_entity_idx
        valid = (neg_rht[:, :, 1] >= lo) & (neg_rht[:, :, 2] >= lo)

        if do_filter:
            valid = valid & self._filter_mask_batched(neg_rht)

        # Entity corruption draws with replacement, so when the (post-filter)
        # candidate pool is smaller than k the row contains duplicate
        # triples. Mark all but the first occurrence of each (r, h, t)
        # tuple as invalid so callers asking for unique=True get k
        # unique negatives or zero-padding when the pool is exhausted.
        # Uses an O(B*k^2) pairwise comparison rather than sort + scatter
        # because `torch.argsort` on CUDA is non-deterministic even under
        # ``use_deterministic_algorithms(True, warn_only=True)`` and the
        # regression suite needs bit-stable MRR across reruns at the same
        # seed.
        if do_unique and k > 1:
            keys = (
                neg_rht[:, :, 0] * 10_000_000
                + neg_rht[:, :, 1] * 10_000
                + neg_rht[:, :, 2]
            )
            slot_offsets = torch.arange(
                k, device=device, dtype=keys.dtype
            ).unsqueeze(0)
            INVALID_KEY = 1 << 62
            keys = torch.where(valid, keys, INVALID_KEY + slot_offsets)
            # pairwise_eq[i, j, j'] = (keys[i, j] == keys[i, j'])
            pairwise_eq = keys.unsqueeze(2) == keys.unsqueeze(1)
            # strict_lower[j, j'] = True iff j' < j
            strict_lower = torch.tril(
                torch.ones(k, k, dtype=torch.bool, device=device), diagonal=-1
            )
            is_dup = (pairwise_eq & strict_lower).any(dim=2)
            valid = valid & ~is_dup

        neg_rht = neg_rht * valid.unsqueeze(-1)

        sort_key = (~valid).long()
        _, perm = torch.sort(sort_key, dim=1, stable=True)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)
        neg_rht = neg_rht[batch_idx, perm]
        valid = valid[batch_idx, perm]

        if self.order_negatives and k > 0:
            keys = neg_rht[:, :, 0] * 10_000_000 + neg_rht[:, :, 1] * 10_000 + neg_rht[:, :, 2]
            keys = keys.masked_fill(neg_rht.sum(-1) == 0, 2**62)
            _, idx = torch.sort(keys, dim=1)
            neg_rht = neg_rht[batch_idx, idx]
            valid = valid[batch_idx, idx]

        return neg_rht, valid

    # ---- canonical entry point -------------------------------------------

    def corrupt(
        self,
        positives: LongTensor,
        *,
        num_negatives: Optional[int] = None,
        mode: Optional[Literal["head", "tail", "both"]] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
        return_mask: bool = False,
    ) -> Union[LongTensor, Tuple[LongTensor, torch.BoolTensor]]:
        """Generate corruptions, optionally with an explicit validity mask.

        Args:
            positives: Positive triples in public ``(r, h, t)`` format.
            num_negatives: Number of negatives per positive. ``None`` means
                exhaustive corruption over the active candidate pool.
            mode: Which side to corrupt: ``head``, ``tail``, or ``both``.
                Defaults to the sampler's ``default_mode``.
            device: Optional override for the runtime device.
            filter: Remove known positives using the indexed fact set.
            unique: Deduplicate negatives within each row after filtering.
            return_mask: If True, return ``(negatives, valid_mask)`` —
                needed by mask-aware loss/eval reductions where padded
                or filtered slots must be gated out. If False (default),
                return just the negatives tensor.

        Returns:
            If ``return_mask=False`` (default): ``negatives`` of shape
            ``[B, K, 3]`` in ``(r, h, t)`` format.
            If ``return_mask=True``: ``(negatives, valid_mask)`` where
            ``valid_mask`` has shape ``[B, K]`` and is ``True`` for slots
            holding a real, filtered, unique corruption.
        """
        device = device or self.device
        mode = mode or self.default_mode
        pos = positives.to(device=device)
        batch_size = pos.shape[0]

        if batch_size == 0:
            empty_neg = torch.zeros((0, num_negatives or 0, 3), dtype=pos.dtype, device=device)
            empty_valid = torch.zeros((0, num_negatives or 0), dtype=torch.bool, device=device)
            return (empty_neg, empty_valid) if return_mask else empty_neg

        pos_hrt = torch.stack([pos[:, 1], pos[:, 0], pos[:, 2]], dim=1)
        # Map mode → internal (h, r, t) column list.
        cols = [0] if mode == "head" else [2] if mode == "tail" else [0, 2]
        is_exhaustive = num_negatives is None

        pool_size = (self.max_pool_len - 1) if self._has_domain_info() else (self.num_entities - 1)
        k = pool_size * len(cols) if is_exhaustive else num_negatives
        neg = pos_hrt.unsqueeze(1).expand(batch_size, k, 3).clone()

        per_col = k // len(cols)
        remainder = k % len(cols)
        start = 0
        for idx, col in enumerate(cols):
            count = per_col + (1 if idx < remainder else 0)
            if count == 0:
                continue
            if col == 1:
                result = self._corrupt_relation_col(pos_hrt, count, is_exhaustive, device)
            elif not self._has_domain_info():
                result = self._corrupt_entity_global(pos_hrt, col, count, is_exhaustive, device)
            else:
                orig_slice = neg[:, start : start + count, col]
                result = self._corrupt_entity_domain(pos_hrt, col, count, is_exhaustive, orig_slice, device)
            neg[:, start : start + count, col] = result
            start += count

        neg_rht, valid = self._postprocess(neg, batch_size, k, do_filter=filter, do_unique=unique, device=device)
        return (neg_rht, valid) if return_mask else neg_rht


class BernoulliSampler(Sampler):
    """Bernoulli-mode corruption sampler (Bordes et al. trick).

    Per-triple coin flip selects the corruption side: with probability
    ``bern_probs[r]`` the head is corrupted, otherwise the tail. This
    biases training toward corrupting the rarer side of the relation
    (1-to-N relations bias toward head, N-to-1 toward tail), which
    yields harder negatives than uniform head-or-tail sampling.

    Construct via :meth:`from_data` with ``bern_probs`` from
    :meth:`compute_probs`. The default :meth:`corrupt` call
    (``mode=None`` or ``mode="bernoulli"``) does the coin-flip.
    Explicit ``mode="head"`` / ``"tail"`` / ``"both"`` defers to the
    base :class:`Sampler` behaviour — useful for eval, where the same
    sampler instance must do plain head- or tail-only corruption while
    training uses the bernoulli mix.
    """

    def __init__(self, cfg: SamplerConfig, bern_probs: _Tensor) -> None:
        super().__init__(cfg)
        self._bern_probs = bern_probs.to(device=cfg.device)

    @staticmethod
    def compute_probs(triples_rht: LongTensor, num_relations: int) -> _Tensor:
        """Per-relation head-corruption probabilities for Bernoulli sampling.

        Computed once from a known-triple set (typically the train split)
        and passed to :meth:`from_data` as ``bern_probs``.

        Args:
            triples_rht: ``[N, 3]`` triples in ``(r, h, t)`` format.
            num_relations: Total number of relations.

        Returns:
            ``[num_relations]`` float tensor of head-corruption
            probabilities, clamped to ``[0.05, 0.95]``.
        """
        t = triples_rht.cpu().long()
        rels = t[:, 0]
        heads = t[:, 1]
        tails = t[:, 2]
        N = t.shape[0]

        ones = torch.ones(N, dtype=torch.float32)
        triple_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(0, rels, ones)

        max_ent = max(heads.max().item(), tails.max().item()) + 1

        rh_rels = torch.unique(rels * max_ent + heads) // max_ent
        unique_head_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(
            0, rh_rels, torch.ones(rh_rels.shape[0], dtype=torch.float32)
        )

        rt_rels = torch.unique(rels * max_ent + tails) // max_ent
        unique_tail_counts = torch.zeros(num_relations, dtype=torch.float32).scatter_add_(
            0, rt_rels, torch.ones(rt_rels.shape[0], dtype=torch.float32)
        )

        # tph = tails per head, hpt = heads per tail
        tph = triple_counts / unique_head_counts.clamp(min=1)
        hpt = triple_counts / unique_tail_counts.clamp(min=1)

        denom = tph + hpt
        probs = torch.where(denom > 0, tph / denom, torch.full_like(denom, 0.5))
        return probs.clamp(0.05, 0.95)

    @classmethod
    def from_data(
        cls,
        all_known_triples_idx: Optional[LongTensor],
        num_entities: int,
        num_relations: int,
        device: torch.device,
        bern_probs: _Tensor,
        seed: int = 0,
        domain2idx: Optional[Dict[str, List[int]]] = None,
        entity2domain: Optional[Dict[int, str]] = None,
        order_negatives: bool = False,
        min_entity_idx: int = 1,
    ) -> "BernoulliSampler":
        """Construct a Bernoulli sampler. See :meth:`Sampler.from_data` for
        the shared args; ``bern_probs`` is the per-relation head-corruption
        probability vector (typically from :func:`compute_bernoulli_probs`).
        """
        # The parent's ``default_mode`` is hit only when callers pass an
        # explicit non-bernoulli mode (e.g. eval); store ``"both"`` so
        # ``mode=None`` on those callers means head+tail.
        cfg = SamplerConfig(
            num_entities, num_relations, device, "both", seed,
            order_negatives, min_entity_idx,
        )
        self = cls(cfg, bern_probs)
        if all_known_triples_idx is not None and all_known_triples_idx.numel() > 0:
            hashes = _mix_hash(
                all_known_triples_idx.detach().to(device=self.device, dtype=torch.long),
                self.b_e, self.b_r,
            )
            self._filter_hashes_sorted = torch.sort(torch.unique(hashes)).values
        else:
            self._filter_hashes_sorted = torch.empty((0,), dtype=torch.long, device=self.device)
        if domain2idx and entity2domain:
            self._build_domain_structures(domain2idx, entity2domain, device)
        return self

    def corrupt(
        self,
        positives: LongTensor,
        *,
        num_negatives: Optional[int] = None,
        mode: Optional[str] = None,
        device: Optional[torch.device] = None,
        filter: bool = True,
        unique: bool = True,
        return_mask: bool = False,
    ) -> Union[LongTensor, Tuple[LongTensor, torch.BoolTensor]]:
        """Corrupt with a per-triple head/tail coin flip (bernoulli) by default.

        Pass an explicit ``mode="head"``, ``"tail"``, or ``"both"`` to fall
        back to the base sampler — useful when the same sampler instance
        is shared between training (bernoulli) and eval (plain head/tail).
        """
        if mode not in (None, "bernoulli"):
            return super().corrupt(
                positives, num_negatives=num_negatives, mode=mode,
                device=device, filter=filter, unique=unique, return_mask=return_mask,
            )
        head_neg, head_valid = super().corrupt(
            positives, num_negatives=num_negatives, mode="head",
            device=device, filter=filter, unique=unique, return_mask=True,
        )
        tail_neg, tail_valid = super().corrupt(
            positives, num_negatives=num_negatives, mode="tail",
            device=device, filter=filter, unique=unique, return_mask=True,
        )
        rels = positives[:, 0]
        bern = self._bern_probs.to(device=rels.device)
        corrupt_head = torch.bernoulli(bern[rels]).bool()
        ch = corrupt_head[:, None, None]
        neg = torch.where(ch, head_neg, tail_neg)
        valid = torch.where(corrupt_head[:, None], head_valid, tail_valid)
        return (neg, valid) if return_mask else neg


__all__ = [
    "BernoulliSampler",
    "Sampler",
    "SamplerConfig",
]
