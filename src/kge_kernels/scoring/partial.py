"""Partial-atom scoring built on top of ``model.score``.

Two scorer flavours share one interface:

- :class:`PartialScorer` — eager precompute. ``compute_all()`` fills
  ``[P, E]`` lookup tables for every (pred, entity) pair upfront; cheap
  per-atom scoring afterwards via :meth:`score_atoms`.
- :class:`LazyPartialScorer` — incremental fill. Tables start empty;
  :meth:`ensure_for_derived_states` (or :meth:`ensure`) computes only
  the (pred, entity) entries the caller actually needs. Same
  :meth:`score_atoms` lookup. Use this when ``P × E`` is too large to
  precompute or when only a fraction of pairs are touched per batch.

Both keep ``max_tail_score`` and ``max_head_score`` ``[P_im, E_im]``
tensors as the lookup tables. ``score_atoms`` reads them; the lazy
variant updates them on demand.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Pure-tensor lookup helper (used by PartialScorer.score_atoms)
# ============================================================================


def _lookup_partial_atom_scores(
    preds: Tensor,
    args1: Tensor,
    args2: Tensor,
    constant_no: int,
    max_tail_score: Tensor,
    max_head_score: Tensor,
) -> Tensor:
    """Score partially grounded binary atoms by table lookup.

    The function assumes the shared grounding convention:

    - ``pred(const, ?)`` uses ``max_tail_score``
    - ``pred(?, const)`` uses ``max_head_score``
    - fully unbound atoms receive score ``0``
    """
    n = preds.shape[0]
    device = preds.device
    scores = torch.zeros(n, dtype=torch.float32, device=device)
    if n == 0:
        return scores

    a1 = args1.long()
    a2 = args2.long()
    p = preds.long()
    p_max, e_max = max_tail_score.shape

    safe_p = p.clamp(0, p_max - 1)
    safe_a1 = a1.clamp(0, e_max - 1)
    safe_a2 = a2.clamp(0, e_max - 1)

    tail_var = (a1 > 0) & (a1 <= constant_no) & (a2 > constant_no)
    scores = torch.where(tail_var, max_tail_score[safe_p, safe_a1], scores)

    head_var = (a1 > constant_no) & (a2 > 0) & (a2 <= constant_no)
    scores = torch.where(head_var, max_head_score[safe_p, safe_a2], scores)

    return scores


# ============================================================================
# Base scorer (eager precompute)
# ============================================================================


class PartialScorer:
    """Partial-atom scorer with eager precompute.

    Holds two ``[P_im, E_im]`` lookup tables (``max_tail_score`` and
    ``max_head_score``). Call :meth:`compute_all` once after
    construction to fill them, then use :meth:`score_atoms` for
    per-batch scoring.

    For datasets where ``P × E`` is too large to precompute upfront,
    use :class:`LazyPartialScorer` instead — same scoring API,
    incremental fill.

    Args:
        model: tkk-native KGE model exposing ``score(h, r, t)``.
        pred_remap: ``[P_im]`` predicate remap. Caller's predicate id
            ``i`` maps to KGE relation ``pred_remap[i]``; ``-1`` means
            unmapped (no precompute, contributes neutral 0.5 if asked).
        const_remap: ``[E_im]`` entity remap, same convention.
        constant_no: Highest constant id in the caller's index space.
            Atoms with ``args1 > constant_no`` (or ``args2 > constant_no``)
            are treated as unbound variables.
        sigmoid: if True (default), apply ``torch.sigmoid`` to
            ``model.score``'s raw output before taking the per-relation max.
    """

    def __init__(
        self,
        model: nn.Module,
        pred_remap: Tensor,
        const_remap: Tensor,
        constant_no: int,
        *,
        sigmoid: bool = True,
    ) -> None:
        device = const_remap.device
        P_im = pred_remap.shape[0]
        E_im = const_remap.shape[0]

        self.model = model
        self.pred_remap = pred_remap
        self.const_remap = const_remap
        self.constant_no = constant_no
        self.sigmoid = sigmoid

        self.max_tail_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)
        self.max_head_score = torch.zeros(P_im, E_im, dtype=torch.float32, device=device)

        # Auto-tune chunk so each ``model.score`` call peaks at ~2GB.
        E_kge = (const_remap >= 0).sum().item()
        self._batch_chunk = max(1, min(512, int(2e9 / (max(E_kge, 1) * 4 * 1024))))

    @classmethod
    def from_tables(
        cls,
        max_tail_score: Tensor,
        max_head_score: Tensor,
        constant_no: int,
    ) -> "PartialScorer":
        """Construct a scorer from pre-built lookup tables.

        Use this when the tables come from somewhere other than
        :meth:`compute_all` (e.g. cached on disk, or the caller built
        them with their own logic). The resulting scorer can only be
        used for :meth:`score_atoms` lookups — no recompute path.
        """
        scorer = cls.__new__(cls)
        scorer.model = None
        scorer.pred_remap = None
        scorer.const_remap = None
        scorer.constant_no = constant_no
        scorer.sigmoid = True
        scorer.max_tail_score = max_tail_score
        scorer.max_head_score = max_head_score
        scorer._batch_chunk = 64
        return scorer

    @torch.no_grad()
    def compute_all(self, batch_chunk: Optional[int] = None) -> "PartialScorer":
        """Eagerly fill both lookup tables for every (pred, entity) pair.

        For each mapped predicate/entity pair:
          - ``max_tail_score[p, e]`` = best score over all tails for ``p(e, ?)``.
          - ``max_head_score[p, e]`` = best score over all heads for ``p(?, e)``.

        Args:
            batch_chunk: Number of entities scored per chunk. ``None``
                uses the auto-tuned default (~2GB peak per call).

        Returns:
            ``self`` so the call chains: ``PartialScorer(...).compute_all()``.
        """
        bc = batch_chunk if batch_chunk and batch_chunk > 0 else self._batch_chunk

        valid_preds = (self.pred_remap >= 0).nonzero(as_tuple=True)[0]
        valid_ents = (self.const_remap >= 0).nonzero(as_tuple=True)[0]
        if valid_preds.numel() == 0 or valid_ents.numel() == 0:
            return self

        kge_ents = self.const_remap[valid_ents]
        for im_pred in valid_preds:
            kge_rel = self.pred_remap[im_pred]
            tail_scores = self._chunked_max(kge_ents, kge_rel, role="tail", batch_chunk=bc)
            head_scores = self._chunked_max(kge_ents, kge_rel, role="head", batch_chunk=bc)
            self.max_tail_score[im_pred, valid_ents] = tail_scores
            self.max_head_score[im_pred, valid_ents] = head_scores

        return self

    def _chunked_max(
        self,
        kge_ents: Tensor,
        kge_rel: Tensor,
        role: str,
        batch_chunk: int,
    ) -> Tensor:
        """Best head/tail completion per entity, chunked over the entity axis."""
        num_entities = kge_ents.shape[0]
        device = kge_ents.device
        result = torch.empty(num_entities, dtype=torch.float32, device=device)

        for start in range(0, num_entities, batch_chunk):
            end = min(start + batch_chunk, num_entities)
            chunk = kge_ents[start:end]
            rel_exp = kge_rel.expand(chunk.shape[0])
            if role == "tail":
                raw = self.model.score(chunk, rel_exp, None)
            else:
                raw = self.model.score(None, rel_exp, chunk)
            if self.sigmoid:
                raw = torch.sigmoid(raw)
            result[start:end] = raw.max(dim=1).values

        return result

    def score_atoms(
        self,
        preds: Tensor,
        args1: Tensor,
        args2: Tensor,
    ) -> Tensor:
        """Score partially grounded atoms via lookup. Returns ``[N]``."""
        return _lookup_partial_atom_scores(
            preds, args1, args2, self.constant_no,
            self.max_tail_score, self.max_head_score,
        )


# ============================================================================
# Lazy variant (incremental fill)
# ============================================================================


class LazyPartialScorer(PartialScorer):
    """Lazy partial-atom scorer with on-demand fill.

    Tables start at zero. Call :meth:`ensure_for_derived_states` (or
    :meth:`ensure`) before scoring to fill the entries for the
    (pred, entity) pairs you're about to look up. Subsequent
    :meth:`score_atoms` calls read from the partially-filled tables
    just like the eager :class:`PartialScorer`.

    Use this when the full ``[P × E]`` precompute is too expensive or
    only a small fraction of pairs are visited per batch (proof-search
    workloads).
    """

    def __init__(
        self,
        model: nn.Module,
        pred_remap: Tensor,
        const_remap: Tensor,
        constant_no: int,
        padding_idx: int,
        true_pred_idx: int = -1,
        false_pred_idx: int = -1,
        *,
        sigmoid: bool = True,
    ) -> None:
        super().__init__(
            model, pred_remap, const_remap, constant_no, sigmoid=sigmoid,
        )
        self.padding_idx = padding_idx
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx

        device = const_remap.device
        P_im = pred_remap.shape[0]
        E_im = const_remap.shape[0]
        self._cached_tail = torch.zeros(P_im, E_im, dtype=torch.bool, device=device)
        self._cached_head = torch.zeros(P_im, E_im, dtype=torch.bool, device=device)
        self._computed: set = set()
        self._n_computed = 0

    @torch.no_grad()
    def ensure(self, preds: Tensor) -> None:
        """Pre-compute ALL entities for given predicates (eager, per-predicate)."""
        unique = preds.unique().tolist()
        new = [int(p) for p in unique if p > 0 and int(p) not in self._computed]
        if not new:
            return
        valid_mask = self.const_remap >= 0
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_kge = self.const_remap[valid_indices]
        E = valid_kge.shape[0]
        if E == 0:
            return
        for im_pred in new:
            kge_rel = self.pred_remap[im_pred]
            if kge_rel < 0:
                self.max_tail_score[im_pred, valid_indices] = 0.5
                self.max_head_score[im_pred, valid_indices] = 0.5
                self._cached_tail[im_pred, valid_indices] = True
                self._cached_head[im_pred, valid_indices] = True
                self._computed.add(im_pred)
                continue
            for start in range(0, E, self._batch_chunk):
                end = min(start + self._batch_chunk, E)
                ents = valid_kge[start:end]
                idx = valid_indices[start:end]
                rel = kge_rel.expand(ents.shape[0])
                tail = self.model.score(ents, rel, None)
                head = self.model.score(None, rel, ents)
                if self.sigmoid:
                    tail = torch.sigmoid(tail)
                    head = torch.sigmoid(head)
                self.max_tail_score[im_pred, idx] = tail.max(dim=1).values
                self.max_head_score[im_pred, idx] = head.max(dim=1).values
            self._cached_tail[im_pred, valid_indices] = True
            self._cached_head[im_pred, valid_indices] = True
            self._computed.add(im_pred)
            self._n_computed += E

    @torch.no_grad()
    def ensure_for_derived_states(
        self,
        derived_states: Tensor,
        derived_counts: Tensor,
    ) -> None:
        """Compute and cache partial scores for atoms in derived states.

        Only processes valid (non-padding) atoms. Single GPU→CPU sync.
        """
        B, G, A, _ = derived_states.shape
        C = self.constant_no
        pad = self.padding_idx

        valid_s = torch.arange(G, device=derived_states.device).unsqueeze(0) < derived_counts.unsqueeze(1)
        valid_atoms = valid_s.unsqueeze(2).expand(B, G, A).reshape(-1)
        flat = derived_states.reshape(-1, 3)[valid_atoms]

        if flat.shape[0] == 0:
            return

        preds = flat[:, 0]
        args1 = flat[:, 1]
        args2 = flat[:, 2]

        a1_const = (args1 > 0) & (args1 <= C)
        a2_const = (args2 > 0) & (args2 <= C)
        is_partial_tail = a1_const & ~a2_const & (preds != pad) & (preds != self.true_pred_idx)
        is_partial_head = ~a1_const & a2_const & (preds != pad) & (preds != self.true_pred_idx)

        tp = preds[is_partial_tail]
        te = args1[is_partial_tail]
        hp = preds[is_partial_head]
        he = args2[is_partial_head]

        tail_uncached = ~self._cached_tail[tp, te] if tp.numel() > 0 else tp
        head_uncached = ~self._cached_head[hp, he] if hp.numel() > 0 else hp
        n = (tail_uncached.sum() + head_uncached.sum()).item()
        if n == 0:
            return

        if tail_uncached.numel() > 0 and tail_uncached.any():
            self._compute_entries(tp[tail_uncached], te[tail_uncached], role="tail")
        if head_uncached.numel() > 0 and head_uncached.any():
            self._compute_entries(hp[head_uncached], he[head_uncached], role="head")

    def _compute_entries(self, preds: Tensor, entities: Tensor, role: str) -> None:
        """Batch-compute missing partial scores, grouped by predicate."""
        keys = preds.long() * (self.const_remap.shape[0] + 1) + entities.long()
        unique_keys = keys.unique()
        unique_preds = unique_keys // (self.const_remap.shape[0] + 1)
        unique_ents = unique_keys % (self.const_remap.shape[0] + 1)

        for p_im in unique_preds.unique().tolist():
            p_im = int(p_im)
            kge_rel = self.pred_remap[p_im]
            mask = unique_preds == p_im
            im_ents = unique_ents[mask]

            if kge_rel < 0:
                table = self.max_tail_score if role == "tail" else self.max_head_score
                cache = self._cached_tail if role == "tail" else self._cached_head
                table[p_im, im_ents] = 0.5
                cache[p_im, im_ents] = True
                continue

            kge_ents = self.const_remap[im_ents]
            valid = kge_ents >= 0
            if not valid.any():
                continue
            kge_ents_valid = kge_ents[valid]
            im_ents_valid = im_ents[valid]

            table = self.max_tail_score if role == "tail" else self.max_head_score
            cache = self._cached_tail if role == "tail" else self._cached_head
            bc = self._batch_chunk

            for start in range(0, kge_ents_valid.shape[0], bc):
                end = min(start + bc, kge_ents_valid.shape[0])
                chunk_kge = kge_ents_valid[start:end]
                chunk_im = im_ents_valid[start:end]
                rel = kge_rel.expand(chunk_kge.shape[0])
                if role == "tail":
                    s = self.model.score(chunk_kge, rel, None)
                else:
                    s = self.model.score(None, rel, chunk_kge)
                if self.sigmoid:
                    s = torch.sigmoid(s)
                table[p_im, chunk_im] = s.max(dim=1).values
                cache[p_im, chunk_im] = True

            self._n_computed += kge_ents_valid.shape[0]

    @property
    def n_computed(self) -> int:
        return self._n_computed


__all__ = ["LazyPartialScorer", "PartialScorer"]
