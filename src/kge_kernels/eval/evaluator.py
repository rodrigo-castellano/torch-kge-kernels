"""Unified KGE evaluator.

One class for all evaluation modes:

  - **Exhaustive** (``num_corruptions == 0``): rank against ALL entities,
    optionally filter known facts. Standard KGE paper protocol (TransE, 2013).
  - **Sampled** (``num_corruptions > 0``): rank against K randomly sampled
    corruptions. Faster, used by DpRL PPO real-time evaluation.

Both modes support optional per-relation domain constraints and optional
multi-scorer fusion (RRF, z-score).

The class is model-first: it takes an ``nn.Module`` with a ``score(h, r, t)``
dispatch method (as provided by ``kge_kernels.models.KGEModel``). For sampled
mode, ``_score_pool`` can be overridden in subclasses for compiled/optimized
scoring (e.g. DpRL's ``PPOEvaluator`` with CUDA-graph replay).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

import torch
from torch import Tensor, nn

from ..scoring import SupportsCorruptWithMask
from .pool import CandidatePool
from .ranking import compute_ranks, ranking_metrics

logger = logging.getLogger(__name__)

ScorerFn = Callable[[Tensor], Dict[str, Tensor]]
FusionFn = Callable[[Dict[str, Tensor], int, int, torch.device], Dict[str, Tensor]]


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel and torch.compile wrappers.

    Inlined to avoid eagerly importing ``training/`` at package load time.
    """
    inner: nn.Module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(inner, "_orig_mod"):
        inner = inner._orig_mod  # type: ignore[assignment]
    return inner


def _apply_masks(
    scores: Tensor,
    idx1: Tensor,
    idx2: Tensor,
    filter_map: Optional[Dict[Tuple[int, int], Set[int]]],
    true_entities: Optional[Tensor],
    domain: Optional[Set[int]],
) -> None:
    """Apply domain and known-fact filter masks to a score matrix in-place."""
    batch_size, num_entities = scores.shape
    device = scores.device

    if domain is not None:
        domain_mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
        domain_mask[torch.tensor(sorted(domain), dtype=torch.long, device=device)] = True
        scores[:, ~domain_mask] = float("-inf")

    if filter_map is not None and true_entities is not None:
        idx1_list = idx1.tolist() if idx1.dim() > 0 else [int(idx1.item())] * batch_size
        idx2_list = idx2.tolist() if idx2.dim() > 0 else [int(idx2.item())] * batch_size
        for row in range(batch_size):
            key = (int(idx1_list[row]), int(idx2_list[row]))
            known = filter_map.get(key)
            if known:
                true_ent = int(true_entities[row].item())
                filtered = [ent for ent in known if ent != true_ent]
                if filtered:
                    scores[row, torch.tensor(filtered, dtype=torch.long, device=device)] = float("-inf")


class Evaluator:
    """Unified KGE evaluator — exhaustive or sampled, filtered or unfiltered.

    Args:
        model: KGE model with ``score(h, r, t=None)`` dispatch (as in
            ``kge_kernels.models.KGEModel``). ``DataParallel`` and
            ``torch.compile`` wrappers are handled automatically.
        num_entities: Size of the entity vocabulary.
        num_corruptions: ``0`` = exhaustive ranking against all entities
            (standard paper protocol). ``>0`` = sampled ranking against
            that many corruptions (faster, used by PPO).
        head_filter: ``{(r, t): {known_heads}}`` — entities to mask when
            ranking heads (filtered protocol).
        tail_filter: ``{(h, r): {known_tails}}`` — analogous for tails.
        head_domain: ``{r: {valid_head_entities}}`` — restrict head
            candidates per relation.
        tail_domain: Analogous for tails.
        sampler: Corruption sampler (required if ``num_corruptions > 0``).
        corruption_scheme: ``"head"``, ``"tail"``, or ``"both"``.
        batch_size: Fixed sub-batch size for pool scoring. ``0`` = auto.
        fusion: Optional fusion callable (``rrf``, ``zscore_fusion``).
        seed: RNG seed for tie-breaking and sampled corruption.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        num_entities: int,
        *,
        num_corruptions: int = 0,
        head_filter: Optional[Dict[Tuple[int, int], Set[int]]] = None,
        tail_filter: Optional[Dict[Tuple[int, int], Set[int]]] = None,
        head_domain: Optional[Dict[int, Set[int]]] = None,
        tail_domain: Optional[Dict[int, Set[int]]] = None,
        sampler: Optional[SupportsCorruptWithMask] = None,
        corruption_scheme: Literal["head", "tail", "both"] = "both",
        batch_size: int = 0,
        fusion: Optional[FusionFn] = None,
        seed: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        if num_corruptions > 0 and sampler is None:
            raise ValueError("Sampled evaluation (num_corruptions > 0) requires a sampler")
        self.model = model
        self.num_entities = num_entities
        self.num_corruptions = num_corruptions
        self.head_filter = head_filter
        self.tail_filter = tail_filter
        self.head_domain = head_domain
        self.tail_domain = tail_domain
        self.sampler = sampler
        self.corruption_scheme = corruption_scheme
        self.batch_size = batch_size
        self.fusion = fusion
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pool_buffer: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        test_triples: Tensor,
        *,
        n_queries: Optional[int] = None,
        show_progress: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the model on test triples.

        Dispatches to exhaustive or sampled evaluation based on
        ``num_corruptions``.

        Args:
            test_triples: ``[N, 3]`` test triples in ``(r, h, t)`` format.
            n_queries: Limit evaluation to first *n* queries.
            show_progress: Print rolling MRR during exhaustive evaluation.

        Returns:
            ``{"MRR": ..., "Hits@1": ..., "Hits@3": ..., "Hits@10": ...}``
        """
        if n_queries is not None:
            test_triples = test_triples[:n_queries]

        if self.num_corruptions == 0:
            return self._evaluate_exhaustive(test_triples, show_progress=show_progress)
        else:
            return self._evaluate_sampled(test_triples)

    # ------------------------------------------------------------------
    # Exhaustive evaluation (standard filtered ranking protocol)
    # ------------------------------------------------------------------

    def _evaluate_exhaustive(
        self,
        test_triples: Tensor,
        show_progress: bool = False,
    ) -> Dict[str, float]:
        """Rank against ALL entities, filter known facts."""
        triples_list: List[Tuple[int, int, int]] = [
            (int(r), int(h), int(t))
            for r, h, t in test_triples.tolist()
        ]
        if not triples_list:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

        training_mode = self.model.training
        self.model.eval()
        actual_model = _unwrap_model(self.model)

        # Chunk size heuristic
        dim = getattr(actual_model, "dim", 512)
        max_chunk = max(1, min(512, (4 * 1024**3) // max(1, self.num_entities * dim * 4)))

        by_relation: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        for r, h, t in triples_list:
            by_relation[r].append((r, h, t))

        all_ranks_list: List[Tensor] = []
        processed = 0
        report_every = max(1, len(triples_list) // 20)
        last_report = 0

        for r, rel_triples in by_relation.items():
            all_heads = torch.tensor([h for _, h, _ in rel_triples], dtype=torch.long, device=self.device)
            all_tails = torch.tensor([t for _, _, t in rel_triples], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor(r, dtype=torch.long, device=self.device)
            t_domain = self.tail_domain.get(r) if self.tail_domain else None
            h_domain = self.head_domain.get(r) if self.head_domain else None

            for chunk_start in range(0, len(rel_triples), max_chunk):
                chunk_end = min(chunk_start + max_chunk, len(rel_triples))
                heads = all_heads[chunk_start:chunk_end]
                tails = all_tails[chunk_start:chunk_end]

                chunk_ranks: List[Tensor] = []

                if self.corruption_scheme in ("tail", "both"):
                    tail_scores = self._score_all_tails(actual_model, heads, r_tensor)
                    _apply_masks(tail_scores, heads, r_tensor, self.tail_filter, tails, t_domain)
                    chunk_ranks.append(compute_ranks(tail_scores, tails))

                if self.corruption_scheme in ("head", "both"):
                    head_scores = self._score_all_heads(actual_model, r_tensor, tails)
                    _apply_masks(head_scores, r_tensor, tails, self.head_filter, heads, h_domain)
                    chunk_ranks.append(compute_ranks(head_scores, heads))

                all_ranks_list.append(torch.cat(chunk_ranks))

            processed += len(rel_triples)
            if show_progress and (processed >= len(triples_list) or processed - last_report >= report_every):
                running_ranks = torch.cat(all_ranks_list)
                running_mrr = (1.0 / running_ranks.double()).mean().item()
                print(f"Evaluation {processed}/{len(triples_list)} triples | rolling_mrr={running_mrr:.4f}")
                last_report = processed

        if training_mode:
            self.model.train()

        if not all_ranks_list:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        return ranking_metrics(torch.cat(all_ranks_list))

    def _score_all_tails(self, model: nn.Module, h: Tensor, r: Tensor) -> Tensor:
        """Score all entities as tails for (h, r) pairs."""
        if r.dim() == 0:
            r = r.expand(h.shape[0])
        if hasattr(model, "score"):
            return torch.sigmoid(model.score(h, r, None))
        if hasattr(model, "score_all_tails"):
            return torch.sigmoid(model.score_all_tails(h, r))
        # Fallback: expand to [B, E] via score_triples
        B = h.shape[0]
        E = self.num_entities
        all_t = torch.arange(E, device=h.device).unsqueeze(0).expand(B, -1)
        h_exp = h.unsqueeze(1).expand(B, E).reshape(-1)
        r_exp = r.expand(B * E) if r.dim() == 0 else r.unsqueeze(1).expand(B, E).reshape(-1)
        t_exp = all_t.reshape(-1)
        raw = model.score_triples(h_exp, r_exp, t_exp) if hasattr(model, "score_triples") else model.score_atoms(r_exp, h_exp, t_exp)
        return torch.sigmoid(raw).view(B, E)

    def _score_all_heads(self, model: nn.Module, r: Tensor, t: Tensor) -> Tensor:
        """Score all entities as heads for (r, t) pairs."""
        if r.dim() == 0:
            r = r.expand(t.shape[0])
        if hasattr(model, "score"):
            return torch.sigmoid(model.score(None, r, t))
        if hasattr(model, "score_all_heads"):
            return torch.sigmoid(model.score_all_heads(r, t))
        B = t.shape[0]
        E = self.num_entities
        all_h = torch.arange(E, device=t.device).unsqueeze(0).expand(B, -1)
        h_exp = all_h.reshape(-1)
        r_exp = r.expand(B * E) if r.dim() == 0 else r.unsqueeze(1).expand(B, E).reshape(-1)
        t_exp = t.unsqueeze(1).expand(B, E).reshape(-1)
        raw = model.score_triples(h_exp, r_exp, t_exp) if hasattr(model, "score_triples") else model.score_atoms(r_exp, h_exp, t_exp)
        return torch.sigmoid(raw).view(B, E)

    # ------------------------------------------------------------------
    # Sampled evaluation (corruption-pool based)
    # ------------------------------------------------------------------

    def _evaluate_sampled(self, test_triples: Tensor) -> Dict[str, float]:
        """Rank against K sampled corruptions."""
        assert self.sampler is not None
        N = test_triples.shape[0]
        corruption_modes: Sequence[str] = (
            ("head", "tail") if self.corruption_scheme == "both" else (self.corruption_scheme,)
        )
        nm = len(corruption_modes)
        K = 1 + self.num_corruptions

        eff_batch = self.batch_size if self.batch_size > 0 else K * 64
        chunk_queries = max(1, eff_batch // (K * nm))
        chunk_queries = min(chunk_queries, N)

        max_pool = chunk_queries * K * nm
        pool_buffer = self._ensure_pool_buffer(max_pool)

        tie_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        all_ranks: Dict[str, List[Tensor]] = {}

        n_chunks = (N + chunk_queries - 1) // chunk_queries
        for chunk_i, start in enumerate(range(0, N, chunk_queries)):
            end = min(start + chunk_queries, N)
            chunk_q = test_triples[start:end].to(self.device)
            if n_chunks > 5 and (chunk_i % max(1, n_chunks // 10) == 0 or chunk_i == n_chunks - 1):
                logger.info("Evaluating chunk %d/%d (queries %d-%d/%d)", chunk_i + 1, n_chunks, start + 1, end, N)

            combined, pools = CandidatePool.build_batched(
                chunk_q, self.sampler, self.num_corruptions,
                modes=corruption_modes, device=self.device, buffer=pool_buffer,
            )

            scores_dict = self._score_pool(combined)

            offset = 0
            for mode, pool_obj in zip(corruption_modes, pools):
                P = pool_obj.pool_size
                mode_scores_dict = {
                    mode_name: scores_flat[offset:offset + P]
                    for mode_name, scores_flat in scores_dict.items()
                }

                if self.fusion is not None and len(mode_scores_dict) > 1:
                    fused = self.fusion(mode_scores_dict, pool_obj.K, pool_obj.CQ, self.device)
                    mode_scores_dict.update(fused)

                for mode_name, mode_scores in mode_scores_dict.items():
                    scores_2d = mode_scores.view(pool_obj.K, pool_obj.CQ).t()
                    # True item is always at column 0 (CandidatePool convention)
                    true_idx = torch.zeros(pool_obj.CQ, dtype=torch.long, device=self.device)
                    valid = torch.ones_like(scores_2d, dtype=torch.bool)
                    valid[:, 1:] = pool_obj.valid_mask[:, 1:]

                    key = f"{mode_name}_{mode}" if nm > 1 else mode_name
                    ranks = compute_ranks(scores_2d, true_idx, valid_mask=valid,
                                          tie_handling="random", generator=tie_gen)
                    all_ranks.setdefault(key, []).append(ranks)
                offset += P

        # Aggregate across modes
        if nm > 1:
            mode_names = set()
            for key in all_ranks:
                mode_names.add(key.rsplit("_", 1)[0])
            combined_metrics: Dict[str, float] = {}
            for mn in sorted(mode_names):
                combined_ranks_list = []
                for cm in corruption_modes:
                    key = f"{mn}_{cm}"
                    if key in all_ranks:
                        combined_ranks_list.append(torch.cat(all_ranks[key]))
                if combined_ranks_list:
                    m = ranking_metrics(torch.cat(combined_ranks_list))
                    if not combined_metrics:
                        combined_metrics = m
                    else:
                        for k, v in m.items():
                            combined_metrics[k] = v
            return combined_metrics
        else:
            for key, rank_list in all_ranks.items():
                return ranking_metrics(torch.cat(rank_list))
        return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}

    def _ensure_pool_buffer(self, max_pool: int) -> Tensor:
        """Allocate or reuse the pre-allocated pool buffer."""
        if self._pool_buffer is None or self._pool_buffer.shape[0] < max_pool:
            self._pool_buffer = torch.zeros(max_pool, 3, dtype=torch.long, device=self.device)
        return self._pool_buffer

    def _score_pool(self, pool: Tensor) -> Dict[str, Tensor]:
        """Score candidate pool. Override in subclass for compiled scoring.

        Default implementation calls ``self.model.score_triples(h, r, t)``
        with sub-batching and optional padding.
        """
        P = pool.shape[0]
        B = self.batch_size if self.batch_size > 0 else P
        actual_model = _unwrap_model(self.model)
        all_scores: List[Tensor] = []

        for sb_start in range(0, P, B):
            sb_end = min(sb_start + B, P)
            batch = pool[sb_start:sb_end]
            actual_B = batch.shape[0]

            if self.batch_size > 0 and actual_B < B:
                padded = torch.zeros(B, 3, dtype=batch.dtype, device=batch.device)
                padded[:actual_B] = batch
                batch = padded

            r, h, t = batch[:, 0], batch[:, 1], batch[:, 2]
            # Clamp entity indices to valid range — padding/invalid pool
            # entries may contain OOB indices; they are masked out later.
            h = h.clamp(0, self.num_entities - 1)
            t = t.clamp(0, self.num_entities - 1)
            if hasattr(actual_model, "score"):
                raw = actual_model.score(h, r, t)
            elif hasattr(actual_model, "score_triples"):
                raw = actual_model.score_triples(h, r, t)
            elif hasattr(actual_model, "score_atoms"):
                raw = actual_model.score_atoms(r, h, t)
            else:
                raise AttributeError("Model requires score(), score_triples(), or score_atoms()")
            scores = torch.sigmoid(raw)

            if self.batch_size > 0 and actual_B < B:
                scores = scores[:actual_B]
            all_scores.append(scores)

        return {"default": torch.cat(all_scores)}


__all__ = ["Evaluator", "ScorerFn", "FusionFn"]
