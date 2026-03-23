"""Generic corruption-based evaluator.

Takes a single ``scorer: [B, 3] → Dict[str, Tensor]`` callable and handles
pool construction, sub-batching, padding, ranking, fusion, and metrics.
All pool construction uses pre-allocated buffers for reduce-overhead compatibility.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

import torch
from torch import Tensor

from ..ranking import ranking_metrics, ranks_from_scores
from ..types import SupportsCorruptWithMask
from .pool import CandidatePool
from .results import EvalResults

ScorerFn = Callable[[Tensor], Dict[str, Tensor]]
FusionFn = Callable[[Dict[str, Tensor], int, int, torch.device], Dict[str, Tensor]]


class Evaluator:
    """Corruption-based KGR evaluator.

    The evaluator is scorer-agnostic: it accepts any callable that maps a
    batch of triples ``[B, 3]`` to a dict of score tensors ``{mode: [B]}``.

    All modes (head/tail) are batched into a single ``_score_pool`` call
    for maximum throughput. Pool buffers are pre-allocated for
    ``torch.compile(mode='reduce-overhead')`` compatibility.

    Args:
        scorer: Callable ``[B, 3] → Dict[str, Tensor[B]]``.
        sampler: Sampler with ``corrupt_with_mask`` for corruption generation.
        n_corruptions: Negatives per positive (default 100).
        corruption_scheme: ``"head"``, ``"tail"``, or ``"both"``.
        batch_size: If > 0, pad sub-batches to this fixed size. 0 = no padding.
        fusion: Optional fusion callable (e.g. ``rrf``, ``zscore_fusion``).
        seed: RNG seed for tie-breaking.
        device: Computation device.
    """

    def __init__(
        self,
        scorer: ScorerFn,
        sampler: SupportsCorruptWithMask,
        n_corruptions: int = 100,
        corruption_scheme: Literal["head", "tail", "both"] = "both",
        batch_size: int = 0,
        fusion: Optional[FusionFn] = None,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.scorer = scorer
        self.sampler = sampler
        self.n_corruptions = n_corruptions
        self.corruption_scheme = corruption_scheme
        self.batch_size = batch_size
        self.fusion = fusion
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._pool_buffer: Optional[Tensor] = None

    def _ensure_pool_buffer(self, max_pool: int) -> Tensor:
        """Allocate or reuse the pre-allocated pool buffer."""
        if self._pool_buffer is None or self._pool_buffer.shape[0] < max_pool:
            self._pool_buffer = torch.zeros(max_pool, 3, dtype=torch.long, device=self.device)
        return self._pool_buffer

    @torch.no_grad()
    def evaluate(
        self,
        test_queries: Tensor,
        n_queries: Optional[int] = None,
    ) -> EvalResults:
        """Run full corruption-based evaluation.

        All corruption modes are batched into one ``_score_pool`` call per
        chunk for maximum throughput and CUDA graph compatibility.

        Args:
            test_queries: ``[N, 3]`` test triples in ``(r, h, t)`` format.
            n_queries: Limit evaluation to first *n* queries.

        Returns:
            ``EvalResults`` with per-mode metrics and aggregate stats.
        """
        if n_queries is not None:
            test_queries = test_queries[:n_queries]
        N = test_queries.shape[0]
        corruption_modes: Sequence[str] = (
            ("head", "tail") if self.corruption_scheme == "both" else (self.corruption_scheme,)
        )
        nm = len(corruption_modes)
        K = 1 + self.n_corruptions

        # Chunk size
        eff_batch = self.batch_size if self.batch_size > 0 else K * 64
        chunk_queries = max(1, eff_batch // (K * nm))
        chunk_queries = min(chunk_queries, N)

        # Pre-allocate pool buffer for all modes
        max_pool = chunk_queries * K * nm
        pool_buffer = self._ensure_pool_buffer(max_pool)

        tie_gen = torch.Generator(device=self.device).manual_seed(self.seed)
        all_ranks: Dict[str, List[Tensor]] = {}

        for start in range(0, N, chunk_queries):
            end = min(start + chunk_queries, N)
            chunk_q = test_queries[start:end].to(self.device)
            CQ = chunk_q.shape[0]

            # Build all mode pools into one buffer
            combined, pools = CandidatePool.build_batched(
                chunk_q, self.sampler, self.n_corruptions,
                modes=corruption_modes, device=self.device, buffer=pool_buffer,
            )

            # One _score_pool call for all modes
            scores_dict = self._score_pool(combined)

            # Apply fusion per-mode if configured
            if self.fusion is not None and len(scores_dict) > 1 and nm == 1:
                fused = self.fusion(scores_dict, pools[0].K, pools[0].CQ, self.device)
                scores_dict.update(fused)

            # Split results per mode, compute ranks
            offset = 0
            for mode, pool_obj in zip(corruption_modes, pools):
                P = pool_obj.pool_size

                for mode_name, scores_flat in scores_dict.items():
                    mode_scores = scores_flat[offset:offset + P]
                    scores_2d = mode_scores.view(pool_obj.K, pool_obj.CQ).t()
                    pos_scores = scores_2d[:, 0]
                    neg_scores = scores_2d[:, 1:]
                    neg_valid = pool_obj.valid_mask[:, 1:]

                    key = f"{mode_name}_{mode}" if nm > 1 else mode_name
                    ranks = ranks_from_scores(
                        pos_scores, neg_scores,
                        valid_mask=neg_valid,
                        tie_handling="random",
                        generator=tie_gen,
                    )
                    all_ranks.setdefault(key, []).append(ranks)
                offset += P

        # Aggregate
        results = EvalResults(config={
            "n_queries": N,
            "n_corruptions": self.n_corruptions,
            "corruption_scheme": self.corruption_scheme,
            "batch_size": self.batch_size,
            "seed": self.seed,
        })

        if nm > 1:
            mode_names = set()
            for key in all_ranks:
                mode_names.add(key.rsplit("_", 1)[0])
            for mn in sorted(mode_names):
                combined_ranks = []
                for cm in corruption_modes:
                    key = f"{mn}_{cm}"
                    if key in all_ranks:
                        combined_ranks.append(torch.cat(all_ranks[key]))
                if combined_ranks:
                    results.metrics[mn] = ranking_metrics(torch.cat(combined_ranks))
        else:
            for key, rank_list in all_ranks.items():
                results.metrics[key] = ranking_metrics(torch.cat(rank_list))

        return results

    def _score_pool(self, pool: Tensor) -> Dict[str, Tensor]:
        """Score pool with sub-batching and optional padding.

        Subclasses override this for compiled/optimized scoring.

        Args:
            pool: ``[P, 3]`` triples to score (may contain multiple modes).

        Returns:
            ``{mode: [P]}`` scores for each scoring mode.
        """
        P = pool.shape[0]
        B = self.batch_size if self.batch_size > 0 else P
        all_scores: Dict[str, List[Tensor]] = {}

        for sb_start in range(0, P, B):
            sb_end = min(sb_start + B, P)
            batch = pool[sb_start:sb_end]
            actual_B = batch.shape[0]

            if self.batch_size > 0 and actual_B < B:
                padded = torch.zeros(B, 3, dtype=batch.dtype, device=batch.device)
                padded[:actual_B] = batch
                batch = padded

            scores = self.scorer(batch)

            for mode_name, s in scores.items():
                trimmed = s[:actual_B] if self.batch_size > 0 and actual_B < B else s
                all_scores.setdefault(mode_name, []).append(trimmed)

        return {k: torch.cat(v) for k, v in all_scores.items()}


__all__ = ["Evaluator", "ScorerFn", "FusionFn"]
