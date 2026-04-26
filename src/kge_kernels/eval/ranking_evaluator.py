"""Compile-aware filtered ranking evaluator.

One class, one method, one result. Setup-once: ``__init__`` allocates
static-address buffers and compiles the scorer once per mode (mode is
Python str, baked into the specialized graph — no branch in the hot
path). Run-many: ``evaluate(triples)`` returns a :class:`RankingResult`.

Subclassing
-----------
Override :meth:`evaluate` ONLY to wrap the result in a different
envelope (e.g. DpRL's ``EvalResults``). The chunked loop is final — to
change scoring, pass a different ``scorer=``; to change candidate
generation, pass a different ``candidates=``.

Compile contract
----------------
- ``B`` (batch_size) and ``K_fixed`` (from CandidateSource) are frozen
  at construction.
- ``q_buf [B, 3]``, ``pool_buf [B, K_fixed+1]``, ``valid_buf [B, K_fixed+1]``
  pre-allocated with ``mark_static_address``. CUDA graph captures their
  addresses on first scorer call.
- One compiled scorer per mode; mode baked into the closure via default
  arg. Each call inside :meth:`evaluate` does
  ``cudagraph_mark_step_begin()`` then ``compiled[mode](q_buf, pool_buf)``.
- Sampler call (``candidates.candidates(...)``) and ranking
  (``compute_ranks``) are eager — the compile boundary is the scorer.
- Per-instance compile cache. Reuse the evaluator across many
  ``evaluate()`` calls; different instances on the same model recompile.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor

from .candidates import CandidateSource, Mode
from .ranking import compute_ranks, metrics_from_ranks


ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]
"""``(q_buf [B, 3], pool_buf [B, K_fixed+1], mode) -> [B, K_fixed+1]``.

q_buf rows: ``(relation, head, tail)`` — original triples, padded.
pool_buf col 0 = the true entity for the corruption side; cols 1..K_fixed
= candidates (or padding). Returns score per (query, slot).
Mode is a Python string fixed per compile — baked into the specialized
graph, no branch.
"""


@dataclass(frozen=True)
class RankingResult:
    """Raw outputs of one :meth:`RankingEvaluator.evaluate` call.

    Aggregations are methods on the result, computed on demand from the
    raw fields. Slice ``ranks`` / ``scores`` directly for custom analyses.
    """

    triples: Tensor                          # [N, 3] queries we ranked
    modes: Tuple[Mode, ...]                  # M modes
    ranks: Tensor                            # [M, N] 1-based ranks
    valid: Tensor                            # [M, N] False = invalid/padded
    scores: Optional[Tensor] = None          # [M, N, K_fixed+1] iff track_scores
    elapsed_s: float = 0.0

    def metrics(self, ks: Tuple[int, ...] = (1, 3, 10)) -> Dict[str, float]:
        """MRR + Hits@k pooled across all modes."""
        flat = self.ranks[self.valid]
        return metrics_from_ranks(flat, ks=ks)

    def metrics_per_mode(
        self, ks: Tuple[int, ...] = (1, 3, 10)
    ) -> Dict[Mode, Dict[str, float]]:
        """One MRR + Hits@k entry per corruption side."""
        out: Dict[Mode, Dict[str, float]] = {}
        for i, mode in enumerate(self.modes):
            row_ranks = self.ranks[i][self.valid[i]]
            out[mode] = metrics_from_ranks(row_ranks, ks=ks)
        return out

    def metrics_per_group(
        self,
        group_ids: Tensor,
        ks: Tuple[int, ...] = (1, 3, 10),
    ) -> Dict[int, Dict[str, float]]:
        """One MRR + Hits@k per group, where ``group_ids`` is any
        ``[N]`` integer tensor — depths, types, custom labels, etc.

        Generic grouping primitive. Per-relation, per-depth, etc. are
        all the same call with a different ``group_ids``::

            result.metrics_per_group(query_depths)
            result.metrics_per_group(triples[:, 0])           # per-relation
            result.metrics_per_group(some_difficulty_label)
        """
        gids = group_ids.cpu()
        out: Dict[int, Dict[str, float]] = {}
        for g in gids.unique().tolist():
            row_mask = (gids == g)
            sub_ranks = self.ranks[:, row_mask]
            sub_valid = self.valid[:, row_mask]
            flat = sub_ranks[sub_valid]
            out[int(g)] = metrics_from_ranks(flat, ks=ks)
        return out

    def metrics_per_relation(
        self, ks: Tuple[int, ...] = (1, 3, 10)
    ) -> Dict[int, Dict[str, float]]:
        """One MRR + Hits@k per relation. Groups by ``triples[:, 0]``."""
        return self.metrics_per_group(self.triples[:, 0], ks=ks)


class RankingEvaluator:
    """Compile-aware filtered ranking evaluator. See module docstring.

    The lifecycle is two-phase:

    - ``__init__`` does the expensive setup: allocates static-address
      buffers, compiles the scorer once per mode. Pay once.
    - ``evaluate(triples)`` runs the chunked loop. The CUDA graph
      captured on the first call replays for subsequent ones — cheap.
    """

    def __init__(
        self,
        scorer: ScoreFn,
        candidates: CandidateSource,
        *,
        batch_size: int,
        modes: Tuple[Mode, ...] = ("head", "tail"),
        device: torch.device,
        compile: bool = True,
        compile_mode: str = "reduce-overhead",
        tie_handling: Literal["average", "random"] = "average",
        seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.modes = tuple(modes)
        self.device = device
        self.tie_handling = tie_handling
        self.seed = int(seed)
        self.candidates = candidates
        self.K_fixed = int(candidates.K_fixed)
        self.P = self.K_fixed + 1

        B = self.batch_size
        # Static-address buffers — captured by the CUDA graph on first scorer call.
        self._q_buf = torch.empty(B, 3, dtype=torch.long, device=device)
        self._pool_buf = torch.empty(B, self.P, dtype=torch.long, device=device)
        self._valid_buf = torch.empty(B, self.P, dtype=torch.bool, device=device)
        self._true_idx_const = torch.zeros(B, dtype=torch.long, device=device)

        # Eager scratch for assembling pool/valid before the static copy.
        # Outside the compile boundary; reused to keep the allocator quiet.
        self._pool_scratch = torch.empty(B, self.P, dtype=torch.long, device=device)
        self._valid_scratch = torch.empty(B, self.P, dtype=torch.bool, device=device)

        if hasattr(torch, "_dynamo"):
            for buf in (self._q_buf, self._pool_buf, self._valid_buf, self._true_idx_const):
                torch._dynamo.mark_static_address(buf)

        # One compiled scorer per mode. Mode baked in via default-arg
        # closure so each compile specializes one direction; no branch
        # in the compile region.
        self._compile_enabled = bool(compile)
        self._compiled: Dict[Mode, Callable[[Tensor, Tensor], Tensor]] = {}
        for mode in self.modes:
            def _make(m: Mode = mode):
                def _fn(q: Tensor, pool: Tensor) -> Tensor:
                    return scorer(q, pool, m)
                return _fn
            fn = _make()
            if self._compile_enabled:
                fn = torch.compile(fn, fullgraph=True, mode=compile_mode)
            self._compiled[mode] = fn

    @torch.no_grad()
    def evaluate(
        self,
        triples: Tensor,
        *,
        track_scores: bool = False,
    ) -> RankingResult:
        """Run filtered ranking over ``triples``. Returns a :class:`RankingResult`.

        ``track_scores=True`` allocates and fills a
        ``[len(modes), N, K_fixed+1]`` score tensor (memory:
        ``M * N * (K+1) * 4`` bytes). Default off.
        """
        device = self.device
        B = self.batch_size
        P = self.P
        modes = self.modes
        M = len(modes)
        triples = triples.to(device=device, dtype=torch.long)
        N = int(triples.shape[0])

        # Output tensors. Allocate once to avoid Python-list .append-then-cat.
        ranks_out = torch.zeros(M, N, dtype=torch.float32, device=device)
        valid_out = torch.zeros(M, N, dtype=torch.bool, device=device)
        scores_out: Optional[Tensor] = None
        if track_scores:
            scores_out = torch.zeros(M, N, P, dtype=torch.float32, device=device)

        # Save / restore training mode (only if the scorer has it — PPOScorer doesn't).
        scorer_obj = getattr(self, "_scorer_obj", None)  # not stored; check the closure target instead
        # Generic: try to find a model with .training. Most callers pass nn.Module-bound methods.
        # Skip by default; if needed, callers can do this themselves around the call.

        tie_gen = torch.Generator(device=device).manual_seed(self.seed)
        t0 = time.perf_counter()

        if N == 0:
            return RankingResult(
                triples=triples, modes=modes,
                ranks=ranks_out, valid=valid_out,
                scores=scores_out,
                elapsed_s=time.perf_counter() - t0,
            )

        for q_start in range(0, N, B):
            q_end = min(q_start + B, N)
            q_slice = triples[q_start:q_end]                  # [actual_B, 3]
            actual_B = int(q_slice.shape[0])

            # Pad q_buf with zeros for the batch tail (keeps the compiled
            # graph at its captured shape).
            self._q_buf[:actual_B].copy_(q_slice)
            if actual_B < B:
                self._q_buf[actual_B:].zero_()

            for mode_idx, mode in enumerate(modes):
                # Eager: get candidates for the active rows.
                cand_ents, cand_valid = self.candidates.candidates(q_slice, mode)
                # cand_ents: [actual_B, K_fixed]; cand_valid: [actual_B, K_fixed]

                # Assemble pool [B, P]: col 0 = true entity, cols 1.. = candidates.
                true_col = 2 if mode == "tail" else 1
                self._pool_scratch[:actual_B, 0] = q_slice[:, true_col]
                self._pool_scratch[:actual_B, 1:].copy_(cand_ents)
                self._valid_scratch[:actual_B, 0] = True
                self._valid_scratch[:actual_B, 1:].copy_(cand_valid)
                if actual_B < B:
                    self._pool_scratch[actual_B:].zero_()
                    self._valid_scratch[actual_B:].zero_()
                self._pool_buf.copy_(self._pool_scratch)
                self._valid_buf.copy_(self._valid_scratch)

                # Compiled scoring step. Graph replays onto static buffers.
                if self._compile_enabled:
                    torch.compiler.cudagraph_mark_step_begin()
                pool_scores = self._compiled[mode](self._q_buf, self._pool_buf)  # [B, P]

                # Eager ranking — O(B*P), negligible vs the matmul.
                chunk_ranks = compute_ranks(
                    pool_scores[:actual_B],
                    self._true_idx_const[:actual_B],
                    valid_mask=self._valid_buf[:actual_B],
                    tie_handling=self.tie_handling,
                    generator=tie_gen,
                )

                # Write into output tensors.
                ranks_out[mode_idx, q_start:q_start + actual_B] = chunk_ranks
                valid_out[mode_idx, q_start:q_start + actual_B] = True
                if scores_out is not None:
                    scores_out[mode_idx, q_start:q_start + actual_B] = pool_scores[:actual_B]

        return RankingResult(
            triples=triples,
            modes=modes,
            ranks=ranks_out,
            valid=valid_out,
            scores=scores_out,
            elapsed_s=time.perf_counter() - t0,
        )


__all__ = ["RankingEvaluator", "RankingResult", "ScoreFn"]
