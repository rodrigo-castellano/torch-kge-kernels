"""Unified ranking evaluation — one path for KGE and reasoner models.

This module is the contract for how every KGE-style evaluation in this
workspace runs: tkk standalone, torch-ns (with or without a reasoner),
DpRL. There is one framework; the only per-model difference is the
``eval_scores`` hook.

Invariants
----------
1. **One evaluator**. All ranking evaluation flows through
   :func:`evaluate`. Callers don't branch on "is this a reasoner".

2. **Chunked over candidates**. Peak memory is bounded by
   ``batch_size × chunk_size`` (plus the model's own intermediates),
   independent of ``|E|``, ``|triples|``, or ``k``.

3. **Two candidate modalities**:

   - ``k=None`` → rank against all valid entities. If the sampler is
     domain-aware, this means the per-relation domain; otherwise all
     ``num_entities``.
   - ``k=N`` → rank against N sampled entities, same filtering.

4. **Always filtered**. Known facts (train ∪ valid ∪ test positives)
   never appear as candidates. Handled by the sampler's filter/domain
   machinery, not by the evaluator.

5. **Fullgraph-compile compatible**. The per-chunk scoring function
   (``model.eval_scores`` if defined, else
   :func:`kge_kernels.eval.eval_scores` over ``model.score``) runs with
   fixed-shape inputs and outputs. Compile boundary sits there. The
   Python loop
   over batches × modes × chunks runs eager.

Shape contract
--------------
Each call into the model's scoring hook operates on::

    q_buf     : [B, 3]     int64   (r, h, t) positive triples
    cand_buf  : [B, C]     int64   entity indices to score
    mode      : "tail" | "head"    which entity column is corrupted
    → scores  : [B, C]     float   score per (query, candidate)

Where ``B = batch_size`` and ``C = chunk_size`` are fixed across calls.
The final chunk of each batch is zero-padded to ``C`` so the compiled
graph keeps a single traced shape; padded columns are masked out of
the rank computation downstream.

Candidate provider
------------------
:class:`CandidateProvider` owns the "which entities compete with each
other" question. Given a ``query_batch: [B, 3]`` and ``mode``, it
returns::

    cand_entities : [B, K_fixed]   int64, padded with 0 where invalid
    valid_mask    : [B, K_fixed]   bool,  True where candidate counts

``K_fixed`` is set once at construction:

- ``k=N``          → ``K_fixed = N``
- ``k=None``       → ``K_fixed = num_entities`` (or ``max_domain_size``
                    if a domain map is installed)

Invalid positions cover (a) filtered-out known positives, (b) entities
outside the relation's domain, (c) pool-short positions when the
sampler can't produce enough unique candidates.

The evaluator chunks ``[B, K_fixed]`` into slices of ``chunk_size`` for
scoring; the mask flows through so invalid slots contribute nothing to
ranks regardless of what score the compiled graph happens to write.

Compile boundary (why here and not elsewhere)
---------------------------------------------
Modelled on DpRL's PPO ranking loop (``kge_experiments/ppo/compilation.py``):

- **The step is compiled, the orchestrator is Python.** Each call to
  ``score_chunk(q_buf, cand_buf)`` is a self-contained tensor
  computation with fixed shapes. Fullgraph + reduce-overhead captures
  a CUDA graph; subsequent calls replay in microseconds.

- **Static-address buffers.** ``q_buf``, ``cand_buf``, ``score_buf``
  are preallocated once per evaluator instance and registered via
  ``torch._dynamo.mark_static_address``. This lets CUDA graph replay
  reuse the same device addresses across chunks.

- **``torch.compiler.cudagraph_mark_step_begin()``** is called before
  each compiled step so each chunk is a well-defined graph replay.

- **No data-dependent branching inside the compiled step.** Mode
  selection (``"tail"`` vs ``"head"``) is resolved via two separate
  compiled specialisations or passed as a Python constant so
  ``torch.compile`` traces each branch independently.

Per-model implementations
-------------------------
::

    class KGEModel:
        def eval_scores(self, q_buf, cand_buf, mode):
            # Per-atom batched scoring; one score_triples call of
            # shape [B*C]. For large C and KGE-factorable scoring,
            # can specialise to matmul (score_all_tails); the shape
            # contract is identical either way.
            ...

    class ReasonerModel:
        def eval_scores(self, q_buf, cand_buf, mode):
            # Assemble [B*C, 3] atom pool from (q, cand) under mode,
            # run the reasoner forward (grounder + reasoning layers
            # operating on the pre-built fact index), reshape [B, C].
            ...

The reasoner doesn't "lose" fast eval — it runs the same forward it
uses at training time, over an atom pool sized ``B*C``. Chunking is
how large ``|E|`` becomes manageable for any model.

Replaces
--------
- ``kge_kernels.eval.Evaluator._evaluate_exhaustive``
- ``kge_kernels.eval.Evaluator._evaluate_sampled``
- ``kge_kernels.eval.Evaluator._evaluate_sampled_pool``
- torch-ns ``_build_eval_cache`` + ``evaluate_chunked``
- torch-ns ``_run_tkk_kge_only`` bypass (no longer needed — no path
  has a reason to skip ``evaluate``)
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Protocol

import torch
from torch import Tensor


Mode = Literal["tail", "head"]


# Per-model cache stored as an attribute on the model — the _EvalState
# for a given ``(B, K_fixed, modes, compile, device)`` lives inside the
# dict at ``model._tkk_eval_state_cache``.
#
# Why: ``torch.compile(...)`` wrappers own CUDA-graph pools that persist
# for the wrapper's lifetime. If we rebuild the wrapper on every
# ``evaluate()`` call, each call captures a fresh pool and the old ones
# never free. In a sweep that calls ``evaluate()`` 100+ times, the pools
# accumulate until OOM. The fix is to compile **once per model** and
# reuse — so ``evaluate()`` fetches the cached compiled scorer plus
# static buffers from here.
#
# Stored on the model (rather than a module-level WeakKeyDictionary) so
# the cache is naturally collected when the model is deleted: the
# closures inside _EvalState strongly reference the model, which forms
# a reference cycle with the attribute. CPython's cyclic GC handles
# that the same way it handles any nn.Module's internal parameter /
# hook cycles.
_EVAL_STATE_ATTR = "_tkk_eval_state_cache"


class ScoresModel(Protocol):
    """Duck-type the evaluator requires of any scored model."""

    def eval_scores(
        self,
        q_buf: Tensor,      # [B, 3]   int64, (r, h, t)
        cand_buf: Tensor,   # [B, C]   int64, entity indices
        mode: Mode,
    ) -> Tensor:            # [B, C]   float
        ...


class CandidateProvider:
    """Produces per-batch candidate entity pools for ranking evaluation.

    Wraps a tkk ``Sampler`` so the evaluator doesn't need to know
    anything about filtering, domain restriction, or exhaustive-vs-
    sampled dispatch. One instance per evaluation call. ``K_fixed``
    is chosen at construction and stays constant across batches so
    downstream buffers can be preallocated once.

    Two modalities (set via ``k``):

    - ``k=None``: exhaustive over the valid pool for each query's
      relation. ``K_fixed = max_pool_len - 1`` if the sampler has a
      domain (else ``num_entities - 1``) — i.e. "every entity in the
      domain, minus the true entity slot". Rows with shorter valid
      pools get ``valid_mask=False`` on the tail.
    - ``k=N`` (int): sample N entities per query; ``K_fixed = N``.
      ``valid_mask`` is ``False`` where the sampler couldn't find
      enough filtered-unique candidates in the available pool.

    The sampler's ``filter=True`` removes known positives (train ∪
    valid ∪ test facts installed at sampler construction) from the
    candidate pool. The true entity for each query is therefore NOT
    in the returned candidates; evaluators must score it separately
    and combine.

    Per-relation domain restriction: pass ``head_domain`` /
    ``tail_domain`` (``{relation_id: {valid_entity_ids}}``) to mask out
    candidates that aren't in the relation's domain. This is the
    standard countries/ablation eval protocol.
    """

    def __init__(
        self,
        sampler,
        num_entities: int,
        k: Optional[int],
        *,
        head_domain: Optional[Dict[int, set]] = None,
        tail_domain: Optional[Dict[int, set]] = None,
    ) -> None:
        self.sampler = sampler
        self.num_entities = num_entities
        self.k = k

        has_domain = getattr(sampler, "_has_domain_info", lambda: False)()
        if k is None:
            pool_size = (
                (getattr(sampler, "max_pool_len", num_entities) - 1)
                if has_domain
                else (num_entities - 1)
            )
            self.K_fixed = int(pool_size)
            self._is_exhaustive = True
        else:
            self.K_fixed = int(k)
            self._is_exhaustive = False

        # Per-relation domain masks: [num_relations, num_entities] booleans.
        self._head_domain_mask: Optional[Tensor] = None
        self._tail_domain_mask: Optional[Tensor] = None
        if head_domain is not None or tail_domain is not None:
            num_relations = sampler.num_relations
            device = sampler.device
            if head_domain is not None:
                self._head_domain_mask = self._build_domain_mask(
                    head_domain, num_relations, num_entities, device
                )
            if tail_domain is not None:
                self._tail_domain_mask = self._build_domain_mask(
                    tail_domain, num_relations, num_entities, device
                )

    @staticmethod
    def _build_domain_mask(
        domain_dict: Dict[int, set],
        num_relations: int,
        num_entities: int,
        device: torch.device,
    ) -> Tensor:
        """Convert ``{rel: {ents}}`` dict to ``[R, E]`` boolean mask."""
        mask = torch.zeros(num_relations, num_entities, dtype=torch.bool, device=device)
        for r, ents in domain_dict.items():
            if not ents:
                continue
            ent_t = torch.tensor(sorted(ents), dtype=torch.long, device=device)
            mask[int(r), ent_t] = True
        return mask

    def candidates(
        self,
        query_batch: Tensor,   # [B, 3]  int64, (r, h, t)
        mode: Mode,
    ) -> tuple[Tensor, Tensor]:
        """Return candidate entities + validity mask for a query batch.

        Returns:
            cand_entities: ``[B, K_fixed]`` int64 entity indices. Padded
                with 0 where invalid (use the mask).
            valid_mask:    ``[B, K_fixed]`` bool.
        """
        # Sampler draws `num_negatives=None` for exhaustive, int for sampled.
        num_neg = None if self._is_exhaustive else self.K_fixed
        neg, mask = self.sampler.corrupt(
            query_batch,
            num_negatives=num_neg,
            mode=mode,
            filter=True,
            unique=True,
            return_mask=True,
        )
        # neg: [B, K, 3] in (r, h, t) format; extract the column that was
        # actually corrupted — that's our candidate entity for each slot.
        col = 1 if mode == "head" else 2
        cand_entities = neg[:, :, col]   # [B, K]

        # Sampler marks shortfalls with the sentinel ``num_entities``
        # (out-of-range). Clamp to 0 so downstream embedding lookups
        # never fault; the validity mask still gates contribution.
        cand_entities = cand_entities.clamp_(0, self.num_entities - 1)

        # Pad/truncate to K_fixed so downstream compile shapes stay static.
        B, K = cand_entities.shape
        if K < self.K_fixed:
            pad = self.K_fixed - K
            cand_entities = torch.cat(
                [cand_entities, torch.zeros(B, pad, dtype=cand_entities.dtype, device=cand_entities.device)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(B, pad, dtype=torch.bool, device=mask.device)],
                dim=1,
            )
        elif K > self.K_fixed:
            # Shouldn't happen for correctly-sized sampler; clip defensively.
            cand_entities = cand_entities[:, : self.K_fixed]
            mask = mask[:, : self.K_fixed]

        # Per-relation domain restriction (e.g. countries: head of relation r
        # must be in head_domain[r]). Applied after sampling so we mask
        # out-of-domain candidates regardless of sampler's domain config.
        domain_mask = self._head_domain_mask if mode == "head" else self._tail_domain_mask
        if domain_mask is not None:
            rels = query_batch[:, 0]                      # [B]
            row_mask = domain_mask[rels]                  # [B, num_entities]
            cand_in_domain = row_mask.gather(1, cand_entities)  # [B, K_fixed]
            mask = mask & cand_in_domain
        return cand_entities, mask


class _EvalState:
    """Cached eval buffers + compiled scorers for one model+config combo.

    One instance per ``(model, B, K_fixed, modes, compile, device)``;
    stored on the model under :data:`_EVAL_STATE_ATTR`. Buffers are
    allocated once here and reused across every :func:`evaluate` call
    for the same model — so the CUDA graph captured on the first call
    replays on subsequent calls instead of rebuilding a fresh pool
    each time.

    Without this cache, a sweep that evaluates 100 val epochs × 2 modes
    creates 200 separate ``torch.compile`` wrappers, each retaining its
    own CUDA-graph pool; on a 24 GB card this trips OOM around val
    epoch ~50 on family rotate. With this cache, the pool count stays
    at ``len(modes)`` per model.
    """

    __slots__ = (
        "B", "K_fixed", "P", "device", "modes",
        "q_buf", "pool_buf", "pool_valid_buf", "true_idx_const",
        "pool_tmp", "valid_tmp",
        "score_fns",
    )

    def __init__(
        self,
        model,
        B: int,
        K_fixed: int,
        device: torch.device,
        modes: tuple,
        compile_enabled: bool,
    ) -> None:
        self.B = B
        self.K_fixed = K_fixed
        self.P = 1 + K_fixed
        self.device = device
        self.modes = tuple(modes)

        # Static-address buffers — the compile boundary reads from
        # these. Addresses never change across calls, so the CUDA graph
        # captured on the first invocation replays directly.
        self.q_buf = torch.empty(B, 3, dtype=torch.long, device=device)
        self.pool_buf = torch.empty(B, self.P, dtype=torch.long, device=device)
        self.pool_valid_buf = torch.empty(B, self.P, dtype=torch.bool, device=device)
        self.true_idx_const = torch.zeros(B, dtype=torch.long, device=device)

        # Eager-side scratch for composing the pool before the static
        # copy. Outside the compile boundary but still reused to keep
        # the allocator quiet.
        self.pool_tmp = torch.empty(B, self.P, dtype=torch.long, device=device)
        self.valid_tmp = torch.empty(B, self.P, dtype=torch.bool, device=device)

        if hasattr(torch, "_dynamo"):
            for _b in (self.q_buf, self.pool_buf, self.pool_valid_buf, self.true_idx_const):
                torch._dynamo.mark_static_address(_b)

        # One compiled scorer per corruption mode. Mode is baked in via
        # closure so each compile specialises one direction; the model
        # reference is captured once here and lives for the state's
        # lifetime. Prefer ``model.eval_scores`` if the model defines it
        # (e.g. ns ReasonerModel for the candidate-pool replay path),
        # else use the default :func:`kge_kernels.eval.eval_scores` over
        # ``model.score``.
        from .eval_hooks import eval_scores as _default_eval_scores
        if hasattr(model, "eval_scores"):
            _scores = model.eval_scores
        else:
            def _scores(q, cand, m):
                return _default_eval_scores(model, q, cand, m)
        self.score_fns = {}
        for mode in self.modes:
            def _make(m=mode):
                def _score(q: Tensor, cand: Tensor) -> Tensor:
                    return _scores(q, cand, m)
                return _score
            fn = _make()
            if compile_enabled:
                fn = torch.compile(fn, fullgraph=True, mode="reduce-overhead")
            self.score_fns[mode] = fn


def _get_eval_state(
    model,
    B: int,
    K_fixed: int,
    device: torch.device,
    modes: tuple,
    compile_enabled: bool,
) -> _EvalState:
    """Fetch or allocate the cached :class:`_EvalState` on this model."""
    cache = getattr(model, _EVAL_STATE_ATTR, None)
    if cache is None:
        cache = {}
        # object.__setattr__ bypasses nn.Module's Parameter / Buffer /
        # submodule routing — this is plain per-instance state, not
        # part of the model's persisted spec.
        object.__setattr__(model, _EVAL_STATE_ATTR, cache)
    key = (B, K_fixed, tuple(modes), bool(compile_enabled), str(device))
    st = cache.get(key)
    if st is None:
        st = _EvalState(model, B, K_fixed, device, modes, compile_enabled)
        cache[key] = st
    return st


def clear_eval_cache(model) -> None:
    """Drop the cached eval buffers + compiled scorers on ``model``.

    Not normally needed: the cache is collected with the model itself
    when the model goes out of scope. Call this only to force a rebuild
    mid-run (e.g. the model's compile spec changed in-place) or to
    reclaim CUDA-graph pools early before the model is destroyed.
    """
    if hasattr(model, _EVAL_STATE_ATTR):
        try:
            object.__delattr__(model, _EVAL_STATE_ATTR)
        except AttributeError:
            pass


def evaluate(
    model: ScoresModel,
    triples: Tensor,                         # [N, 3]   int64
    provider: CandidateProvider,
    *,
    scheme: Literal["tail", "head", "both"] = "both",
    batch_size: int = 512,
    chunk_size: int = 2048,
    seed: int = 0,
    device: Optional[torch.device] = None,
    compile: bool = True,
    tie_handling: Literal["average", "random"] = "random",
) -> dict[str, float]:
    """Unified ranking evaluation. See module docstring for the contract.

    Uses DpRL-style preallocated static-address buffers: ``q_buf``,
    ``pool_buf`` (col 0 = true entity, cols 1..K_fixed = candidates),
    ``pool_valid_buf``, and the compiled ``score_buf``. Every iteration
    writes into these buffers via ``.copy_()`` — **no per-step
    allocations**, so a long sweep can't fragment the CUDA allocator.

    The scoring step is compiled with ``fullgraph=True,
    mode='reduce-overhead'`` and wrapped with
    ``torch.compiler.cudagraph_mark_step_begin()`` between calls; the
    first call per ``(B, 1+K_fixed, mode)`` shape captures a CUDA
    graph, subsequent calls replay it.

    Returns ``{'MRR', 'Hits@1', 'Hits@3', 'Hits@10'}``.
    """
    del chunk_size  # superseded — the model owns scoring chunking

    from .ranking import compute_ranks, metrics_from_ranks

    if device is None:
        device = next(model.parameters()).device

    B = batch_size
    K_fixed = provider.K_fixed
    modes = ("tail", "head") if scheme == "both" else (scheme,)

    # Fetch-or-allocate the per-model buffers + compiled scorers. The
    # first call for a given ``(model, B, K_fixed, modes, compile)``
    # combo allocates and compiles; every subsequent call reuses,
    # including across val epochs — that's what keeps the CUDA-graph
    # pool from leaking. See ``_EvalState`` docstring.
    state = _get_eval_state(model, B, K_fixed, device, modes, compile)
    q_buf = state.q_buf
    pool_buf = state.pool_buf
    pool_valid_buf = state.pool_valid_buf
    true_idx_const = state.true_idx_const
    _pool_tmp = state.pool_tmp
    _valid_tmp = state.valid_tmp
    score_fns = state.score_fns

    # Preserve and restore training mode (only relevant for nn.Module scorers).
    was_training = getattr(model, "training", False)
    if hasattr(model, "eval") and callable(model.eval):
        model.eval()
    try:
        tie_gen = torch.Generator(device=device).manual_seed(seed)
        all_ranks: list[Tensor] = []
        triples = triples.to(device=device, dtype=torch.long)
        N = triples.shape[0]

        with torch.no_grad():
            for q_start in range(0, N, B):
                q_end = min(q_start + B, N)
                q_slice = triples[q_start:q_end]                   # [aB, 3]
                actual_B = q_slice.shape[0]

                # Pad q_buf with zeros for the batch tail.
                q_buf[:actual_B].copy_(q_slice)
                if actual_B < B:
                    q_buf[actual_B:].zero_()

                for mode in modes:
                    # Candidates for the active-rows portion.
                    cand_ents, valid_mask = provider.candidates(
                        q_slice, mode,
                    )                                              # [aB, K_fixed], [aB, K_fixed]

                    # Assemble [B, P] pool with col 0 = true entity, cols 1.. = candidates.
                    true_col = 2 if mode == "tail" else 1
                    _pool_tmp[:actual_B, 0] = q_slice[:, true_col]
                    _pool_tmp[:actual_B, 1:].copy_(cand_ents)
                    _valid_tmp[:actual_B, 0] = True
                    _valid_tmp[:actual_B, 1:].copy_(valid_mask)
                    if actual_B < B:
                        _pool_tmp[actual_B:].zero_()
                        _valid_tmp[actual_B:].zero_()
                    pool_buf.copy_(_pool_tmp)
                    pool_valid_buf.copy_(_valid_tmp)

                    # Compiled scoring step — graph replays onto the
                    # static buffers captured on the first call.
                    if compile:
                        torch.compiler.cudagraph_mark_step_begin()
                    pool_scores = score_fns[mode](q_buf, pool_buf)  # [B, P]

                    # Ranking (eager, O(B*P) — negligible next to the matmul).
                    ranks = compute_ranks(
                        pool_scores[:actual_B], true_idx_const[:actual_B],
                        valid_mask=pool_valid_buf[:actual_B],
                        tie_handling=tie_handling, generator=tie_gen,
                    )
                    all_ranks.append(ranks.clone())  # clone: detach from graph buffer

        if not all_ranks:
            return {"MRR": 0.0, "Hits@1": 0.0, "Hits@3": 0.0, "Hits@10": 0.0}
        return metrics_from_ranks(torch.cat(all_ranks))
    finally:
        if was_training and hasattr(model, "train") and callable(model.train):
            model.train()


__all__ = [
    "CandidateProvider",
    "Mode",
    "ScoresModel",
    "clear_eval_cache",
    "evaluate",
]
