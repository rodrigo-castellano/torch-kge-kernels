# tkk–torch-ns consolidation: status

Working doc for the phase-1 consolidation: **one framework, one data loader,
one evaluator, shared by tkk (standalone) and torch-ns (KGE-only and reasoner
paths)**. Reasoner specialisations live in ns, KGE primitives live in tkk.

## Goals

1. **One unified evaluation framework.** Same call contract for KGE and
   reasoner; only the per-model `eval_scores(q, cand, mode)` hook differs
   (KGE: matmul via `score_all_tails/heads`; reasoner: atom-pool forward).
2. **One unified training helper.** Same `train_epoch(model, sampler, ...)`
   for both; only `model.train_step(pos, neg, mask)` differs (KGE: lean
   `cat`+BCE; reasoner: static-shape padded pool + mask-aware weighted BCE).
3. **Static-buffer design throughout.** Every compile boundary consumes
   preallocated, mark-static-address buffers; no per-iteration allocations
   in the hot path — follows DpRL's PPO ranking / rollout pattern.
4. **`torch.compile(fullgraph=True, mode='reduce-overhead')`** on every
   compiled step, plus `torch.compiler.cudagraph_mark_step_begin()` between
   calls so a single captured graph replays.
5. **KGE speed ≥ tkk-standalone.** ns runs with `--m no_reasoner` must be at
   least as fast as `tkk.train_model` at the same config.

## What is done

### tkk (library)
- `kge_kernels.models.base.KGEModel.eval_scores(q_buf, cand_buf, mode)` —
  default matmul path via `score_all_tails`/`score_all_heads` + `.gather`.
  Inherited by all 8 KGE models.
- `kge_kernels.models.base.KGEModel.train_step(pos, neg, mask)` — lean
  default: `cat([pos, neg_valid])` + BCE. Overridden by a `_RotateLossMixin`
  for RotatE/RotatENS (sigmoid+BCE to avoid saturated gradients).
- `kge_kernels.eval.unified.CandidateProvider` — wraps tkk Sampler; returns
  `[B, K_fixed]` candidates + validity mask; one instance per eval call; K_fixed
  fixed at construction (`k=None` → exhaustive, `k=N` → sampled).
- `kge_kernels.eval.unified.evaluate` — the single eval entry point. Static
  `q_buf [B, 3]`, `pool_buf [B, 1+K_fixed]`, `pool_valid_buf [B, 1+K_fixed]`
  mark-static-address; compiles one scorer per mode with
  `fullgraph=True, mode='reduce-overhead'`; `cudagraph_mark_step_begin()`
  per call; ranks via `compute_ranks` (eager, small). **Fixed 2026-04-23**:
  the compiled scorer + static buffers are now cached per-model via the
  `_tkk_eval_state_cache` attribute so `torch.compile` runs once per
  `(model, B, K_fixed, modes, compile, device)` combo — no more per-call
  leak. Cache is collected with the model (CPython cyclic GC).
- `kge_kernels.training.epoch.train_epoch` — static-buffer compiled
  training step. Per-model `_tkk_train_state_cache` holds preallocated
  `pos_buf [B, 3]`, `neg_buf [B, K, 3]`, `mask_buf [B, K]`,
  `pos_valid_buf [B]` plus the compiled step. Uses
  `torch.compile(fullgraph=True, mode='default')` — `reduce-overhead`
  hit a backward-pass multi-use tensor error on KGE models where e.g.
  ComplEx splits an embedding into real/imag parts that participate in
  multiple backward nodes. Default mode avoids it, still gets inductor
  lowering for forward + backward.
- `kge_kernels.models.base.KGEModel.train_step(pos, neg, mask, pos_valid)`
  — static-shape default: `cat([pos, neg_flat])` over `[B + B*K, 3]`
  atoms, mask-aware BCE reduction gating both invalid negatives
  (`mask=False`) and padded positives (`pos_valid=False`, for partial
  last batch). Fullgraph-compile compatible.
- Fixed `RotatENS.reset_parameters` to use `±6/√half_dim` entity init
  (ns-old convention) instead of `±embedding_range` — was a pre-existing
  drift vs the "ns-old aligned" claim in the class docstring.

### torch-ns
- `experiments.model.KGEModel.eval_scores` and `.train_step` delegate to
  `self._tkk` (the underlying tkk model). Applies ns's output-layer sigmoid
  convention where applicable.
- `experiments.model.ReasonerModel.eval_scores` chunks internally over
  candidates (`_EVAL_ATOM_BUDGET = 1024`) so reasoner forward over
  exhaustive `K_fixed` stays memory-bounded.
- `experiments.model.ReasonerModel.train_step` builds `[B*(1+K), 3]`
  static-shape atom pool with interleaved `[pos, *neg]`, runs the compiled
  forward, computes mask-aware weighted BCE over `kge_score` +
  `reasoning_score` + optional `gate_reg`.
- `experiments.train.train_loop` now calls `kge_kernels.training.train_epoch`
  — no more ns-local `_build_train_cache` loop.
- `experiments.train._run_unified_ranking_eval` → `kge_kernels.eval.evaluate`
  wrapper with ns metric-key translation.
- Removed `_run_tkk_kge_only` bypass — `no_reasoner` runs through the same
  `main() → setup_data → setup_model → train_loop` as any reasoner.
- Removed redundant `_reinit_rotate_entity_` — tkk's new `RotatENS` init
  covers it.
- `_build_train_cache` batch_size semantic fixed: now queries/batch
  (matching tkk's convention), not atoms/batch (legacy ns).
- `train_epoch` loss switched to mask-aware `sum / mask.sum()` reduction,
  matching tkk's "BCE over valid atoms only".
- RNG state save/restore around val_cache build + val MRR eval so the
  sampler's RNG consumption during val doesn't shift the training RNG
  trajectory.

### Sweep harness
- `scripts/run_3way_sweep.py` now calls
  `torch._dynamo.reset()` +
  `torch._inductor.cudagraph_trees.reset_cudagraph_trees()` +
  `torch.cuda.empty_cache()` +
  `gc.collect()` between sub-runs. Still not enough for the per-call
  compile leak below.
- tkk column uses `rotate_ns` for rotate rows (matching what ns uses via
  `_tkk = build_model(name='rotate_ns', ...)`).

## Results

Seed=0, `~/repos/data-swarm/main/<dataset>`, config in
`scripts/run_3way_sweep.py`. Times are wall-clock per 100-300 epoch run.

### Pre-consolidation baseline (2026-04-22, `scripts/baseline_sweep_20260422.log`)
| Dataset | KGE | tkk | ns-old | ns-fast |
|---|---|---|---|---|
| ablation_d3 | complex | 93.3% / 5s | 85.7% / 12s | 90.1% / 3s |
| ablation_d3 | rotate | 96.8% / 3s | 100.0% / 13s | 98.0% / 3s |
| countries_s3 | complex | 81.6% / 3s | 75.8% / 10s | 86.8% / 3s |
| countries_s3 | rotate | 95.8% / 4s | 95.1% / 13s | 96.7% / 4s |
| family | complex | 91.0% / 24s | 88.3% / 140s | 91.2% / 56s |
| family | rotate | 85.7% / 29s | 80.3% / 162s | 92.0% / 62s |
| wn18rr | complex | 42.7% / — | — | — |
| wn18rr | rotate | — | — | — |

Significant divergence across columns — the original motivation for the
consolidation. ns-old was 5–10× slower and 1–10pp worse on MRR; ns-fast
was a separate fast path that diverged further.

### Latest sweep (2026-04-23, model-aware eval B, static-buffer training compile)
Results from `output/runs/20260423-model-aware-B/family.log` and
`output/runs/20260423-full-sweep/small-datasets.log`.

| Dataset | KGE | tkk | ns-old | ns-fast |
|---|---|---|---|---|
| ablation_d3 | complex | 93.3% / 5s | 93.3% / 10s | 93.3% / 7s |
| ablation_d3 | rotate | 100.0% / 4s | 100.0% / 11s | 100.0% / 8s |
| countries_s3 | complex | 84.0% / 5s | 84.0% / 7s | 84.0% / 7s |
| countries_s3 | rotate | 95.8% / 4s | 95.8% / 7s | 95.8% / 7s |
| family | complex | 90.8% / 47s | **91.1% / 21s** | **91.3% / 22s** |
| family | rotate | 90.9% / 57s | **91.1% / 42s** | **91.1% / 40s** |
| wn18rr | complex | not yet re-run | not yet re-run | not yet re-run |
| wn18rr | rotate | not yet re-run | not yet re-run | not yet re-run |

All exhaustive-eval rows (ablation_d3, countries_s3) remain
**bit-identical** at the same seed. Family rotate **no longer OOMs**
— `eval_B` now comes from `KGEModel.recommended_eval_batch_size(num_entities)`
which knows each model's memory profile (RotatE: `B × E × half_dim`,
others: `B × E`). ComplEx gets the full `test_batch_size=4096` since
its matmul-style `[B, E]` output fits; RotatE auto-caps to `~1800`
on family and `~130` on wn18rr.

**ns is now faster than tkk standalone on family** (complex 21s vs
47s, rotate 42s vs 57s). The small-dataset rows (<10s budget) still
have ns a few seconds slower than tkk because the ns wrapper
(checkpoint callback, train_loop orchestration, val_cache build) has
a fixed ~2-3s overhead that dominates at those scales.

## Blockers

### #1 — Family rotate OOM  ✅ RESOLVED 2026-04-23

Two root causes, both fixed:

**(a) Per-call compile leak in `evaluate()`.** Creating a fresh
`torch.compile(...)` wrapper on every `evaluate()` invocation stacked
CUDA-graph pool retainers — 200 pools over a 100-epoch × 2-mode run.
**Fix**: per-model cache via the `_tkk_eval_state_cache` attribute;
compile runs once per `(model, B, K_fixed, modes, compile, device)`
combo, buffers are mark-static-address and stable across every
evaluate call for the model's lifetime. Cache dies with the model
(CPython cyclic GC cleans the closure ↔ model cycle).

**(b) `test_batch_size=4096` too large for RotatE's score_all_heads.**
The compile-graph B controls memory for the `[B, |E|, half_dim]`
broadcast inside RotatE; at B=4096 on family (|E|≈3k, half_dim=100)
that's 4.86 GB just for one intermediate. **Fix**: tkk KGE models now
expose `recommended_eval_batch_size(num_entities)` that returns a
safe B given the model's memory profile. RotatE overrides to account
for the extra half_dim factor. ns calls `model.recommended_eval_batch_size(...)`
and takes `min(test_batch_size, recommended)` for the compile-graph
B — outer-loop chunking inside `evaluate()` iterates the remaining
queries in Python (same pattern as DpRL's `PPOEvaluator` picking a
`fixed_batch_size`).

### #2 — Training static-buffer compiled step (#64)  ✅ RESOLVED 2026-04-23

`kge_kernels.training.epoch.train_epoch` now preallocates per-model
`pos_buf [B, 3]`, `neg_buf [B, K, 3]`, `mask_buf [B, K]`,
`pos_valid_buf [B]` via the `_tkk_train_state_cache` attribute.
`KGEModel.train_step` is static-shape (`cat([pos, neg_flat])` on
`[B + B*K, 3]` atoms, mask-aware BCE over both `pos_valid` and
`mask`) so the compiled step has no dynamic branches. Compiled with
`torch.compile(fullgraph=True, mode='default')` — `reduce-overhead`
tripped on a backward multi-use-tensor error for ComplEx-style
split-embedding models (real/imag participate in multiple backward
nodes, reduce-overhead's aggressive buffer reuse invalidates the
shared intermediate). Default mode still lowers the full
forward + backward via inductor. See task #69.

## To do (ordered)

1. **Run full 8-row sweep** to re-verify ablation_d3 / countries_s3
   bit-identicality and collect wn18rr numbers. Blockers #1+#2 done;
   family rotate now runs cleanly.
2. **Investigate reduce-overhead for train_step** (#69) — the
   backward multi-use-tensor error on ComplEx/RotatE. Would save ~20%
   per step vs default mode if solvable.
3. **Close KGE speed gap** (#61). ns family complex is ~1.5× slower
   than tkk standalone. Profile; suspect is `evaluate()`'s Python-
   level chunking loop over the smaller-B eval. Candidates: batch
   provider.candidates calls, precompute the eval pool once per val
   epoch instead of per batch, or move the ranking into the graph.
4. **Rewire tkk `train_model` to shared `train_epoch`** (#58). Low
   urgency — correctness is already bit-identical; this is dedup.
5. **Static-buffer reasoner eval internals** (#65). ReasonerModel
   eval currently allocates atom buffers per chunk; preallocate once.
6. **Save sweep logs to `output/runs/`** (#66) — wrap
   `run_3way_sweep.py` with `RunContext.stdout_capture`. Already being
   manually tee'd into dated directories; formalize via RunContext.
7. **Speed-test baseline fix** (#62). `profile_train.py` currently
   fails on the SBR+fb15k237 slow reasoner val eval; either regenerate
   or restrict profile to training step.
8. **Full regression suites** (#53). torch-ns precommit + tkk pytest
   + DpRL pytest. Blocked by 1.

## How to continue

### Reproduce the OOM
```bash
cd /home/castellanoontiv/repos/torch-kge-kernels-swarm/tkk-consolidation
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tkk-refactor
python /tmp/debug_rotate_oom.py 2>&1 | grep -E "mem|OOM|Tried to allocate"
```
(That debug script sits at `/tmp/debug_rotate_oom.py`; if it's gone,
recreate from the in-repo equivalent `scripts/run_3way_sweep.py --only
family`.)

### Where the leak is
`src/kge_kernels/eval/unified.py::evaluate` around line ~310:
```python
def _mk_scorer(mode: Mode):
    def _score(q, cand):
        return model.eval_scores(q, cand, mode)
    if compile:
        return torch.compile(_score, fullgraph=True, mode="reduce-overhead")
    return _score
score_fns = {m: _mk_scorer(m) for m in modes}
```
These closures + compile wrappers are re-made every call. Compile cache
retains their CUDA-graph pools.

### Check that #60 cleanup works cross-run (already in place)
`scripts/run_3way_sweep.py` around line ~175 clears
`torch._dynamo.reset()`,
`torch._inductor.cudagraph_trees.reset_cudagraph_trees()`,
`torch.cuda.empty_cache()` between sub-runs. Test:
```bash
python -u scripts/run_3way_sweep.py --only family \
    2>&1 | tee output/runs/<date>-oom/family.log
```
Expect: family complex ns completes (works now). Family rotate ns should
work too after #1 is fixed.

### Quick verification command after the compile fix lands
```bash
python -u scripts/run_3way_sweep.py --only family \
    2>&1 | tee output/runs/<date>-post-compile-fix/family.log
grep "MRR=" output/runs/<date>-post-compile-fix/family.log
```
Expected: 6 MRR lines, no OOM. Within-pair test MRRs: tkk complex 90.8 ≈
ns 91.2 (±0.4pp sampled-val noise); tkk rotate 90.9 ≈ ns ???.

### Full sweep when blockers resolved
```bash
python -u scripts/run_3way_sweep.py 2>&1 \
    | tee output/runs/<date>-phase1-verify/sweep.log
```
Acceptance: 8 rows × 3 paths = 24 runs, no OOM, ablation/countries
bit-identical, family/wn18rr ≤ 0.5pp sampled-val noise, ns timing ≤
tkk timing or comparable.
