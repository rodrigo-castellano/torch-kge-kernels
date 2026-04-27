# Evaluation contract — architecture map

This document explains how knowledge-graph-completion evaluation is wired across
the three layers tkk owns (`framework/` primitives, `search/` Searchers,
`eval/` ranking evaluator) and how external callers (torch-ns, DpRL-KGR) plug
in. It starts at the highest level and progressively zooms in.

---

## 1. The high-level picture

Evaluation answers one question: **for each test triple `(h, r, t)`, how good
is the model at putting the true entity at the top of the candidate list?**

Concretely:

- For each test triple, build two ranking problems: head corruption
  (`(?, r, t)`) and tail corruption (`(h, r, ?)`).
- For each problem, score the true entity + a pool of corruption candidates.
- Filter out known-true triples from the corruptions (so they can't
  out-rank the answer).
- Compute the rank of the true entity → aggregate to MRR and Hits@k.

```
   test triples [N, 3]
         │
         ▼
   ┌─────────────────────┐
   │  score candidates   │ ◀── method-specific
   │  (KGE / SBR / PPO   │     (the only thing that varies)
   │   rollout / ...)    │
   └─────────────────────┘
         │
         ▼
   ┌─────────────────────┐
   │  rank true entity   │ ◀── shared across all methods
   │  (filter known,     │     (tkk owns this code)
   │   compute_ranks)    │
   └─────────────────────┘
         │
         ▼
   MRR  +  Hits@1 / Hits@3 / Hits@10
```

Everything that varies between methods is funneled into a single function,
`ScoreFn`. Everything else — candidate generation, filtering, rank
computation, metric aggregation — is shared. tkk's job is to keep it that way.

---

## 2. The contract surfaces

Five things plug together. Everything else is an implementation detail of one
of these five.

### `Mode` (`eval/candidates.py`)

```python
Mode = Literal["head", "tail"]
```

A Python string. Frozen at compile time per scorer instance.

### `CandidateSource` (`eval/candidates.py`)

```python
class CandidateSource(Protocol):
    K_fixed: int

    def candidates(self, queries: Tensor, mode: Mode) \
        -> tuple[Tensor, Tensor]:
        # returns (cand_ents [B, K_fixed], valid [B, K_fixed])
        ...
```

How candidates are picked. `SamplerCandidates` is the standard impl; it
filters known-true triples at sampling time and applies per-relation domain
restrictions.

### `ScoreFn` (`eval/ranking_evaluator.py`)

```python
ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]
#                  q_buf [B, 3]  pool_buf [B, P]  mode  →  scores [B, P]
#                                       │                         │
#                                       │                         └── one score per slot
#                                       └── P = K_fixed + 1
#                                           slot 0   = true entity (lives at idx 0 by construction)
#                                           slot 1.. = K_fixed sampled candidates
```

The narrow inner contract. `q_buf` rows are `(relation, head, tail)`.
The pool always carries the true entity in slot 0 plus `K_fixed` corruption
candidates in slots 1..P-1, so input and output both have length `P` (not
`P + 1` — the `+1` for the true entity is already baked into `P`). This
is the **only** thing that goes inside the compile boundary.

**Why both `q_buf` and `pool_buf`?** Corruption varies one column of the
triple, not all three. `pool_buf` holds only that varying entity per slot;
`q_buf` holds the two columns that stay fixed across all P slots for a
query. The scorer reconstructs full triples by substituting the corruption
column from `pool_buf` into `q_buf`:

```
mode = "tail":  triple = (q_buf[:, 1],  q_buf[:, 0],  pool_buf[:, k])
mode = "head":  triple = (pool_buf[:, k], q_buf[:, 0],  q_buf[:, 2])
```

This split is ~3× smaller than carrying full triples in the pool and makes
the mode's job semantic: `mode` literally names which column gets
replaced.

### `RankingEvaluator` (`eval/ranking_evaluator.py`)

```python
class RankingEvaluator:
    def __init__(self, scorer: ScoreFn, candidates: CandidateSource, *,
                  batch_size, modes=("head","tail"), device,
                  compile=True, compile_mode="reduce-overhead",
                  tie_handling="average", seed=0): ...

    @torch.no_grad()
    def evaluate(self, triples: Tensor, *, track_scores=False) \
        -> RankingResult: ...
```

Two-phase lifecycle:
- **Setup-once** (`__init__`): allocates static-address buffers, compiles the
  scorer once per mode (mode baked into the closure → specialized graph per
  mode, no runtime branch).
- **Run-many** (`evaluate`): chunked loop. CUDA graph captures on first
  scorer call; replays for subsequent ones.

### `RankingResult` (`eval/ranking_evaluator.py`)

```python
@dataclass(frozen=True)
class RankingResult:
    triples: Tensor                  # [N, 3]
    modes: Tuple[Mode, ...]          # M
    ranks: Tensor                    # [M, N] — 1-based
    valid: Tensor                    # [M, N] — False = padded/skipped
    scores: Optional[Tensor] = None  # [M, N, P] iff track_scores
    elapsed_s: float = 0.0

    def metrics(...)             -> Dict[str, float]
    def metrics_per_mode(...)    -> Dict[Mode, Dict[str, float]]
    def metrics_per_group(gids,) -> Dict[int, Dict[str, float]]
    def metrics_per_relation(...) -> Dict[int, Dict[str, float]]
```

Aggregations are **methods on the result**, computed on demand. The same
`RankingResult` answers many questions (overall, per-mode, per-relation,
per-depth) without re-running the model.

---

## 3. The Searcher path (`search/`)

For methods that don't naturally fit `(q_buf, pool_buf, mode) → [B, P]`,
tkk adds a second contract one level above:

```python
class Searcher(Protocol):
    def __call__(self, queries: Tensor) -> Dict[str, Tensor]:
        # queries [N, 3] → {mode_key: scores [N]}
        ...
```

Plus an adapter that bridges the two:

```python
def make_scorer_from_searcher(searcher: Searcher, mode_key: str) -> ScoreFn:
    # eager reshape: [B, 3] queries × [B, P] pool  ─▶  flat [P*B, 3] K-major
    # call searcher(flat) → result[mode_key] → reshape back to [B, P]
```

The K-major flat layout `[k*B + b]` is the convention DpRL's compiled CUDA-
graph rollout writes into. The adapter is eager — outside the compile
boundary — so the compiled scorer keeps its `(q_buf, pool_buf) → [B, P]`
shape.

Two paths produce a `ScoreFn`:

- **Direct**: caller writes `ScoreFn` themselves. ns's KGE-only baseline does
  this — `model.score(h, r, t)` over the substituted pool.
- **Via Searcher**: caller writes a `Searcher`, lets `make_scorer_from_searcher`
  do the reshape. DpRL's PPO and Lookahead use this.

The evaluator can't tell the two paths apart.

### What `mode_key` refers to

The `mode_key` argument of `make_scorer_from_searcher` is **NOT** the
head/tail corruption side — that's `Mode = Literal["head", "tail"]` and
varies per evaluator call. `mode_key` is the **scoring-formula name** that
the Searcher's returned dict is keyed by, and it's frozen at adapter
construction. Concretely:

```
Searcher.__call__(queries)
     ↓
{ "logprob":         scores [N],     ← scoring-formula names
  "proof_binary":    scores [N],
  "depth_weighted":  scores [N],
  ...                                   }
     ↓ make_scorer_from_searcher(searcher, mode_key="logprob")
ScoreFn(q_buf, pool_buf, mode="head" | "tail")  ← Mode picks corruption side
     ↓
[B, P] scores
```

Two orthogonal axes: `mode_key` (which formula) is config-time; `Mode`
(which corruption) is per-call.

---

## 4. Trajectory scoring (the formulas behind `mode_key`)

The Searcher Protocol allows each method to emit any number of named
scores per query. Concretely those scores are produced by a
**`QueryRepr`** — the framework's per-trajectory → scalar reducer.

For methods whose scoring depends on the trajectory's running statistics
(success bit, total log-prob, depth, value-head outputs, etc.), tkk
provides a single unified dispatcher:

```python
class TrajectoryScoreQueryRepr(nn.Module):
    """~36 trajectory-scoring formulas in one Protocol-conforming class."""

    def __init__(self, mode: str, *, alpha=5.0, beta=2.0,
                 fail_penalty=1000.0, partial_weight=0.5,
                 endf_penalty=200.0): ...

    def forward(self, traj_repr: Repr, evidence) -> Repr:
        # reads traj_repr.summaries[...]
        # returns Repr.scores [B]
```

This replaces the older split between proof-search modes
(`ProofScoreQueryRepr`) and rollout modes (`PolicyRolloutQueryRepr`) —
both were "scalar score from trajectory summaries"; the split was
implementation drift from each upstream search choosing which summaries
to track.

### How summaries flow

`Repr.summaries: Optional[Dict[str, Tensor]]` carries trajectory-level
named tensors. The upstream Searcher populates the subset it actually
tracks; the QueryRepr reads what its mode needs.

Standard summary keys (canonical names; missing → loud `KeyError`):

```
success                [B] bool    trajectory succeeded?
depths                 [B] long    final depth reached
cumulative_log         [B]         Σ log p(action_d) along trajectory
min_step_log           [B]         min_d log p(action_d)
final_step_log         [B]         log p(action_{D-1})
final_state_score      [B]         KGE state-score at terminal step
best_cumulative        [B]         max_d Σ_{d'≤d} log p(action_{d'})
best_prefix_avg        [B]         max_d (Σ_{d'≤d} log p) / d
best_ever_state_score  [B]         max state-score across all steps
sbr_body_min           [B]         min over body-atom SBR scores
endf                   [B] bool    rollout ended via END action?
p_end                  [B]         step-0 P(end action)
v_pos, v_neg           [B]         step-0 value heads
kge_embed              [B]         KGE-policy embedding similarity
cum_value              [B]         accumulated value-head signal
value_steps            [B] long    #steps the value signal accumulated
```

PPO's compiled rollout populates ~6 of these (success, depths,
cumulative_log, p_end, v_pos, v_neg, kge_embed); proof-search populates
~10 (success, depths, cumulative_log, min_step_log, best_cumulative,
sbr_body_min, final_step_log, final_state_score, best_ever_state_score,
best_prefix_avg). Each mode reads only the keys it needs.

### Mode catalogue

`ALL_TRAJECTORY_SCORE_MODES` enumerates the ~36 supported keys. Headline
groups:

- **Trajectory aggregations** (no proved/failed branch): `no_cliff`,
  `no_cliff_sqrt`, `no_cliff_log`, `best_avg`, `max_intermediate_nc`,
  `trajectory_range`, `raw_cumulative`, `geometric_mean`.
- **Binary success/fail formulas**: `proof_logprob`, `leaf_only`,
  `final_state`, `min_step`, `best_partial`, `sbr`, `max_intermediate`,
  `depth_weighted_quotient` (was `ProofScore.depth_weighted` =
  `cum_log / d`).
- **RL log-prob formulas**: `logprob`, `logprob_endf`, `proof_binary`,
  `proof_bonus`, `logprob_clipped`, `proof_rank`, `success_only`,
  `logprob_scaled_penalty`, `depth_weighted_bonus` (was
  `PolicyRollout.depth_weighted` = `lp + β·d`).
- **Step-0 policy signals**: `value_pos`, `end_prob`, `hybrid_end`,
  `hybrid_value`, `combined`, `value_diff`, `endf_kge_embed`.
- **KGE-embed hybrids**: `kge_embed`, `kge_embed_hybrid`,
  `logprob_kge_embed`.
- **RL value modes**: `rl_value`, `rl_value_mean`, `rl_value_neg`,
  `rl_value_mean_neg`.

The only legacy collision (`depth_weighted` had different math on each
side) is resolved by explicit suffix: `_quotient` for the proof-search
variant, `_bonus` for the rollout variant.

---

## 5. Static-buffer layout

Allocated once in `__init__`; addresses captured by the CUDA graph.
Throughout this section `P = K_fixed + 1` (the `+1` is the slot-0 true
entity; `K_fixed` is the sampler's per-query candidate count).

```
_q_buf          [B, 3]   int64    test triple rows (padded if last chunk)
_pool_buf       [B, P]   int64    col 0: true ent ; col 1..: K_fixed candidates
_valid_buf      [B, P]   bool     True for active+valid slots
_true_idx_const [B]      int64    all zeros (true entity always at slot 0)
```

Eager scratch (outside the compile boundary):

```
_pool_scratch   [B, P]   int64
_valid_scratch  [B, P]   bool
```

---

## 6. Filtering happens at two points

| Stage | What's filtered | Where |
|---|---|---|
| Sampling | Known-true triples (train ∪ val ∪ test) excluded from the corruption pool | `Sampler.corrupt(filter=True)` inside `SamplerCandidates.candidates` |
| Pool assembly | Sampler shortfalls / per-relation domain restrictions marked `valid=False` | `SamplerCandidates.candidates` post-pad |
| Scoring | nothing — every slot is scored, shape stays `[B, P]` | (compiled scorer, fixed shape) |
| Ranking | invalid slots set to `-inf` → can't out-rank the truth | `compute_ranks(scores, true_idx, valid_mask)` |

Every layer is shape-preserving. Validity flows as a parallel mask, never
as a tensor reshape — that's what keeps the compile boundary clean.

---

## 7. Per-repo extension points

| What you customize | What you reuse |
|---|---|
| The score function (raw KGE, fused SBR, PPO rollout) | `RankingEvaluator`, `RankingResult`, `compute_ranks` |
| The candidate source (sampled-K vs full vocab; domain restrictions) | the chunked loop, the static buffers |
| The result envelope (DpRL wraps `RankingResult` in its own `EvalResults`) | `evaluate`'s body — only the wrapper subclass overrides |

The chunked loop is **final**: override `evaluate` only to wrap the result in
a different envelope. To change scoring, pass a different `scorer=`. To
change candidate generation, pass a different `candidates=`.

| Repo | Scorer construction | Evaluator class |
|---|---|---|
| **torch-ns** | `experiments/model.py:KGEModel.eval_scores` is a `ScoreFn` directly | uses `RankingEvaluator` straight, called from `experiments/evaluator.py:run_unified_ranking_eval` |
| **DpRL-KGR** | `PolicyRolloutSearcher` (or `LookaheadSearcher` subclass) → `make_scorer_from_searcher(s, mode_key)` → `ScoreFn` | `ppo/evaluator.py:PPOEvaluator(RankingEvaluator)` |

DpRL's subclass is thin — it owns the searcher lifecycle (set/swap/configure
n_corruptions) but the chunked loop is inherited unmodified.

---

## 8. The three views

### View 1 — Layered architecture & ownership

Where each piece lives and which repo customizes what.

```
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  test_triples [N, 3]                                                         │
   │       │                                                                      │
   │       ▼                                                          (caller)    │
   │  ╔══════════════════════════════════╗   ns:    experiments/evaluator.py      │
   │  ║       ENTRY  POINT               ║   DpRL:  ppo/evaluator.py              │
   │  ║  evaluator.evaluate(triples)     ║─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│
   │  ╚══════════════════════════════════╝                                        │
   │                                                                              │
   │     ┌─────────────────────────────────────────────────┐  ◀── tkk-owned       │
   │     │           RankingEvaluator (FINAL)              │      eval/           │
   │     │    chunked loop · static buffers · graph cap.   │      ranking_evaluator│
   │     └─────────────────────────────────────────────────┘                      │
   │              │                  │                  │                         │
   │   ┌──────────▼──┐     ┌─────────▼──────┐    ┌──────▼──────┐                  │
   │   │CandidateSrc │     │   ScoreFn      │    │compute_ranks│                  │
   │   │  Protocol   │     │  Protocol      │    │ + metrics   │                  │
   │   └──────────┬──┘     └─────────┬──────┘    └──────┬──────┘                  │
   │              │                  │                  │                         │
   │     ┌────────▼────┐    ┌────────▼─────────────┐    │                         │
   │     │ Sampler-    │    │  • direct (ns: KGE)  │    │   tkk reuses everywhere │
   │     │ Candidates  │    │  • via Searcher      │    │                         │
   │     │ (tkk std)   │    │      ─ adapter ──┐   │    │                         │
   │     └─────────────┘    └──────────────────┼───┘    │                         │
   │                                           │        │                         │
   │                    ┌──────────────────────▼───┐    │                         │
   │                    │   make_scorer_from_      │    │                         │
   │                    │   searcher(s, mode_key)  │    │                         │
   │                    │   K-major flat reshape   │    │                         │
   │                    └──────────────┬───────────┘    │                         │
   │                                   │                │                         │
   │            ┌──────────────────────┼──────────┐     │                         │
   │            ▼                      ▼          ▼     │                         │
   │     ┌────────────┐       ┌────────────┐  ┌─────────────┐                     │
   │     │ Greedy/    │       │ Policy-    │  │ Lookahead-  │   ◀── method-       │
   │     │ Beam/      │       │ Rollout-   │  │ Searcher    │       specific      │
   │     │ Multi-     │       │ Searcher   │  │ (subclass)  │                     │
   │     │ Restart    │       │ (compiled  │  └─────────────┘                     │
   │     │ (tkk ref)  │       │  CUDA-graph│                                      │
   │     └────────────┘       │  rollout)  │   tkk:  search/*                     │
   │                          └────────────┘   DpRL: ppo/policy_rollout_searcher  │
   │                                                  lookahead/searcher          │
   └──────────────────────────────────────────────────────────────────────────────┘
```

### View 2 — Per-chunk dataflow + compile boundary

What runs eagerly vs inside the compiled CUDA graph.

```
                    ┌─ EAGER ──────────────┐  ┌─ COMPILE ─┐  ┌─ EAGER ──────────────┐
                    │                      │  │           │  │                      │
   triples ───────▶ │  q_buf.copy_(slice)  │  │           │  │                      │
   [N, 3]           │                      │  │           │  │                      │
                    │  candidates.candidates                │  │                      │
                    │   (q_slice, mode)    │  │           │  │                      │
                    │   ─▶ cand_ents [B,K] │  │           │  │                      │
                    │      cand_valid [B,K]│  │           │  │                      │
                    │                      │  │           │  │                      │
                    │  pool_scratch[:,0] = │  │           │  │                      │
                    │    q[:, true_col]    │  │           │  │                      │
                    │  pool_scratch[:,1:] =│  │           │  │                      │
                    │    cand_ents         │  │           │  │                      │
                    │  pool_buf.copy_(     │  │           │  │                      │
                    │    pool_scratch)     │  │           │  │                      │
                    │  valid_buf.copy_(...)│  │           │  │                      │
                    │                      │  │           │  │                      │
                    │  cudagraph_mark_     │  │           │  │                      │
                    │    step_begin()  ────┼─▶│           │  │                      │
                    │                      │  │ScoreFn    │  │                      │
                    │                      │  │(q_buf,    │  │                      │
                    │                      │  │ pool_buf, │  │                      │
                    │                      │  │ mode)     │  │                      │
                    │                      │  │  ─▶ [B,P] │  │                      │
                    │                      │  │           │  │                      │
                    │                      │  │ replays   │  │                      │
                    │                      │  │ static    │  │                      │
                    │                      │  │ buffers   │  │                      │
                    │                      │  └─────┬─────┘  │                      │
                    │                      │        ▼        │  compute_ranks(      │
                    │                      │  pool_scores    │    pool_scores,      │
                    │                      │                 │    true_idx=zeros,   │
                    │                      │                 │    valid_mask)       │
                    │                      │                 │   ─▶ chunk_ranks[B'] │
                    │                      │                 │                      │
                    │                      │                 │  ranks_out[m, qs:] = │
                    │                      │                 │  valid_out[m, qs:] = │
                    │                      │                 │                      │
                    └──────────────────────┘  └───────────┘  └──────────────────────┘

       Static-address tensors: q_buf, pool_buf, valid_buf, true_idx_const
       Loop: for chunk in N // B:  for mode in modes:  ScoreFn(...)
       One CUDA graph per mode   (mode is Python str, baked into closure)
```

### View 3 — Two paths to a `ScoreFn`

How the two contract surfaces (direct `ScoreFn` vs `Searcher` + adapter)
both end up satisfying the evaluator's interface.

```
                            RankingEvaluator expects:
                            ScoreFn(q_buf [B,3], pool_buf [B,P], mode) ─▶ [B, P]
                                          ▲
                          ┌───────────────┴───────────────┐
                          │                               │
              ┌───────────┴───────────┐       ┌───────────┴────────────────┐
              │   PATH A — direct     │       │   PATH B — via Searcher    │
              │                       │       │                            │
              │  caller writes the    │       │  caller writes a Searcher: │
              │  reshape themselves   │       │  (queries [N,3]) ─▶        │
              │                       │       │      {mode_key: [N]}       │
              │  e.g. ns KGEModel.    │       │                            │
              │  eval_scores:         │       │  ↓ eager reshape (adapter) │
              │    h, r, t = pool     │       │                            │
              │     into model.score  │       │  make_scorer_from_searcher │
              │     ─▶ [B, P]         │       │   • triples [P, B, 3]      │
              │                       │       │   • flat [P*B, 3]   K-major│
              │  no adapter needed    │       │   • searcher(flat) ─▶ [N]  │
              │                       │       │   • result.view(P,B).t()   │
              │                       │       │     ─▶ [B, P]              │
              └───────────────────────┘       └────────────────────────────┘
                          ▲                               ▲
              used by:    │                               │  used by:
              ns          │                               │  DpRL — PolicyRollout-
              (KGE-only   │                               │  Searcher (compiled
              baseline)   │                               │  CUDA-graph rollout) +
                          │                               │  LookaheadSearcher


      Either path is one ScoreFn function. The evaluator can't tell them apart.
      Filtering (drop known-true triples) happens INSIDE the candidate source —
      both paths see only sampler-filtered candidates in pool_buf.
```

---

## 9. Why the contract is shaped this way

- **One scoring call per (chunk, mode)** maximizes batch-level GPU work, gives
  CUDA graph capture a clean target, and lets mode-specialized graphs avoid
  runtime branching.
- **Static buffers + `mark_static_address`** enable
  `mode="reduce-overhead"` to actually replay graphs instead of retracing.
- **Aggregation as result methods, not eval-time work** lets the same
  `RankingResult` answer many questions (overall, per-mode, per-relation,
  per-depth) without re-running the model.
- **K-major flat pool as the Searcher convention** lets DpRL's compiled
  rollout treat the entire pool as one big batch (no per-query loop in the
  compiled region) while keeping the evaluator's `[B, P]` interface intact.
- **Filtering as a mask, not a reshape**: invalid slots stay in the buffer
  (so shape never changes), but contribute nothing to the rank — compile-friendly.
