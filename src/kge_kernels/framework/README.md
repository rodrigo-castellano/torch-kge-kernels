# `framework/` — the proof-search abstraction

> Companion to [`framework.pdf`](https://github.com/) (canonical spec, §1–§13).
> This README is the implementation-level orientation: where each piece lives,
> what shapes it consumes/emits, and how the layers compose.

---

## 1. The big idea

Every proof-based NeSy method is a search over proof space, decomposed into
**six pluggable functions** that compose in **one canonical loop**:

```
Method = (resolve, atom_repr, state_repr, select, traj_repr, query_repr)
```

```
              ┌──────────────────────── one loop ────────────────────────┐
              ▼                                                          │
state ─▶ resolve ─▶ atom_repr ─▶ state_repr ─▶ select ─▶ traj_repr ─────┤
                                                            │            │
                                                            ▼            │
                                                     accum (Repr) ───────┘
                                                            │
                                                            ▼
                                                       query_repr ─▶ score [B]
```

Each method is a **row in a table**: SBR is `(Enum, KGEScore, TNorm(min),
Exhaustive, TNorm(min), Max)`, DPrL is `(SLD, MLP, Sum, Beam/Sample,
PolicyProduct, Sum)`. New methods are new rows; ablations swap one cell.

The **exhaustive/sequential** split is just whether `select = identity`
(SBR/DCR/R2N: one iteration) or not (DPrL: D iterations).

---

## 2. Dim convention

Aligned with `framework.pdf` §3.2–§3.3.

| Dim | Meaning |
|---|---|
| `B` | batch (queries) |
| `P` | max proofs per query (collected groundings) |
| `D` | max depth (rule applications per proof) |
| `M` | max body atoms per single rule application |
| `G` | max parallel goals per query in live search (search width / branches) |
| `A` | max atoms per goal (open atom stack length) |
| `E` | embedding dim |
| `N` | flat per-atom count (`atom_repr` inputs) |
| `K` | corruptions per query (negatives in ranking) |
| `C` | candidates per query = K + 1 (ranking pool size) |

A "**goal**" is a conjunction of atoms (a search-node state, classical-AI
sense). An "**atom**" is one triple. The trailing `3` in `[..., 3]` is
always `(predicate, subject, object)`.

Relationship: `A_max = 1 + D·(M − 1)` (each rule application replaces
one atom with M body atoms).

---

## 3. Core data structures

### `Repr` (universal carrier)

Flows between all four representation levels. Holds optional embeddings,
optional scores, optional summaries:

```python
@dataclass
class Repr:
    embeddings: Optional[Tensor]                       # [*, E]
    scores: Optional[Tensor]                           # [*]
    summaries: Optional[Dict[str, Tensor]] = None      # named per-traj signals
```

Each level's signature is `Repr → Repr`. What's populated depends on the
method (SBR: scores only; DPrL/R2N: embeddings; DCR: both).

### `ProofState` (live search state)

```python
@dataclass
class ProofState:
    proof_goals: Tensor      # [B, G, A, 3] — open goals per branch
    state_valid: Tensor      # [B, G]
    top_ridx: Tensor         # [B, G]
```

Mutable across resolution steps (sequential search reads/writes this).

### `ProofEvidence` (accumulated proof trace)

```python
@dataclass
class ProofEvidence:
    body: Tensor             # [B, P, D, M, 3] — body atoms per (proof, depth, pos)
    proof_mask: Tensor       # [B, P]            — valid proofs
    depth_mask: Tensor       # [B, P, D]         — active depths per proof
    body_mask: Tensor        # [B, P, D, M]      — valid atoms per depth
    rule_idx: Tensor         # [B, P, D]         — which rule at each depth
```

Output of `resolve`. For exhaustive scorers (SBR/DCR/R2N) D = 1; for
sequential proofs D > 1. Carries only structural data — scores and
embeddings live in `Repr`, computed by the representation pipeline.

---

## 4. The six primitives

| Slot | File | Protocol signature | Output shape |
|---|---|---|---|
| `resolve` | (grounder repos) | `(state) → ProofEvidence` | structural |
| `atom_repr` | `repr_atom.py` | `(preds [N], subjs [N], objs [N], model) → Repr` | `[N, ...]` |
| `state_repr` | `repr_state.py` | `(atom_repr: Repr, evidence) → Repr` | `[B, P, D, ...]` |
| `select` | `select.py` | `(evidence, s_repr) → (next_state, info)` | varies |
| `traj_repr` | `repr_traj.py` | `(state_repr: Repr, evidence) → Repr` | `[B, P, ...]` |
| `query_repr` | `repr_query.py` | `(traj_repr: Repr, evidence) → Repr` | `[B]` |

### `resolve` — produce successor goals

Two implementations, both satisfy `ResolutionOp` Protocol:

- **Enum** (`grounder.bc.grounder`) — backward-chaining with bounded width
  and depth. Produces all groundings in batch. Used by SBR/DCR/R2N.
- **SLD** (`DpRL.env.RLGrounder`) — single-step MGU-based resolution.
  Returns one step's successors. Used by DPrL.

### `atom_repr` — atom → Repr

Maps `(pred, subj, obj)` triples to per-atom representations:

| Impl | Output | Used by |
|---|---|---|
| `KGEScoreAtom` | scores | SBR |
| `KGEEmbedAtom` | embeddings | R2N |
| `KGEBothAtom` | both | DCR |
| `MLPAtom` | embeddings | DPrL |
| `RemappedKGEScoreAtom` | scores (with KGE-vocab remap) | DpRL hybrid |

### `state_repr` — atoms → state-level Repr

Aggregates per-atom representations within a goal:

| Impl | Output | Notes |
|---|---|---|
| `TNormStateRepr` | scores `[B, P, D]` | min / product (Gödel / product t-norm) |
| `SumStateRepr` | embs `[B, P, D, E]` | sum of atom embeddings |
| `MeanStateRepr` | embs `[B, P, D, E]` | mean |
| `ConcatStateRepr` | embs `[B, P, D, M*E]` | flattens body |
| `PhiPsiStateRepr` | scores `[B, P, D]` | DCR-specific (Φ/Ψ then t-norm) |

### `select` — choose next state

Returns `(next_state, info)` where `info` carries per-step signals (chosen
scores, log-probs, ...):

| Impl | Description | Used by |
|---|---|---|
| `ExhaustiveSelect` | identity, terminate after 1 iter | SBR/DCR/R2N |
| `GreedySelect` | argmax over s_repr | baseline |
| `BeamSelect(k)` | top-k | DPrL eval |
| `SampleSelect(n)` | categorical sampling | DPrL train (PPO) |
| `PolicySelect` (DpRL) | policy network argmax + Gumbel | DpRL PPO eval |

`Greedy/Beam/Sample/Policy` accept an optional `gumbel_scale_buf` for
Gumbel-max stochasticity (see §6 mid-life mutation).

### `traj_repr` — depth aggregation

Two callable interfaces (must agree):
- **Batch**: `forward(s_repr, evidence) → Repr` — exhaustive path, reduces
  over D in one call.
- **Incremental**: `init(B, device) + step(accum, s_repr, info)` —
  sequential path, accumulates one depth at a time.

| Impl | Output | Description |
|---|---|---|
| `TNormTrajRepr` | scores `[B, P]` | conjunction across depths (min/prod) |
| `RuleMLPTrajRepr` | embs `[B, P, E]` | per-rule MLP per depth, then conj |
| `CumulativeLogTrajRepr` | scores `[B, P]` | Σ log s(state) |
| `PolicyProductTrajRepr` | scores `[B, P]` | Σ log π(a|s) for PG |
| `MultiTrajRepr` | dict of named summaries | populates `Repr.summaries` |

### `query_repr` — proofs → query score

Reduces per-trajectory to per-query scalar:

| Impl | Output | Formula |
|---|---|---|
| `MaxQueryRepr` | scores `[B]` | `max_p s(d_p)` |
| `SumQueryRepr` | scores `[B]` | `Σ_p s(d_p)` |
| `MeanQueryRepr` | scores `[B]` | `mean_p s(d_p)` |
| `MLPSumQueryRepr` | scores `[B]` | `MLP(Σ_p e(d_p))` |
| `LogSumExpQueryRepr` | scores `[B]` | `log Σ_p exp(s(d_p))` |
| `TrajectoryScoreQueryRepr` | scores `[B]` | ~36-mode dispatcher reading `Repr.summaries` |

`TrajectoryScoreQueryRepr` is the unified scoring-formula dispatcher used
by RL methods (DPrL). Modes: `logprob`, `proof_binary`, `depth_weighted_*`,
`value_pos`, `end_prob`, `kge_embed*`, etc. — see
`repr_query.py:ALL_TRAJECTORY_SCORE_MODES`.

---

## 5. The canonical scoring loop

The body of `ProofScorer.search_and_score` (private helper
`search/searcher.py:_canonical_loop`) composes the six primitives:

```python
def _canonical_loop(query, *, resolve, atom_repr, state_repr, select,
                    traj_repr, query_repr, model, max_depth) -> Tensor:
    state = init(query)
    accum = traj_repr.init(B)
    for d in range(max_depth):
        evidence = resolve(state)
        a_repr = atom_repr(*atoms_from(evidence), model)
        s_repr = state_repr(a_repr, evidence)
        state, info = select(evidence, s_repr)
        accum = traj_repr.step(accum, s_repr, info)
        if state.done: break
    return query_repr(accum, evidence).scores                # [B]
```

- **Exhaustive** (`max_depth=1`, `ExhaustiveSelect`): `resolve` returns
  the full evidence in one call; `traj_repr.forward` reduces over D in
  batch; `query_repr` produces the final score. One iteration.
- **Sequential** (`max_depth>1`, non-exhaustive `select`): `traj_repr`
  accumulates per-step via `init + step×D`; `select` prunes at each step.

There is no separate `score_query` — the loop IS the scorer.

---

## 6. `ProofScorer` — the concrete class

`search/searcher.py:ProofScorer` is the canonical implementation of
the `Searcher` Protocol. One class for every 6-tuple composition.

```python
scorer = ProofScorer(
    resolve=...,            # ResolutionOp
    atom_repr=...,          # AtomRepr
    state_repr=...,         # StateRepr
    select=...,             # Select
    traj_repr=...,          # TrajRepr
    query_repr=...,         # QueryRepr
    spec=SearchSpec(batch_size=B, max_depth=D, ...),
    capture="static" | "dynamic",
)
scores = scorer(queries)                   # {mode_key: [N]}
```

### Capture modes

- **`capture="static"`** (default) — `_compile()` wraps the canonical
  loop with `torch.compile(mode="reduce-overhead", fullgraph=True)` for
  cudagraph cheap-replay.
- **`capture="dynamic"`** — eager `_canonical_loop`, no compilation,
  shape-flexible. For dev / debug / shape-variable use.

Both modes have identical `__call__(queries) → Dict[str, Tensor]` output.

### Compilation contract — for subclassing

`ProofScorer.__init__` calls two lifecycle hooks subclasses override to
plug in specialized rollouts (e.g., DpRL's `PPOProofScorer` /
`LookaheadProofScorer` running an alternated-buffer compiled CUDA-graph
rollout):

- `_allocate_buffers()` — allocate persistent tensors owned by this
  scorer. Base: no-op (primitives own theirs).
- `_compile()` — build compiled bodies / closures. Base:
  `torch.compile` the canonical 6-tuple loop. Subclasses that override
  `search_and_score` entirely may make this a no-op.

Subclasses that fully override `search_and_score` may pass `None` for
the framework primitives the parent would compose.

### Mid-life mutation lives on the primitives

```python
scorer.select.set_gumbel_scale(0.3)         # GreedySelect / BeamSelect / PolicySelect
scorer.resolve.configure(n_corruptions=5)   # PPOProofScorer
```

`ProofScorer.set_gumbel_scale` / `configure` / `reset_stats` /
`aggregate_stats` are convenience forwarders.

### Strategy factory

`search/__init__.py:make_searcher(strategy=...)` builds a `ProofScorer`
with the right `Select` for `"exhaustive"` / `"greedy"` / `"beam"` /
`"multi_restart"` / `"direct"`. `MultiRolloutSearcher` and
`MultiRestartSearcher` are higher-order wrappers (K rollouts × Gumbel
noise; per-query max).

---

## 7. Evaluation pipeline (one consumer of `Searcher`)

`eval/ranking_evaluator.py:RankingEvaluator` is the shared chunked
loop + static-buffer + CUDA-graph evaluator. It consumes a `ScoreFn`:

```python
ScoreFn = Callable[[Tensor, Tensor, Mode], Tensor]
#                  q_buf [B, 3]  pool_buf [B, C]  mode  →  scores [B, C]
```

`q_buf` rows are `(relation, head, tail)`; `pool_buf` carries the true
entity in slot 0 + K corruption candidates in slots 1..C-1 (so
`C = K + 1`). `mode ∈ {"head", "tail"}` selects which column gets
substituted from `pool_buf`. The pool is pre-filtered (known-true
triples excluded by the sampler).

Two paths produce a `ScoreFn`:

- **Direct** — caller writes `ScoreFn` themselves (ns's KGE-only baseline:
  `model.score(h, r, t)` over the substituted pool).
- **Via Searcher** — `make_scorer_from_searcher(searcher, mode_key)`
  reshapes the pool to K-major flat `[C*B, 3]`, calls
  `searcher(flat) → {mode_key: [N]}`, reshapes back to `[B, C]`. The
  reshape is eager — outside the compile boundary.

```
test triples [N, 3]
        │
        ▼
   ┌─────────────────────────────────────────────────────┐
   │  RankingEvaluator (chunked loop, static buffers)    │
   │                                                     │
   │  per chunk:                                         │
   │    sampler  → (cand_ents [B,K], cand_valid [B,K])   │
   │    pool_buf[:, 0] = q[:, true_col]                  │
   │    pool_buf[:, 1:] = cand_ents                      │
   │    cudagraph_mark_step_begin()                      │
   │    pool_scores = ScoreFn(q_buf, pool_buf, mode)     │  ◀── compile boundary
   │    chunk_ranks = compute_ranks(pool_scores, ...)    │
   └─────────────────────────────────────────────────────┘
        │
        ▼
   RankingResult → MRR, Hits@1/3/10
```

### Filtering at two points

| Stage | What's filtered | Where |
|---|---|---|
| Sampling | Known-true triples (train ∪ val ∪ test) | `Sampler.corrupt(filter=True)` inside `SamplerCandidates.candidates` |
| Pool assembly | Sampler shortfalls / domain restrictions → `valid=False` | `SamplerCandidates.candidates` post-pad |
| Scoring | (none — every slot scored, shape stays `[B, C]`) | (compiled) |
| Ranking | invalid slots → `-inf` → can't out-rank truth | `compute_ranks(...)` |

Validity flows as a parallel mask, never as a reshape — that's what keeps
the compile boundary clean.

---

## 8. Method instantiations (the 6-tuples)

From `framework.pdf` §9.1:

| Method | resolve | atom_repr | state_repr | select | traj_repr | query_repr |
|---|---|---|---|---|---|---|
| **SBR** | Enum | KGEScore | TNorm(min) | Exhaustive | TNorm(min) | Max |
| **DCR** | Enum | KGEBoth | PhiPsi | Exhaustive | TNorm(min) | Max |
| **R2N** | Enum | KGEEmbed | Concat | Exhaustive | RuleMLP | MLPSum |
| **DPrL** | SLD | MLP | Sum | Beam/Sample | PolicyProduct | Sum |
| **SBR-Greedy** | Enum | KGEScore | TNorm | Greedy | TNorm | Max |
| **SBR-Beam(k)** | Enum | KGEScore | TNorm | Beam(k) | TNorm | Max |

These are exercised end-to-end in `tests/search/test_method_tuples.py`.

---

## 9. Per-repo extension points

| Repo | Builds the searcher via | Plugs into eval via |
|---|---|---|
| **torch-ns** | direct grounder + framework primitives (SBR/DCR/R2N) | `experiments/model.py:KGEModel.eval_scores` (a `ScoreFn` directly) |
| **DpRL-KGR (PPO)** | `PPOEvaluator.searcher` → `PPOProofScorer` (rollout, default) or `KGEFusionScorer` (rollout + external-KGE bridge, when `kge_inference=True`) or `DirectSearcher` (`reasoning_mode="direct_kge"`) | `make_scorer_from_searcher(s, mode_key)` |
| **DpRL-KGR (Lookahead)** | `LookaheadEvaluator.searcher` → `LookaheadProofScorer(ppo, scoring_method=...)` (subclasses `PPOProofScorer`; KGE-guided rollout) | same |

Common pattern:
- Customize `ScoreFn` (raw KGE / fused SBR / PPO rollout) — reuse
  `RankingEvaluator`, `RankingResult`, `compute_ranks`.
- Customize `CandidateSource` (sampled-K vs full-vocab; domain restrictions)
  — reuse the chunked loop + static buffers.
- Customize the result envelope (DpRL wraps `RankingResult` in
  `EvalResults`) — only override the wrapping.

---

## 10. Why the contract is shaped this way

- **One scoring call per (chunk, mode)** maximizes batch-level GPU work,
  gives CUDA-graph capture a clean target, and lets mode-specialized
  graphs avoid runtime branching.
- **Static buffers + `mark_static_address`** enable
  `mode="reduce-overhead"` to actually replay graphs instead of retracing.
- **Aggregations as result methods, not eval-time work** lets the same
  `RankingResult` answer many questions (overall, per-mode, per-relation,
  per-depth) without re-running the model.
- **K-major flat pool as the Searcher convention** lets DpRL's compiled
  rollout treat the entire pool as one big batch (no per-query loop in
  the compiled region) while keeping the evaluator's `[B, C]` interface.
- **Filtering as a mask, not a reshape**: invalid slots stay in the
  buffer (shape never changes) but contribute nothing to the rank.
- **Mid-life mutation on primitives**: `set_gumbel_scale` writes into a
  static-address scalar buffer, so the next replay sees the new value
  without re-tracing the graph.

---

## 11. Pointers

- `framework.pdf` — canonical spec (§1–§13). This README is the
  implementation orientation; the PDF is the math + theory.
- `tests/framework/` — primitive coverage (one test file per slot).
- `tests/search/test_method_tuples.py` — end-to-end exercises of the
  6-tuples in §8.
- `tests/search/test_proof_scorer.py` — `ProofScorer` capture-mode
  + contract-hook + setter-propagation coverage.
- `repr.py` — the `Repr` carrier docstring (the universal flow object).
