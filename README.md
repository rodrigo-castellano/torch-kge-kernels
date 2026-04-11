# torch-kge-kernels

[![tests](https://github.com/rodrigo-castellano/torch-kge-kernels/actions/workflows/tests.yml/badge.svg)](https://github.com/rodrigo-castellano/torch-kge-kernels/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![torch](https://img.shields.io/badge/torch-%3E%3D2.0-ee4c2c)](https://pytorch.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Shared PyTorch building blocks for proof-based neural-symbolic KGE
methods.** `torch-kge-kernels` (*tkk*) is the single source of truth for
the six framework primitives from the `framework.tex` spec plus the raw
KGE models, losses, data utilities, ranking metrics, and evaluation
pipeline used by `torch-ns` (SBR / DCR / R2N), `DpRL-KGR-swarm` (DPrL),
and any future proof-based method.

## What's inside

```text
kge_kernels/
├── framework/        # Pluggable primitives from framework.tex §5-§7
│   ├── repr.py            Repr container (embeddings ∪ scores)
│   ├── protocols.py       AtomRepr / StateRepr / TrajRepr /
│   │                      QueryRepr / Select / ResolutionOp
│   ├── types.py           ProofState / ProofEvidence Protocols +
│   │                      SelectInfo dataclass
│   ├── atom_repr.py       KGEScoreAtom, KGEEmbedAtom, KGEBothAtom, MLPAtom
│   ├── state_repr.py      TNorm / Sum / Mean / Max / Concat state aggregators
│   ├── traj_repr.py       TNorm / CumulativeLog / MinStep / BestCumulative /
│   │                      PolicyProduct / SBRBodyMin trajectory aggregators
│   ├── query_repr.py      Max / Sum / Mean / LogSumExp / MLPSum / ConceptMax
│   ├── select.py          Exhaustive / Greedy / Beam / Sample
│   └── scorer.py          Reference search_and_score + build_scorer
│
├── models/           # Raw KGE nn.Module classes on a common KGEModel base
│   ├── base.py            KGEModel ABC (score_triples / all_tails / all_heads /
│   │                      compose / score dispatch)
│   ├── transe.py          TransE
│   ├── distmult.py        DistMult
│   ├── complex.py         ComplEx
│   ├── rotate.py          RotatE
│   ├── tucker.py          TuckER
│   ├── mode.py            ModE
│   └── conve.py           ConvE
│
├── losses/           # KGE + neural-symbolic losses with a build_loss factory
│   ├── classification.py  BCE-with-mask, weighted BCE, BCE-ragged, CE-ragged
│   └── ranking_losses.py  PairwiseCE, Hinge, L2
│
├── data/             # Triple-file loading + filter-map construction
│   ├── loaders.py         TripleExample, load_triples (TSV/CSV/Prolog),
│   │                      load_triples_with_mappings, encode_split_triples
│   └── transforms.py      add_reciprocal_triples, build_filter_maps,
│                          build_relation_domains
│
├── eval/             # Candidate pool construction + filtered ranking evaluation
│   ├── pool.py            CandidatePool
│   ├── evaluator.py       Evaluator (subclass to plug in a custom _score_pool)
│   ├── fusion.py          rrf, zscore_fusion
│   └── results.py         EvalResults dataclass
│
├── logging/          # Experiment spec / run context / structured run_experiment
│
├── ranking.py        # ranks_from_scores, ranks_from_scores_matrix,
│                     # ranks_from_labeled_predictions, ranking_metrics(ks=...),
│                     # StreamingRankingMetrics (in-place GPU accumulator)
├── sampler.py        # Sampler.corrupt_with_mask, SamplerConfig
├── scoring.py        # Low-level score() over an explicit KGEBackend
├── adapter.py        # build_backend, kge_score_*, apply_masks
├── partial.py        # LazyPartialScorer, score_partial_atoms
└── types.py          # KGEBackend, CorruptionOutput, ScoreOutput
```

## Install

```bash
pip install "torch-kge-kernels @ git+https://github.com/rodrigo-castellano/torch-kge-kernels.git@main"
```

`torch>=2.0` and Python 3.10+. CPU-only is sufficient for the tkk test
suite; consumer repos use CUDA at runtime.

## Quickstart: compose a scorer

```python
from kge_kernels.framework import (
    ExhaustiveSelect,
    KGEScoreAtom,
    MaxQueryRepr,
    TNormStateRepr,
    TNormTrajRepr,
    build_scorer,
)
from kge_kernels.models import TransE

# Any grounder producing ProofEvidence works; this is a stub
def my_grounder(state):
    ...

model = TransE(num_entities=14505, num_relations=237, dim=256)

scorer = build_scorer(
    resolve=my_grounder,
    atom_repr=KGEScoreAtom(),
    state_repr=TNormStateRepr("min"),      # Gödel conjunction
    traj_repr=TNormTrajRepr("min"),        # min-over-depths
    query_repr=MaxQueryRepr(),             # max-over-proofs (SBR)
    select=ExhaustiveSelect(),             # one-shot scoring
    model=model,
    max_depth=1,
)

scores = scorer(queries)  # {"default": [B]}
```

Switching from SBR-style exhaustive scoring to DPrL-style sequential
sampling is a matter of swapping `ExhaustiveSelect()` for
`BeamSelect(k=5)` or `SampleSelect(n=1)` and raising `max_depth`.

## Quickstart: Evaluator subclass

```python
from kge_kernels.eval import Evaluator

class MyEvaluator(Evaluator):
    def _score_pool(self, pool):
        # Custom scoring (e.g. compiled CUDA-graph closure) — everything
        # else (pool building, ranking, filter masking, metrics) is
        # inherited from the base Evaluator.
        return {"my_mode": self._my_compiled_scorer(pool)}

results = MyEvaluator(
    scorer=lambda p: {},  # overridden
    sampler=my_sampler,
    n_corruptions=50,
    corruption_scheme="both",
).evaluate(test_queries)
# results.metrics == {"my_mode_MRR": ..., "my_mode_Hits@1": ..., ...}
```

## Architectural contract

Each downstream method ships its **own** `search_and_score`-equivalent
function. Consumers use tkk primitives wherever the math is the same as
the reference implementation, but are free to inline, fuse, or
specialize them when performance constraints require:

- **torch-ns SBR / DCR / R2N** use fused, `torch.compile`-safe reasoning
  layers in `ns_lib/nn/reasoning.py` that reproduce framework primitives
  inline.
- **DpRL PPO evaluation** uses a compiled CUDA-graph closure
  (`PPOEvaluator` subclassing `kge_kernels.eval.Evaluator`) — pool
  construction, ranking, and metrics come from tkk; only `_score_pool`
  is overridden.
- **DpRL atom / state embedders** use a different pre-embedded tensor
  interface (`(p_embeddings, c_embeddings)` tuples with slot-based
  variable embeddings) and stay local.

The framework's guarantee is **shared primitives**, not a shared control
loop. See `framework.tex` §1.3 "Architectural contract" for the full
rationale.

## Not implemented

The framework.tex spec describes two methods that are **not yet**
shipped in tkk, both explicitly out of scope for the 2026-04
consolidation:

- **MCTS** (`framework.tex` §12). The Protocols in `framework/` are
  already designed to accommodate a future `UCBSelect` / `MCTSTrajRepr`
  without breaking the reference loop — MCTS will live in its own
  driver file when implemented.
- **Q-Guided Search** (`framework.tex` §13). The $Q(s,a) = H(s,a) +
  V(T(s,a))$ decomposition maps to an alternate `Select` subclass; the
  existing `SelectInfo` dataclass is intentionally extensible to carry
  Q-values and per-rule bias.

## Development

```bash
git clone https://github.com/rodrigo-castellano/torch-kge-kernels.git
cd torch-kge-kernels
pip install -e ".[dev]"
pytest -q
```

Test suite is CPU-only, runs in <2 seconds, and must stay green on every
commit. Consumer-side regression (speed + MRR) gates live in the
downstream repos (`torch-ns-swarm`, `DpRL-KGR-swarm`).

## Cited by

- R. Castellano Ontiveros, F. Giannini, M. Gori, G. Marra, M. Diligenti.
  *Grounding Methods for Neural-Symbolic AI.* IJCAI, 2025.
- Y. Jiao, R. Castellano Ontiveros, L. De Raedt, M. Gori, F. Giannini,
  M. Diligenti, G. Marra. *DeepProofLog: Efficient Proving in Deep
  Stochastic Logic Programs.* AAAI, 2026.

## License

MIT — see [`LICENSE`](LICENSE).
