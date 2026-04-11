# Changelog

All notable changes to `torch-kge-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to semantic versioning starting from 0.3.0.

## [0.3.0] — 2026-04-11

First release with the full framework primitive set, KGE model classes,
losses, dataset utilities, and a tidied public API. The package has
roughly 3× the public surface of 0.2.0 and now backs both `torch-ns`
(SBR / DCR / R2N) and `DpRL-KGR-swarm` (DPrL) as their single source of
truth for shared KGE infrastructure.

### Added

- **`kge_kernels.framework`** — pluggable primitives from `framework.tex` §5–§7:
  - `Repr` container, `ProofState` / `ProofEvidence` Protocols, `SelectInfo`
  - Protocol signatures for `AtomRepr`, `StateRepr`, `TrajRepr`,
    `QueryRepr`, `Select`, `ResolutionOp`
  - Atom reprs: `KGEScoreAtom`, `KGEEmbedAtom`, `KGEBothAtom`, `MLPAtom`
  - State reprs: `TNormStateRepr`, `SumStateRepr`, `MeanStateRepr`,
    `MaxStateRepr`, `ConcatStateRepr`
  - Traj reprs: `TNormTrajRepr`, `CumulativeLogTrajRepr`,
    `MinStepTrajRepr`, `BestCumulativeTrajRepr`, `PolicyProductTrajRepr`,
    `SBRBodyMinTrajRepr`
  - Query reprs: `MaxQueryRepr`, `SumQueryRepr`, `MeanQueryRepr`,
    `LogSumExpQueryRepr`, `MLPSumQueryRepr`, `ConceptMaxQueryRepr`
  - Selects: `ExhaustiveSelect`, `GreedySelect`, `BeamSelect`, `SampleSelect`
  - Reference `search_and_score` and `build_scorer` factory
- **`kge_kernels.models`** — raw KGE `nn.Module` classes on a common
  `KGEModel` base with `score(h, r, t=None)` dispatch:
  - `TransE`, `DistMult`, `ComplEx`, `RotatE`, `TuckER`, `ModE`, `ConvE`
- **`kge_kernels.losses`** — pure-tensor losses + `build_loss(name)` factory:
  - `BinaryCrossEntropyWithMask`, `WeightedBinaryCrossEntropy`,
    `BinaryCrossEntropyRagged`, `PairwiseCrossEntropyRagged`,
    `CategoricalCrossEntropyRagged`, `HingeLossRagged`, `L2LossRagged`
  - Legacy factory names (`binary_crossentropy`,
    `balanced_pairwise_crossentropy`, …) are also accepted to ease
    migration from `torch-ns.KgeLossFactory`.
- **`kge_kernels.data`** — KGE dataset utilities:
  - `TripleExample` dataclass
  - `load_triples`, `load_triples_with_mappings`, `encode_split_triples`
    (TSV / CSV / Prolog auto-detection)
  - `add_reciprocal_triples`, `build_filter_maps`, `build_relation_domains`
- **`kge_kernels.ranking`** extended:
  - `ranking_metrics(ranks, ks=(1, 3, 10))` — `ks` now parameterized
  - `ranks_from_labeled_predictions(y_pred, y_true)` — shared rank kernel
    for `[B, N]` labeled-prediction format with padding and
    multi-positive support
  - `StreamingRankingMetrics` — in-place GPU-scalar accumulator class,
    zero allocations per update, one `.item()` at `compute()` time.
    Supports both `MRR / Hits@k` and lowercase `mrrmetric / hits@k`
    key conventions.
- Packaging:
  - `LICENSE` (MIT)
  - `py.typed` marker (PEP 561)
  - `CHANGELOG.md`
  - Full project metadata in `pyproject.toml` (authors, urls,
    keywords, classifiers)
  - GitHub Actions CI workflow running the test suite on push / PR

### Fixed

- `kge_kernels.eval.fusion.rrf` tie-breaking now works: the perturbation
  noise is added in float64 instead of float32, so perturbations of
  magnitude `1e-10` are visible (previously they were rounded away by
  float32 precision, breaking the "fair tie-breaking" and
  "seeded-reproducibility-across-seeds" tests).

### Changed

- Top-level `kge_kernels.__init__` now re-exports the full framework,
  models, losses, and data surface area. Existing imports from
  `kge_kernels.adapter`, `kge_kernels.eval`, `kge_kernels.logging`,
  `kge_kernels.partial`, `kge_kernels.ranking`, `kge_kernels.sampler`,
  and `kge_kernels.scoring` continue to work unchanged.

## [0.2.0] — 2026-04-10 (retroactive)

- Added `kge_kernels.logging` subpackage (`ExperimentSpec`, `RunContext`,
  `run_experiment`).
- Added `kge_kernels.eval` subpackage (`Evaluator`, `CandidatePool`,
  `EvalResults`, `rrf`, `zscore_fusion`).
- Added `LazyPartialScorer`: runtime per-`(pred, entity)` caching for
  partial-atom scoring, no upfront precompute.
- Model adapter module: unwraps `DataParallel` / `torch.compile`, applies
  sigmoid normalization.
- `Sampler` adds Bernoulli corruption mode and `compute_bernoulli_probs`.

## [0.1.0] — 2026-04 (retroactive, initial shared KGE kernels)

- `kge_kernels.sampler` — vectorized corruption generation with filtering,
  uniqueness, optional typed/domain-aware pools.
- `kge_kernels.scoring` — public `score()` entry point over an explicit
  `KGEBackend` contract.
- `kge_kernels.partial` — partial-atom score precompute and lookup.
- `kge_kernels.ranking` — `ranks_from_scores`, `ranks_from_scores_matrix`,
  `ranking_metrics`.

[0.3.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/releases/tag/v0.1.0
