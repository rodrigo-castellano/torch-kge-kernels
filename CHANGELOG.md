# Changelog

All notable changes to `torch-kge-kernels` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to semantic versioning starting from 0.3.0.

## [0.5.0] — 2026-04-13

### Changed

- **Package reorganization.** All eight root-level modules
  (``adapter``, ``scoring``, ``partial``, ``ranking``, ``sampler``,
  ``types``, ``utils``, ``checkpoints``) have been moved into topical
  subpackages. The new layout:

  - ``kge_kernels.scoring`` — types, corruption sampling, backend
    dispatch, model adaptation, and partial-atom scoring (absorbs the
    former ``adapter``, ``scoring``, ``partial``, ``sampler``,
    ``types``, and ``utils`` modules).
  - ``kge_kernels.eval.ranking`` — ranking metrics and
    ``StreamingRankingMetrics`` (absorbs the former ``ranking`` module).
  - ``kge_kernels.training.checkpoints`` — checkpoint save/load and
    ``unwrap_model`` (absorbs the former ``checkpoints`` module).

  Top-level imports (``from kge_kernels import Sampler``) continue to
  work unchanged. Submodule-level imports (e.g. ``from kge_kernels.sampler
  import Sampler``) must be updated to the new paths (e.g.
  ``from kge_kernels.scoring import Sampler``).

### Fixed

- **``kge_score_*`` naming collision.** Backend-level aliases in the old
  ``scoring`` module collided with model-level functions in ``adapter``.
  The backend-level aliases have been removed; ``kge_score_triples``,
  ``kge_score_all_tails``, and ``kge_score_all_heads`` now exist only in
  ``scoring.adapter`` (model-aware, sigmoid-normalized).
- **Dead ``entity_chunk`` parameter** in
  ``adapter.precompute_partial_scores`` removed.

## [0.4.3] — 2026-04-11

### Fixed

- **``StreamingRankingMetrics`` accumulator dtype.** The MRR sum
  accumulator was allocated as ``float64`` and the per-batch inverse
  rank was cast via ``ranks.double()``. This added an extra GPU kernel
  launch (float32→float64 cast) on every validation batch, which
  measurably slowed downstream consumers that call ``metric.update``
  in a hot loop (e.g. torch-ns ``evaluate_chunked`` in the speed
  regression test). Restored to ``float32`` throughout and switched
  to an in-place ``masked_fill_`` — the old torch-ns
  ``FusedRankingMetrics`` behavior that the class was meant to replace.
  MRR/Hits@k are bit-identical between float32 and float64 for
  vocabulary sizes encountered in practice (max rank ≪ 10⁷), so the
  dtype downgrade is safe.

## [0.4.2] — 2026-04-11

Framework-completeness polish pass. No public API additions — the
goal is to close out the design gaps identified in the post-Phase-7
audit of ``framework.tex``.

### Added

- **``tests/framework/test_compile_safety.py``** — verifies that the
  canonical SBR-exhaustive composition (``KGEScoreAtom`` +
  ``TNormStateRepr("min")`` + ``TNormTrajRepr("min")`` + ``MaxQueryRepr``
  + ``ExhaustiveSelect`` fed through ``search_and_score``) runs under
  ``torch.compile(fullgraph=True)`` with **zero graph breaks**. Uses
  ``torch._dynamo.explain`` for a structured report. Also exercises
  ``TNormStateRepr`` and ``MaxQueryRepr`` in isolation.  This test
  was flagged as blocking in the Phase-1 plan but was never written;
  catching up now. The ``test_compile_safety.py`` gate enforces the
  architectural contract claim that framework primitives are
  fullgraph-compile-safe by design.
- **``examples/``** directory — three runnable self-contained scripts
  that compose tkk primitives end-to-end:
  - ``01_sbr_exhaustive.py``: SBR-style exhaustive scoring via the
    reference ``search_and_score`` loop.
  - ``02_train_kge.py``: pure KGE training using ``train_kge``,
    ``NSSALoss``, cosine warmup scheduler, and
    ``evaluate_filtered_ranking``. Synthesizes a toy KG so the script
    runs without any dataset files.
  - ``03_filtered_eval.py``: standalone filtered ranking evaluation
    with both exhaustive and sampled modes.
  Each example runs in <15 seconds on CPU and is smoke-tested by
  ``tests/test_examples.py`` (parameterized via ``runpy.run_path``).

### Changed

- **``PolicyProductTrajRepr.forward``** now raises ``TypeError``
  instead of ``NotImplementedError`` with a clearer message. The
  primitive is incremental-only by design (there is no batch closed
  form because the quantity depends on which action the
  ``Select`` picked at each step) — the docstring makes that
  explicit so reviewers reading the source don't mistake it for
  a half-baked implementation.

## [0.4.1] — 2026-04-11

### Changed

- **Hot-path import hygiene.** The Phase-9 training / checkpoints /
  filtered-ranking surfaces are no longer eagerly re-exported from the
  top-level ``kge_kernels`` namespace. They are still available via their
  subpackages (``kge_kernels.training.*``, ``kge_kernels.checkpoints.*``,
  ``kge_kernels.eval.evaluate_filtered_ranking``), and ``kge_kernels.training``
  / ``kge_kernels.checkpoints`` remain accessible as module attributes.
  The top-level namespace now only loads what torch-ns and DpRL share on
  their hot paths, which noticeably reduces first-batch warmup flakiness
  on the torch-ns ``test_train_speed`` regression test in GPU-bound
  environments with a tight 10% threshold.

## [0.4.0] — 2026-04-11

Adds the shared KGE training infrastructure so any consumer can stand
up a standalone KGE training pipeline without reimplementing the
optimizer / scheduler / compile / checkpoint glue. Consumers that need
to co-train KGE with reasoning layers (e.g. torch-ns SBR / DCR / R2N)
keep their own training loops; this release gives them a library of
reusable primitives rather than a one-size-fits-all loop.

### Added

- **`kge_kernels.training`** — shared KGE training primitives:
  - `KGETrainConfig` — generic training hyperparameter dataclass
    (optimizer, scheduler, AMP, compile, grad clip, neg sampling).
    Dataset path resolution and checkpoint save dirs are deliberately
    left to each consumer.
  - `TripleDataset` — tensor-backed `torch.utils.data.Dataset` over
    integer `(r, h, t)` triples.
  - `set_seed(seed)` — reproducible Python + torch + CUDA RNG seeding.
  - `make_cosine_warmup_scheduler(opt, total_steps, warmup_ratio)` —
    linear warmup → cosine decay `LambdaLR` factory.
  - `wrap_model_for_training(model, device, cfg)` — applies optional
    `DataParallel` + `torch.compile` with the same semantics as DpRL's
    `build_training_model`.
  - `train_kge(cfg, model, dataloader, ...)` — lean KGE training loop.
    Runs the forward/loss/backward/step inner loop; validation, early
    stopping, metric logging, and checkpointing are delegated to an
    optional `on_epoch_end` callback. Supports RotatE-style entity
    modulus re-projection via `model.project_entity_modulus_()`.
- **`kge_kernels.checkpoints`** — config-agnostic checkpoint helpers:
  - `normalize_loaded_state_dict` strips the `_orig_mod.` prefix added
    by `torch.compile`.
  - `unwrap_model` returns the inner module from a `DataParallel` +
    `torch.compile` chain.
  - `model_state_dict` = `unwrap_model(m).state_dict()` shortcut.
  - `save_state_dict`, `write_json_payload`, `save_checkpoint`,
    `load_checkpoint` — plain `(state_dict, payload_dict)` persistence
    with no coupling to any specific `TrainConfig`.
- **`kge_kernels.eval.evaluate_filtered_ranking`** — filtered KGE
  evaluation with per-relation chunking, optional sampled mode,
  optional head/tail domain constraints. Handles `DataParallel` +
  `torch.compile` unwrapping internally.
- **`kge_kernels.losses.NSSALoss`** — self-adversarial negative
  sampling loss (Sun et al., 2019) used by RotatE / DpRL. Also
  available via `build_loss("nssa", adv_temp=..., neg_ratio=...)`.
- 28 new tests across `tests/training/`, `tests/test_checkpoints.py`,
  `tests/test_filtered_ranking.py`, and the NSSA test in
  `tests/losses/`.

### Changed

- `__version__` bumped to `0.4.0`.

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

[0.5.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.4.3...v0.5.0
[0.4.3]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rodrigo-castellano/torch-kge-kernels/releases/tag/v0.1.0
