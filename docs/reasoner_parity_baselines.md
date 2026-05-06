# Reasoner Parity Baselines (IJCAI '25)

Source of truth for **`scripts/run_reasoner_parity_sweep.py`** (Phase 2). The
parity script compares the consolidated tkk + ns + grounder stack against
the numbers below — drawn from the IJCAI '25 "Grounding Methods for
Neural-Symbolic AI" paper plus the keras-ns experiment CSVs at
`~/repos/keras-ns-swarm/main/experiments/runs/indiv_runs/`.

A separate, more verbose log of the *reproduction journey* (findings, bug
fixes, command lines) lives in
`~/repos/torch-ns-swarm/tkk-consolidation/docs/results_IJCAI.md`. This
document is the trimmed parity-only version: paper numbers + minimal
context, no investigation notes.

## Reasoners covered

- **SBR** — Semantic-Based Reasoning (min/max fuzzy logic).
- **DCR** — Deep Concept Reasoning (filter+sign Gödel attention).
- **R2N** — Rule-to-Neural (per-rule MLP on body-atom embeddings).

> GatedSBR / RNM / DeepStocklog / Clustered DCR appear in torch-ns but
> are NOT in the IJCAI paper's main table — those are checked elsewhere
> (smoke trains in Phase 4 of the framework-primitive migration).

## Grounder mapping (paper notation → tkk grounder string)

The paper's grounders are named `BC_{w}_{d}` (backward chaining with
width `w` and depth `d`). The mapping to tkk's `create_grounder`
type-string vocabulary:

| Paper | keras-ns name | Meaning | tkk type-string |
|---|---|---|---|
| BC₀,₁ | `backward_0_1` | depth=1, width=0 (all body atoms must be known facts; no free vars) | `enum.fp_batch.w0.d1` (with `--preground` for rules with ≥2 free vars) |
| BC₁,₁ | `backward_1_1` | depth=1, width=1 (1 free var per rule) | `enum.fp_batch.d1` |
| BC₁,₂ | `backward_1_2` | depth=2, width=1 | `enum.none.w1.d2` |
| BC₁,₃ | `backward_1_3` | depth=3, width=1 | `enum.none.w1.d3` |

`fp_batch` filters out unsoundable proofs at depth 1; for depth ≥ 2 the
filter is `none` because cross-step intermediate atoms aren't pruned
within a single batch. Pure-KGE rows use the special config
`model_name='no_reasoner'` (no grounder, no rules).

## Shared hyperparameters

| Parameter | Value |
|---|---|
| KGE | ComplEx, E=100 (entity / relation embeddings = 200 each) |
| Optimiser | Adam, lr=0.01 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-4) |
| Loss | BCE, weight_loss=0.5 |
| Train negatives | 1 per positive |
| Epochs | 100 max, early-stopping patience=50 (monitor val MRR) |
| Dropout / regularisation | 0.0 |
| DCR aggregation | max, signed=True, temperature=0.0 |

## Per-dataset overrides

| | ablation_d2 | ablation_d3 | countries_s3 | family | wn18rr |
|---|---|---|---|---|---|
| Train corruption | TAIL | TAIL | TAIL | HEAD+TAIL | HEAD+TAIL |
| Eval corruption | TAIL | TAIL | TAIL | HEAD+TAIL | HEAD+TAIL |
| Test corruptions | All | All | All (5) | All | 1000 |
| Test batch size | 256 | 256 | 128 | 4 | 1 |
| Train batch size | 256 | 256 | 256 | 256 | 256 |
| ResNet | **False** | **False** | True | True | True |
| R2N prediction_type | head | head | full | full | full |
| Domain file | `domain2constants.txt` | `domain2constants.txt` | `domain2constants.txt` | — | — |
| Seeds reported | 5 (0..4) | 5 (0..4) | 5 (0..4) | 1 (0) | 1 (0) |
| Rules | 1 (`neighborOf → locatedInCR`) | 1 | 3 (R0+R1+R2, 3-hop) | 48 (AMIE) | 17 (UniKER) |

## Eval-inflation note (TAIL-only datasets)

The keras-ns `KGCEvalDataset` builds two entries per query for TAIL-only
corruption: one real (5 candidates) and one trivial (positive only).
This **inflates** every reported metric on TAIL-only datasets:

```
inflated = (real + 1.0) / 2     ⇔     real = 2 × inflated − 100
```

Affected: **ablation_d2, ablation_d3, countries_s3, countries_s2**
(all TAIL-only). NOT affected: **family, wn18rr** (HEAD+TAIL).

The torch-ns runner used to expose an `--inflated_eval` flag to mimic
the keras-ns calculation for direct paper comparison. **It has been
removed** (2026-04-28): we always evaluate raw MRR. The numbers below
for `ablation_d2` / `ablation_d3` / `countries_s2` / `countries_s3`
remain the published (inflated) paper figures, so torch-ns will look
~5–15 absolute MRR points lower than the paper on those rows. That is
not a torch-ns regression — it is the keras-ns inflation showing up
when you compare a fixed bug-free evaluator against a buggy one. If
you ever need the paper figure exactly, compute `(real + 1) / 2`
manually from the torch-ns output for those four datasets only.

## Paper baselines

Per cell: **`MRR · Hits@1 · Hits@3 · Hits@10`** (percent). Where the
paper reports timing, **`Train(s) · Test(s)`** appears below the
metrics in italics. Numbers are mean ± std across the seed count noted
in the per-dataset overrides table; `family` / `wn18rr` are single-seed.

### ablation_d2  (5 seeds, TAIL-only, **inflated**, resnet=False)

| Reasoner | Grounder | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|
| ComplEx (no_reasoner) | — | 98.4 ± 0.9 | 97.2 ± 1.6 | 99.6 ± 0.8 | 100.0 |
| SBR | BC₀,₁ | 32.2 ± 1.3 | 9.2 ± 2.0 | 28.8 ± 2.4 | 100.0 |
| SBR | BC₁,₂ | 96.8 ± 0.9 | 95.6 ± 0.8 | 97.2 ± 1.0 | 100.0 |
| SBR | BC₁,₃ | 97.4 ± 0.5 | 96.8 ± 1.0 | 97.2 ± 1.0 | 100.0 |
| DCR | BC₀,₁ | 33.8 ± 2.3 | 6.8 ± 1.6 | 27.6 ± 4.6 | 100.0 |
| DCR | BC₁,₂ | 94.0 ± 0.6 | 92.4 ± 1.5 | 93.2 ± 1.6 | 100.0 |
| DCR | BC₁,₃ | 95.0 ± 2.0 | 92.0 ± 3.3 | 96.8 ± 1.0 | 100.0 |
| R2N | BC₁,₁ | 71.0 ± 1.6 | 59.2 ± 2.0 | 80.4 ± 2.3 | 100.0 |
| R2N | BC₁,₂ | 97.2 ± 0.2 | 95.6 ± 0.8 | 98.0 ± 0.0 | 100.0 |
| R2N | BC₁,₃ | 98.0 ± 0.7 | 96.8 ± 1.0 | 99.6 ± 0.8 | 100.0 |

Depth ≥ 2 needed for the 2-hop reasoning. At depth=1 SBR / DCR
collapse to ≈ 32 % MRR (slightly above the chance baseline of ≈ 25 %
from the inflation).

### ablation_d3  (5 seeds, TAIL-only, **inflated**, resnet=False)

| Reasoner | Grounder | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|
| ComplEx (no_reasoner) | — | 94.8 ± 1.0 | 90.8 ± 1.6 | — | 100.0 |
| SBR | BC₀,₁ | 34.8 ± 3.7 | 14.4 ± 4.5 | — | 100.0 |
| SBR | BC₁,₃ | 86.8 ± 1.2 | 82.0 ± 1.8 | — | 100.0 |
| DCR | BC₁,₃ | 86.7 ± 0.8 | 80.4 ± 1.5 | — | 100.0 |
| R2N | BC₁,₃ | 96.6 ± 1.7 | 94.0 ± 3.3 | — | 100.0 |

Depth ≥ 3 needed for the 3-hop reasoning. Hits@3 not reported in the
keras-ns CSVs we have. **Source**:
`output/legacy/experiments-logs/ijcai25/` and the keras-ns CSVs (the
torch-ns reproduction in `results_IJCAI.md` confirms these).

### countries_s2  (paper Table — **DATA NOT YET LOCATED**)

The keras-ns `experiments/runs/indiv_runs/` directory does not contain
`_ind_log-countries_s2-...csv` files; only `countries_s3` ran in the
saved CSVs. The paper's Table 1 reports countries_s2 numbers but those
need to be transcribed from the PDF / the original experiment output.
Marked **TODO**: ask the user for the table or rerun keras-ns on
countries_s2 to fill this in before the parity sweep covers it.

> The parity script in Phase 2 should accept countries_s2 in its
> config list but skip the comparison row when this section is
> unfilled.

### countries_s3  (5 seeds, TAIL-only, **inflated**, resnet=True)

| Reasoner | Grounder | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---:|---:|---:|---:|
| ComplEx (no_reasoner) | — | 88.4 ± 3.4 | 82.5 ± 6.0 | 92.1 ± 3.1 | 100.0 |
| SBR | BC₀,₁ | 95.3 ± 0.9 | 91.7 ± 1.3 | 99.2 ± 1.0 | 100.0 |
| SBR | BC₁,₂ | 96.8 ± 2.2 | 94.2 ± 3.6 | 99.2 ± 1.7 | 100.0 |
| SBR | BC₁,₃ | 97.7 ± 1.6 | 95.8 ± 2.3 | 99.6 ± 0.8 | 100.0 |
| DCR | BC₀,₁ | 93.5 ± 1.7 | 89.6 ± 2.9 | 95.8 ± 1.3 | 100.0 |
| DCR | BC₁,₂ | 96.9 ± 1.1 | 94.2 ± 2.0 | 100.0 | 100.0 |
| DCR | BC₁,₃ | 97.6 ± 0.9 | 95.4 ± 1.6 | 100.0 | 100.0 |
| R2N | BC₀,₁ | 90.7 ± 2.0 | 85.0 ± 3.3 | 95.0 ± 1.0 | 100.0 |
| R2N | BC₁,₂ | 88.9 ± 3.2 | 82.5 ± 4.9 | 93.8 ± 1.9 | 100.0 |
| R2N | BC₁,₃ | 89.5 ± 3.2 | 83.3 ± 5.1 | 94.2 ± 2.0 | 100.0 |

S3 requires 3-hop reasoning. SBR / DCR at depth ≥ 2 reach ≈ 97 %
(+ 9 over the KGE baseline). R2N flattens / overfits at higher depths.

### family  (single seed, all corruptions, resnet=True)

Paper Table 2 also reports per-config training and test wall-clock
seconds. Both are eager TF on a single A100 — useful to confirm we're
in the right *order of magnitude* on a comparable GPU; absolute
numbers will differ.

| Reasoner | Grounder | MRR | Hits@1 | Hits@3 | Hits@10 | Train(s) | Test(s) |
|---|---|---:|---:|---:|---:|---:|---:|
| ComplEx (no_reasoner) | — | 85.9 | 79.2 | 92.2 | 94.5 | 773 | 285 |
| SBR | BC₀,₁ | 86.9 | 78.0 | 95.6 | 97.1 | 9 067 | 6 209 |
| SBR | BC₁,₂ | 87.7 | 79.1 | 96.0 | 97.3 | 43 355 | 27 448 |
| DCR | BC₀,₁ | 90.1 | 84.1 | 95.9 | 97.0 | 16 480 | 7 659 |
| DCR | BC₁,₂ | 90.1 | 84.1 | 95.9 | 97.0 | 16 295 | 7 517 |
| R2N | BC₀,₁ | **94.0** | **92.1** | 95.6 | 96.5 | 9 573 | 6 616 |
| R2N | BC₁,₂ | 91.8 | 87.1 | **96.4** | **97.4** | 48 809 | 28 249 |

R2N @ BC₀,₁ is best on MRR / Hits@1; R2N @ BC₁,₂ is best on Hits@3 /
Hits@10. Reasoning adds + 1 to + 9 % MRR over the KGE baseline.

> The IJCAI 0.859 family-ComplEx number is single-seed; an independent
> NeSy-paper reproduction with the same code but seed=1 lands at
> 0.842 (a 0.05 swing). When the parity script runs family it should
> use ≥ 3 seeds and report the spread, not the single point.

### wn18rr  (single seed, 1000 head+tail corruptions, resnet=True)

| Reasoner | Grounder | MRR | Hits@1 | Hits@3 | Hits@10 | Train(s) | Test(s) |
|---|---|---:|---:|---:|---:|---:|---:|
| ComplEx (no_reasoner) | — | 42.7 | 40.8 | 42.9 | 45.9 | 1 079 | 139 |
| SBR | BC₀,₁ | 44.0 | 42.3 | 44.2 | 46.6 | 21 941 | 1 910 |
| SBR | BC₁,₂ | 44.7 | 42.5 | 45.2 | 48.2 | 67 852 | 6 666 |
| DCR | BC₀,₁ | 44.2 | 42.2 | 44.8 | 47.6 | 26 133 | 2 338 |
| DCR | BC₁,₂ | **45.6** | **42.9** | **47.1** | **50.2** | 74 627 | 6 944 |
| R2N | BC₀,₁ | 44.2 | 42.3 | 44.6 | 47.3 | 20 614 | 2 183 |
| R2N | BC₁,₂ | 44.1 | 41.4 | 45.4 | 48.1 | 72 213 | 7 353 |

DCR @ BC₁,₂ best overall (45.6 MRR). BC₁,₂ consistently beats BC₀,₁
at ≈ 3× wall-clock cost. Only BC₀,₁ and BC₁,₂ tested due to scale —
BC₁,₃ is skipped on wn18rr in the paper.

## Parity tolerances (proposal for Phase 2)

The parity script flags drift via these defaults; tighten / loosen
per-cell as needed:

- **MRR / Hits**: ± 1.0 percent absolute on multi-seed datasets,
  ± 3.0 percent on single-seed datasets (family, wn18rr) — the
  family-paper-vs-NeSy gap shows that ± 5 points is inside the
  single-seed envelope on its own.
- **Timing**: ± 30 % wall-clock — paper numbers are eager TF on a
  single A100; tkk + ns runs `torch.compile(reduce-overhead)` on
  whatever GPU is local, so ratios across configs are more
  meaningful than absolute seconds.

## Items to chase before Phase 2 runs the sweep

1. **Fill the countries_s2 table** (transcribe from the IJCAI PDF or
   rerun keras-ns; the keras-ns repo ships the data and rules for it
   but no saved CSVs).
2. **ablation_d3 Hits@3** — not in the keras-ns CSVs we have; either
   accept the gap or recompute from the saved logs in
   `output/legacy/experiments-logs/ijcai25/`.
3. **Confirm wn18rr 1000-corruption seeds** — paper says single seed,
   `output/runs/20260414-wn18rr-paper-1000neg/` may have ≥ 1; if more
   are available, replace the std-less row with mean ± std.
