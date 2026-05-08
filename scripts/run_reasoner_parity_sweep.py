#!/usr/bin/env python
"""Reasoner parity sweep — IJCAI '25 grid through tkk + ns + grounder.

Companion to ``docs/reasoner_parity_baselines.md``: trains every
(reasoner × grounder × dataset) row from the paper's Table 1 / Table 2
through ``torch_ns.experiment.pipeline`` and diffs the resulting
MRR / Hits / wall-clock against the baselines parsed from that doc.
Optionally invokes ``tests/profiling/profile_train.py`` and
``profile_eval.py`` to capture per-config ms / batch numbers so the
parity report tracks both *outcomes* (MRR) and *cost* (latency,
peak memory).

Mirrors ``scripts/run_3way_sweep.py``'s shape — same in-process dual
import via ``find_spec("torch_ns")``, same run-bundle layout under
``output/runs/<experiment>/<run>/`` (stdout.log + config.json + final
manifest.json + parity_report.md), same per-iteration GPU cleanup
between sub-runs.

Usage::

    # Quick subset (countries_s3 only, 3 epochs, no profiling)
    python -u scripts/run_reasoner_parity_sweep.py --only countries_s3 --smoke

    # Single reasoner, all its rows
    python -u scripts/run_reasoner_parity_sweep.py --reasoner sbr

    # Full grid + ms/batch profiling (multi-hour)
    python -u scripts/run_reasoner_parity_sweep.py --profile

    # Profile only (no full training; requires existing run_dir for MRR rows)
    python -u scripts/run_reasoner_parity_sweep.py --only family --profile_only

CLI flags:
    --only DS [DS ...]        keep only these datasets
    --reasoner R [R ...]      keep only these reasoners (sbr / dcr / r2n / no_reasoner)
    --grounder G [G ...]      keep only these paper grounder names (BC01 / BC11 / BC12 / BC13)
    --seeds N                 override per-dataset seed count (default: paper convention)
    --epochs N                override max epochs (default 100; smoke=3)
    --smoke                   3-epoch quick run, no inflation correction needed
    --profile                 also run profile_train + profile_eval per config
    --profile_only            skip full training; only ms/batch profiling
    --experiment_name NAME    output subdirectory under output/runs/
    --run_name NAME           run subdirectory under experiment_name
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# In-process resolution of torch-ns (matches run_3way_sweep.py).
_spec = importlib.util.find_spec("torch_ns")
if _spec and _spec.submodule_search_locations:
    NS_REPO = os.path.dirname(next(iter(_spec.submodule_search_locations)))
else:
    NS_REPO = os.environ.get(
        "NS_REPO",
        os.path.expanduser("~/repos/torch-ns-swarm/tkk-consolidation"),
    )
    sys.path.insert(0, NS_REPO)

import torch  # noqa: E402

DATA_ROOT = os.path.expanduser("~/repos/data-swarm/main")
PAPER_SEED_BASE = 0


# ─────────────────────────────────────────────────────────────────────
# Paper grounder mapping (BC_w_d → tkk type-string)
# ─────────────────────────────────────────────────────────────────────
GROUNDER_MAP = {
    # All cells use fp_batch leaf-closure: every body atom of a returned
    # grounding must be a known fact (T_P fixed point per batch). The
    # IJCAI '25 grounder closes its leaves the same way; without it our
    # grounder admits proofs whose leaves are unbound atoms, inflating
    # grounding counts (e.g. 7x on BC13) and feeding spurious firings to
    # the reasoner — which silently wrecks R2N parity once depth ≥ 2.
    # ``.flat`` engages the eager flat-intermediate path which fires
    # ``bc.considered.capture_step`` per step, giving rule_groundings
    # that exactly match keras-ns ``rule2groundings``. The compiled
    # dense path skips that hook (Python list.append breaks fullgraph)
    # and falls back to evidence-only firings, undercounting by ~3×
    # for BC_{w,d} cells with d ≥ 2.
    "BC01": "enum.fp_batch.w0.d1.flat",  # depth=1, w=0; preground for ≥2 free-var rules
    "BC11": "enum.fp_batch.d1.flat",     # depth=1, w=1
    "BC12": "enum.fp_batch.w1.d2.flat",  # depth=2, w=1
    "BC13": "enum.fp_batch.w1.d3.flat",  # depth=3, w=1
}


# ─────────────────────────────────────────────────────────────────────
# Keras-ns grounder mapping (paper BC_{w, d} → ns_lib name)
# ─────────────────────────────────────────────────────────────────────
# keras-ns ``ApproximateBackwardChainingGrounder`` is dispatched by the
# string ``backward_<W>_<D>``. ``backward_0_1`` is special (the
# "no-grounding" baseline; only one rule app per query, body == fact).
# ``u`` (= ``max_unknown_fact_count_last_step``) is hardcoded to 0 in
# our keras-ns patch so the leaf closure matches paper convention,
# regardless of the input ``W``/``D``.
KERAS_GROUNDER_MAP = {
    "BC01": "backward_0_1",
    "BC11": "backward_1_1",
    "BC12": "backward_1_2",
    "BC13": "backward_1_3",
}


# Per-paper-grounder reasoner filter for keras-ns. The keras-ns runner
# rejects the no_reasoner baseline with anything but backward_0_1.
def _keras_skip(dataset: str, reasoner: str, grounder: str) -> bool:
    if reasoner == "no_reasoner" and grounder != "BC01":
        return True
    return False


# ─────────────────────────────────────────────────────────────────────
# Per-dataset overrides (from docs/reasoner_parity_baselines.md)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DatasetSpec:
    corrupt_mode: str
    test_negatives: int           # 0 = exhaustive (all entities)
    valid_negatives: int
    test_batch_size: int
    val_batch_size: int
    domain_file: Optional[str]
    resnet: bool
    r2n_prediction_type: str      # "head" | "full" — preserved for parity hyperparam log
    seeds: int                    # paper-default seed count


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ablation_d2": DatasetSpec(
        corrupt_mode="tail", test_negatives=0, valid_negatives=0,
        test_batch_size=256, val_batch_size=256,
        domain_file="domain2constants.txt", resnet=False,
        r2n_prediction_type="head", seeds=5,
    ),
    "ablation_d3": DatasetSpec(
        corrupt_mode="tail", test_negatives=0, valid_negatives=0,
        test_batch_size=256, val_batch_size=256,
        domain_file="domain2constants.txt", resnet=False,
        r2n_prediction_type="head", seeds=5,
    ),
    "countries_s2": DatasetSpec(
        corrupt_mode="tail", test_negatives=0, valid_negatives=0,
        test_batch_size=128, val_batch_size=128,
        domain_file="domain2constants.txt", resnet=True,
        r2n_prediction_type="full", seeds=5,
    ),
    "countries_s3": DatasetSpec(
        corrupt_mode="tail", test_negatives=0, valid_negatives=0,
        test_batch_size=128, val_batch_size=128,
        domain_file="domain2constants.txt", resnet=True,
        r2n_prediction_type="full", seeds=5,
    ),
    "family": DatasetSpec(
        corrupt_mode="both", test_negatives=0, valid_negatives=0,
        test_batch_size=4, val_batch_size=4,
        domain_file=None, resnet=True,
        r2n_prediction_type="full", seeds=1,
    ),
    "wn18rr": DatasetSpec(
        corrupt_mode="both", test_negatives=1000, valid_negatives=1000,
        test_batch_size=1, val_batch_size=1,
        domain_file=None, resnet=True,
        r2n_prediction_type="full", seeds=1,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Paper baselines (parsed from docs/reasoner_parity_baselines.md)
# Tuple shape: (mrr_mean, mrr_std, h1, h3, h10, train_s, test_s)
# Use ``None`` for any cell the paper doesn't report.
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Baseline:
    mrr_mean: float
    mrr_std: Optional[float] = None
    h1: Optional[float] = None
    h3: Optional[float] = None
    h10: Optional[float] = None
    train_s: Optional[float] = None
    test_s: Optional[float] = None


def _uninflate(v: Optional[float]) -> Optional[float]:
    """Inverse of keras-ns main-branch TAIL-only inflation.

    keras-ns ``main`` ``KGCEvalDataset.__getitem__`` always emits two
    eval entries per query (one head-corruptions, one tail-corruptions).
    For TAIL-only-corruption datasets the head entry collapses to a
    single positive → MRR contribution = 1.0 — so the published metric
    is ``inflated = (real + 1) / 2`` (in percent: ``(real + 100) / 2``).
    The fix lives in branch ``dcr_r2n_with_neural_grounder``
    (``experiments/kge/dataset.py:KGCDataset.__getitem__`` lines 191-200,
    conditional ``if c.head:`` / ``if c.tail:``); torch-ns has the
    equivalent fix natively. Inverse: ``real = 2 * inflated - 100``,
    clamped so negative ``real`` values (which would imply random or
    worse) are truncated at 0.
    """
    if v is None:
        return None
    return max(0.0, 2.0 * v - 100.0)


def _real_baseline(bl: Baseline, dataset: str) -> Baseline:
    """Return the un-inflated baseline (what `dcr_r2n_with_neural_grounder`
    would produce on the same training run). For HEAD+TAIL datasets
    this is a no-op; for TAIL-only it inverts the keras-ns inflation
    cell-by-cell across MRR + Hits@k + std.
    """
    if DATASET_SPECS[dataset].corrupt_mode != "tail":
        return bl
    return Baseline(
        mrr_mean=_uninflate(bl.mrr_mean),
        # std doubles under the linear (2x - 100) inversion
        mrr_std=None if bl.mrr_std is None else 2.0 * bl.mrr_std,
        h1=_uninflate(bl.h1),
        h3=_uninflate(bl.h3),
        h10=_uninflate(bl.h10),
        train_s=bl.train_s,
        test_s=bl.test_s,
    )


BASELINES: dict[tuple, Baseline] = {
    # Cells below are the **published paper baselines** parsed from the
    # keras-ns ``main`` branch CSVs. For TAIL-only datasets — ablation_d2,
    # ablation_d3, countries_s2, countries_s3 — keras-ns ``main``'s
    # ``KGCEvalDataset.__getitem__`` (experiments/dataset.py:202-208)
    # emits a trivial "head-only" entry for every query whose MRR is
    # always 1.0, then averages with the real ranking, yielding
    # ``inflated = (real + 100) / 2``. The bug is fixed in branch
    # ``dcr_r2n_with_neural_grounder`` (experiments/kge/dataset.py:191-200,
    # conditional ``if c.head:`` / ``if c.tail:``).
    #
    # The parity report renders three numbers per TAIL-only cell: the
    # paper figure (as below), its un-inflated equivalent (= what
    # ``dcr_r2n_with_neural_grounder`` would produce ≡ what torch-ns
    # produces natively, computed via ``_real_baseline``), and ours.
    # ΔMRR is taken vs the un-inflated baseline.
    # See CLAUDE.md and ``docs/reasoner_parity_baselines.md``.
    # ── ablation_d2 (5 seeds)
    ("ablation_d2", "no_reasoner", None):  Baseline(98.4, 0.9, 97.2, 99.6, 100.0),
    ("ablation_d2", "sbr", "BC01"):        Baseline(32.2, 1.3, 9.2,  28.8, 100.0),
    ("ablation_d2", "sbr", "BC12"):        Baseline(96.8, 0.9, 95.6, 97.2, 100.0),
    ("ablation_d2", "sbr", "BC13"):        Baseline(97.4, 0.5, 96.8, 97.2, 100.0),
    ("ablation_d2", "dcr", "BC01"):        Baseline(33.8, 2.3, 6.8,  27.6, 100.0),
    ("ablation_d2", "dcr", "BC12"):        Baseline(94.0, 0.6, 92.4, 93.2, 100.0),
    ("ablation_d2", "dcr", "BC13"):        Baseline(95.0, 2.0, 92.0, 96.8, 100.0),
    ("ablation_d2", "r2n", "BC11"):        Baseline(71.0, 1.6, 59.2, 80.4, 100.0),
    ("ablation_d2", "r2n", "BC12"):        Baseline(97.2, 0.2, 95.6, 98.0, 100.0),
    ("ablation_d2", "r2n", "BC13"):        Baseline(98.0, 0.7, 96.8, 99.6, 100.0),
    # ── ablation_d3 (5 seeds; paper omits Hits@3)
    ("ablation_d3", "no_reasoner", None):  Baseline(94.8, 1.0, 90.8, None, 100.0),
    ("ablation_d3", "sbr", "BC01"):        Baseline(34.8, 3.7, 14.4, None, 100.0),
    ("ablation_d3", "sbr", "BC13"):        Baseline(86.8, 1.2, 82.0, None, 100.0),
    ("ablation_d3", "dcr", "BC13"):        Baseline(86.7, 0.8, 80.4, None, 100.0),
    ("ablation_d3", "r2n", "BC13"):        Baseline(96.6, 1.7, 94.0, None, 100.0),
    # ── countries_s2 (5 seeds; baselines extracted from IJCAI '25
    # Figure 5, "Dataset S2, width 1" sub-plot. Y-axis is 97-100, so
    # values are read approximately from chart bars/markers.
    # Paper always uses u=0; at u=0+d=1, BC₀,₁ ≡ BC₁,₁ (no firings),
    # so BC01 here corresponds to the paper's d=1 column under
    # width=1. Chart shows ±1 error bars; std ≈ 1pp.
    # Source: IJCAI '25 Fig 5, page 6.
    ("countries_s2", "no_reasoner", None): Baseline(98.5, 1.0, None, None, None),
    ("countries_s2", "sbr", "BC01"):       Baseline(99.5, 1.0, None, None, None),
    ("countries_s2", "sbr", "BC12"):       Baseline(99.5, 1.0, None, None, None),
    ("countries_s2", "sbr", "BC13"):       Baseline(99.5, 1.0, None, None, None),
    ("countries_s2", "dcr", "BC01"):       Baseline(99.5, 1.0, None, None, None),
    ("countries_s2", "dcr", "BC12"):       Baseline(99.0, 1.0, None, None, None),
    ("countries_s2", "dcr", "BC13"):       Baseline(97.0, 2.0, None, None, None),
    ("countries_s2", "r2n", "BC01"):       Baseline(99.0, 1.0, None, None, None),
    ("countries_s2", "r2n", "BC12"):       Baseline(99.0, 1.0, None, None, None),
    ("countries_s2", "r2n", "BC13"):       Baseline(99.0, 1.0, None, None, None),
    # ── countries_s3 (5 seeds)
    ("countries_s3", "no_reasoner", None): Baseline(88.4, 3.4, 82.5, 92.1, 100.0),
    ("countries_s3", "sbr", "BC01"):       Baseline(95.3, 0.9, 91.7, 99.2, 100.0),
    ("countries_s3", "sbr", "BC12"):       Baseline(96.8, 2.2, 94.2, 99.2, 100.0),
    ("countries_s3", "sbr", "BC13"):       Baseline(97.7, 1.6, 95.8, 99.6, 100.0),
    ("countries_s3", "dcr", "BC01"):       Baseline(93.5, 1.7, 89.6, 95.8, 100.0),
    ("countries_s3", "dcr", "BC12"):       Baseline(96.9, 1.1, 94.2, 100.0, 100.0),
    ("countries_s3", "dcr", "BC13"):       Baseline(97.6, 0.9, 95.4, 100.0, 100.0),
    ("countries_s3", "r2n", "BC01"):       Baseline(90.7, 2.0, 85.0, 95.0, 100.0),
    ("countries_s3", "r2n", "BC12"):       Baseline(88.9, 3.2, 82.5, 93.8, 100.0),
    ("countries_s3", "r2n", "BC13"):       Baseline(89.5, 3.2, 83.3, 94.2, 100.0),
    # ── family (1 seed)
    ("family", "no_reasoner", None):       Baseline(85.9, None, 79.2, 92.2, 94.5,  773.0,   285.0),
    ("family", "sbr", "BC01"):             Baseline(86.9, None, 78.0, 95.6, 97.1,  9067.0,  6209.0),
    ("family", "sbr", "BC12"):             Baseline(87.7, None, 79.1, 96.0, 97.3,  43355.0, 27448.0),
    ("family", "dcr", "BC01"):             Baseline(90.1, None, 84.1, 95.9, 97.0,  16480.0, 7659.0),
    ("family", "dcr", "BC12"):             Baseline(90.1, None, 84.1, 95.9, 97.0,  16295.0, 7517.0),
    ("family", "r2n", "BC01"):             Baseline(94.0, None, 92.1, 95.6, 96.5,  9573.0,  6616.0),
    ("family", "r2n", "BC12"):             Baseline(91.8, None, 87.1, 96.4, 97.4,  48809.0, 28249.0),
    # ── wn18rr (1 seed, 1000 corruptions)
    ("wn18rr", "no_reasoner", None):       Baseline(42.7, None, 40.8, 42.9, 45.9,  1079.0,  139.0),
    ("wn18rr", "sbr", "BC01"):             Baseline(44.0, None, 42.3, 44.2, 46.6,  21941.0, 1910.0),
    ("wn18rr", "sbr", "BC12"):             Baseline(44.7, None, 42.5, 45.2, 48.2,  67852.0, 6666.0),
    ("wn18rr", "dcr", "BC01"):             Baseline(44.2, None, 42.2, 44.8, 47.6,  26133.0, 2338.0),
    ("wn18rr", "dcr", "BC12"):             Baseline(45.6, None, 42.9, 47.1, 50.2,  74627.0, 6944.0),
    ("wn18rr", "r2n", "BC01"):             Baseline(44.2, None, 42.3, 44.6, 47.3,  20614.0, 2183.0),
    ("wn18rr", "r2n", "BC12"):             Baseline(44.1, None, 41.4, 45.4, 48.1,  72213.0, 7353.0),
}


# ─────────────────────────────────────────────────────────────────────
# Run a single (dataset, reasoner, grounder) config
# ─────────────────────────────────────────────────────────────────────
def _build_cfg(
    dataset: str, reasoner: str, grounder: Optional[str], seed: int,
    *, epochs: int,
    torch_ckpt_root: Optional[Path] = None,
):
    """Construct ``NSTrainConfig`` for one parity row.

    If ``torch_ckpt_root`` is set, the per-cell ckpt directory inside
    the run bundle is wired into ``cfg._run_root`` so the
    ``ModelCheckpoint`` callback writes there (otherwise the builder
    raises since ``_run_root`` is normally populated by the torch-ns
    CLI orchestrator, which we bypass).
    """
    from torch_ns.config import NSTrainConfig

    spec = DATASET_SPECS[dataset]
    grounder_str = (
        GROUNDER_MAP[grounder] if grounder is not None else "sld.prune.d1"
    )

    # Reasoner-specific flags. ``no_reasoner`` skips grounding entirely;
    # depth ≥ 2 grounders need a smaller batch to fit memory budget.
    if reasoner == "no_reasoner":
        # Pure-KGE row: reproducer uses HEAD+TAIL training with
        # corrupt_mode set per dataset; bs768 matches keras 256 queries.
        train_bs = 768
        compile_mode = None
    else:
        train_bs = (
            128 if grounder == "BC12" else
            64  if grounder == "BC13" else
            256
        )
        compile_mode = "reduce-overhead"

    overrides = {}
    if grounder == "BC13":
        # Trim grounding budget for the depth-3 OOM corner.
        overrides.update(
            max_groundings=16, max_total_groundings=32, max_facts_per_query=32,
        )

    # R2N at depth ≥ 2 diverges at lr=0.01 (val loss explodes 1.0 → 9.4
    # within 10 epochs on 2/5 seeds; mean MRR 0.63 std 0.36 — bimodal
    # train/stuck split). lr=0.001 closes this gap (mean 0.93 std 0.01)
    # and gives equivalent or better convergence for sbr/dcr (sbr BC13
    # +2pp, dcr BC12 +2pp; sbr BC12 unchanged). Paper used lr=0.01;
    # match-paper convention only for sbr/dcr (where it converges)
    # and use the stabilizing lr for r2n (where it doesn't).
    learning_rate = 0.001 if reasoner == "r2n" else 0.01
    cfg = NSTrainConfig(
        dataset_name=dataset, data_path=DATA_ROOT,
        model_name=reasoner, kge="complex", grounder=grounder_str,
        kge_atom_embedding_size=100,
        num_rules=9999,
        reasoner_depth=None,
        resnet=spec.resnet,
        epochs=epochs, batch_size=train_bs,
        learning_rate=learning_rate, num_negatives=1,
        valid_frequency=1, lr_sched="plateau", lr_patience=10,
        early_stopping=True, patience=50,
        loss="binary_crossentropy", weight_loss=0.5,
        corrupt_mode=spec.corrupt_mode,
        test_negatives=spec.test_negatives,
        valid_negatives=spec.valid_negatives,
        test_batch_size=spec.test_batch_size,
        val_batch_size=spec.val_batch_size,
        domain_file=spec.domain_file,
        seed=seed, seed_run_i=seed,
        compile_mode=compile_mode,
        no_compile=(compile_mode is None),
        # When ``torch_ckpt_root`` is provided we wire the per-cell
        # bundle dir into ``cfg._run_root`` after construction so the
        # ``ModelCheckpoint`` callback writes into the parity bundle.
        # Otherwise leave ``checkpoint_mode='none'`` (the builder
        # raises if checkpoint_mode != 'none' and _run_root is None).
        checkpoint_mode="best" if torch_ckpt_root is not None else "none",
        output_root=os.path.join(NS_REPO, "output"),
        run_signature=f"parity_{dataset}_{reasoner}_{grounder or 'none'}_s{seed}",
        **overrides,
    )
    if torch_ckpt_root is not None:
        cell_dir = (
            torch_ckpt_root
            / f"{dataset}__{reasoner}__{grounder or 'none'}__seed{seed}"
        )
        cell_dir.mkdir(parents=True, exist_ok=True)
        cfg._run_root = str(cell_dir)
    return cfg


def _train_one(cfg) -> dict:
    """Run a single training+eval and return metrics + wall-clock.

    Plumbs ``R2N_PREDICTION_TYPE`` from the dataset spec to the env var
    that ``torch_ns.reasoners.make_reasoner_primitives`` reads at model
    build time. Paper convention: ``"head"`` for ``ablation_*``,
    ``"full"`` for everything else (countries, family, wn18rr).
    Without this plumbing R2N falls back to the module default
    (``"full"``) and ablation runs use the wrong prediction mode.
    """
    spec = DATASET_SPECS.get(cfg.dataset_name)
    prev_r2n_type = os.environ.get("R2N_PREDICTION_TYPE")
    if spec is not None:
        os.environ["R2N_PREDICTION_TYPE"] = spec.r2n_prediction_type
    try:
        from torch_ns.experiment import pipeline as ns_pipeline
        t0 = time.perf_counter()
        train_m, valid_m, test_m, _ = ns_pipeline(cfg)
        wall = time.perf_counter() - t0
    finally:
        if prev_r2n_type is None:
            os.environ.pop("R2N_PREDICTION_TYPE", None)
        else:
            os.environ["R2N_PREDICTION_TYPE"] = prev_r2n_type
    return {
        "mrr":  test_m.get("MRR", 0.0) * 100.0,
        "h1":   test_m.get("Hits@1", 0.0) * 100.0,
        "h3":   test_m.get("Hits@3", 0.0) * 100.0,
        "h10":  test_m.get("Hits@10", 0.0) * 100.0,
        "wall": wall,
    }


def _torch_seed_subprocess(
    dataset: str, reasoner: str, grounder: Optional[str], seed: int,
    *, epochs: int, torch_ckpt_root: Optional[Path] = None,
) -> dict:
    """Run one (dataset, reasoner, grounder, seed) torch training in
    a fresh subprocess so multiple seeds can share the GPU
    concurrently. Each subprocess gets its own CUDA context. The
    spawned process re-invokes this script with ``--torch-cell``.
    """
    import subprocess, json, tempfile
    out = Path(tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False).name)
    cell_arg = f"{dataset}:{reasoner}:{grounder or ''}:{seed}"
    cmd = [
        sys.executable, "-u", __file__,
        "--torch-cell", cell_arg,
        "--torch-cell-out", str(out),
        "--epochs", str(epochs),
    ]
    if torch_ckpt_root is not None:
        cmd += ["--torch-cell-ckpt-root", str(torch_ckpt_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Surface tail of stderr for easier debugging.
        tail = "\n".join(proc.stderr.splitlines()[-30:])
        raise RuntimeError(
            f"torch subprocess failed (returncode={proc.returncode})\n"
            f"--- stderr tail ---\n{tail}")
    try:
        with open(out) as fh:
            return json.load(fh)
    finally:
        out.unlink(missing_ok=True)


def _profile_one(cfg) -> dict:
    """ms/batch + peak memory via ns's existing profile_train + profile_eval."""
    sys.path.insert(0, os.path.join(NS_REPO, "tests"))
    from profiling.profile_train import run as profile_train_run
    from profiling.profile_eval import run as profile_eval_run

    train_metrics = profile_train_run(cfg)["metrics"]
    eval_metrics = profile_eval_run(cfg)["metrics"]
    return {
        "ms_batch_train":     train_metrics.get("ms_batch"),
        "fwd_gpu_ms_batch":   train_metrics.get("fwd_gpu_ms_per_batch"),
        "peak_mem_train_mb":  train_metrics.get("peak_mem_mb"),
        "ms_batch_eval":      eval_metrics.get("ms_batch"),
        "fwd_gpu_ms_eval":    eval_metrics.get("fwd_gpu_ms_per_batch"),
        "peak_mem_eval_mb":   eval_metrics.get("peak_mem_mb"),
    }


# ─────────────────────────────────────────────────────────────────────
# keras-ns runner (subprocess to ``experiments/runner.py``)
# ─────────────────────────────────────────────────────────────────────
KERAS_NS_ROOT = os.path.expanduser("~/repos/keras-ns-swarm/main")

# Backend registry. Each entry maps a short name (used in column
# headings and log subdirs) to the keras-ns checkout root. ``main`` is
# our patched copy (un-inflated eval, u=0 grounder factory, -inf MRR
# padding) — produces apples-to-apples un-inflated MRRs. ``ijcai`` is
# the IJCAI '25 paper code (no patches) — its CSVs match the
# published ``Paper(infl)`` figures up to seed variance.
KERAS_BACKENDS: dict[str, Path] = {
    "main":  Path("~/repos/keras-ns-swarm/main").expanduser(),
    "ijcai": Path("~/repos/keras-ns-swarm/ijcai").expanduser(),
}


def _parse_keras_csv(csv_path: Path) -> dict:
    """Read a keras-ns ``_ind_log-...csv`` → final test metrics dict.

    The file is multi-section, semicolon-separated:

      <epoch_rows>
      <epoch_header>
      All data;<config_kvs>
      train;<train_kvs>
      valid;<valid_kvs>
      test;<test_kvs>           ← we read this row
      training_info;<...>

    Each section is ``<name>;k1:v1;k2:v2;...``. The ``test`` row
    holds ``task_mrr``, ``task_hits@1`` etc. — un-inflated when our
    ``KERAS_NS_FIX_EVAL_INFLATION=1`` patch is active in keras-ns
    ``experiments/dataset.py``.
    """
    test_kvs: dict = {}
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("test;"):
                continue
            # Split off the "test" tag, then parse k:v pairs.
            parts = line.split(";")[1:]
            for kv in parts:
                if ":" not in kv:
                    continue
                k, v = kv.split(":", 1)
                test_kvs[k.strip()] = v.strip()
            break
    if not test_kvs:
        raise ValueError(f"keras-ns log {csv_path} has no ``test`` row")

    def _get(key: str) -> Optional[float]:
        v = test_kvs.get(key)
        if v is None or v == "" or v == "None":
            return None
        try:
            return float(v) * 100.0
        except ValueError:
            return None

    return {
        "mrr": _get("task_mrr"),
        "h1":  _get("task_hits@1"),
        "h3":  _get("task_hits@3"),
        "h10": _get("task_hits@10"),
    }


def _run_keras_one(
    dataset: str, reasoner: str, paper_grounder: str, seed: int,
    *, log_folder: Path, epochs: int, eval_fix: bool = True,
    keras_root: Path,
) -> dict:
    """Drive ``experiments/runner.py`` of one keras-ns checkout as a subprocess.

    Reuses the existing ``_ind_log-...csv`` writer. After the run
    finishes we glob the per-config CSV and parse the final test row.
    Returns ``{mrr, h1, h3, h10}`` in percent.

    The ``keras_root`` parameter selects the checkout (e.g.
    ``~/repos/keras-ns-swarm/main`` for the patched copy or
    ``~/repos/keras-ns-swarm/ijcai`` for the un-patched IJCAI '25
    paper code). The same CLI surface (``--epochs --resnet --kge
    --log_folder --ckpt_folder``) is used in both — the ijcai branch
    accepts these via a small CLI-compat patch.

    The ``KERAS_NS_FIX_EVAL_INFLATION`` env var only takes effect on
    branches that implement it (``main``); on ``ijcai`` it's ignored
    and the original eval (with the inflation bug) runs.

    Always:
      * ``--kge complex`` (paper KGE).
      * ``--resnet False`` for ablation_*, ``True`` for everything else.
      * CPU-only (``CUDA_VISIBLE_DEVICES=""``) so the GPU stays free
        for parallel torch-ns runs.
    """
    import subprocess
    keras_g = KERAS_GROUNDER_MAP.get(paper_grounder)
    if keras_g is None:
        raise ValueError(
            f"No keras-ns grounder for paper grounder {paper_grounder!r}")

    resnet_flag = "False" if dataset.startswith("ablation_") else "True"
    indiv_runs = log_folder / "indiv_runs"
    indiv_runs.mkdir(parents=True, exist_ok=True)
    (log_folder / "experiments").mkdir(parents=True, exist_ok=True)

    # Per-cell ckpt folder lives inside the run bundle so all
    # artifacts (logs + ckpts) stay together. Per-seed stdout log
    # also written into the same folder.
    ckpt_folder = log_folder / "ckpts" / f"seed_{seed}"
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    stdout_log = log_folder / f"seed_{seed}.log"

    # Backend dispatch: ``main`` accepts main-runner CLI flags
    # (--epochs / --resnet / --kge / --log_folder / --ckpt_folder).
    # ``ijcai`` is the paper repo and only accepts its original CLI
    # surface (--d / --m / --g / --s) with --s as a list literal —
    # we keep ijcai untouched, so route logs to its own ``experiments/
    # runs/indiv_runs/`` and glob the latest matching CSV from there.
    backend_name = keras_root.name
    if backend_name == "ijcai":
        cmd = [
            sys.executable, "-u", "experiments/runner.py",
            "--d", dataset, "--m", reasoner, "--g", keras_g,
            "--s", f"[{seed}]",
        ]
        ijcai_csv_dir = keras_root / "experiments" / "runs" / "indiv_runs"
        # ijcai's runner short-circuits if it sees an existing CSV
        # for this (run_signature, seed). Clean any stale files for
        # the same (dataset, grounder, seed) so the run actually
        # executes. Also clean ``_tmp_log-...`` from a prior crashed
        # run to avoid the logger appending to it.
        for stale in ijcai_csv_dir.glob(
                f"_ind_log-{dataset}-{keras_g}-*-seed_{seed}.csv"):
            stale.unlink()
        for stale in (keras_root / "experiments" / "runs").glob(
                f"_tmp_log-{dataset}-{keras_g}-*-seed_{seed}.csv"):
            stale.unlink()
    else:
        cmd = [
            sys.executable, "-u", "experiments/runner.py",
            "--d", dataset, "--m", reasoner, "--g", keras_g,
            "--s", str(seed), "--epochs", str(epochs),
            "--resnet", resnet_flag, "--kge", "complex",
            "--log_folder", str(log_folder),
            "--ckpt_folder", str(ckpt_folder),
        ]
        ijcai_csv_dir = None

    env = os.environ.copy()
    env["KERAS_NS_FIX_EVAL_INFLATION"] = "1" if eval_fix else "0"
    env["CUDA_VISIBLE_DEVICES"] = ""        # CPU-only for keras
    env["WANDB_MODE"] = "disabled"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # Limit per-process CPU threads so we can run many subprocesses
    # in parallel without thrashing. The ablation_d2 sbr model is
    # small (~55K params); 2 threads per process is plenty.
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")

    t0 = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8", buffering=1) as fh:
        fh.write(f"# command: {' '.join(cmd)}\n")
        fh.write(f"# env KERAS_NS_FIX_EVAL_INFLATION={env['KERAS_NS_FIX_EVAL_INFLATION']}\n")
        fh.write(f"# cwd: {keras_root}\n\n")
        fh.flush()
        proc = subprocess.run(
            cmd, cwd=str(keras_root), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        # On failure, surface the tail of the per-seed log.
        try:
            tail = "\n".join(stdout_log.read_text().splitlines()[-30:])
        except Exception:
            tail = "<could not read stdout log>"
        raise RuntimeError(
            f"keras-ns runner failed (returncode={proc.returncode})\n"
            f"log: {stdout_log}\n"
            f"--- last 30 lines ---\n{tail}")

    # The runner names the CSV ``_ind_log-<run_signature>-<date>-<mrr>-seed_<n>.csv``.
    # main writes to ``log_folder/indiv_runs/``; ijcai writes to its
    # own ``experiments/runs/indiv_runs/`` (we don't override its
    # log_folder to keep the paper repo untouched).
    pattern = (f"_ind_log-{dataset}-{keras_g}-*-seed_{seed}.csv")
    if ijcai_csv_dir is not None:
        candidates = sorted(
            ijcai_csv_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    else:
        candidates = sorted(indiv_runs.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"keras-ns CSV not found "
            f"({'ijcai' if ijcai_csv_dir else 'main'}) for pattern {pattern}")
    metrics = _parse_keras_csv(candidates[-1])
    metrics["wall"] = wall
    return metrics


def _run_keras_seeds_one_variant(
    dataset: str, reasoner: str, paper_grounder: str,
    *, log_folder: Path, seeds: int, epochs: int, eval_fix: bool,
    keras_root: Path,
) -> tuple[dict, list]:
    """Run one keras-ns checkout over ``seeds`` seeds in parallel.

    Each seed launches a CPU subprocess (``CUDA_VISIBLE_DEVICES=""``)
    via :func:`_run_keras_one`, so spawning all of them concurrently
    just multiplies CPU usage. ``OMP_NUM_THREADS=2`` is set in
    ``_run_keras_one`` to keep each subprocess lightweight.

    For the ``ijcai`` backend (the paper repo, untouched), the
    ``FileLogger.exists_experiment`` check reads ``<root>/experiments/
    runs/experiments/experiments.csv`` and short-circuits any
    re-run with a matching ``run_signature``. Clean that file once
    per cell so the seeds we're about to launch actually execute.
    """
    if keras_root.name == "ijcai":
        ijcai_exp_csv = (
            keras_root / "experiments" / "runs"
            / "experiments" / "experiments.csv"
        )
        if ijcai_exp_csv.exists():
            try:
                ijcai_exp_csv.unlink()
            except FileNotFoundError:
                pass
    from concurrent.futures import ThreadPoolExecutor, as_completed
    runs: List[dict] = []
    seed_list = list(range(PAPER_SEED_BASE, PAPER_SEED_BASE + seeds))
    with ThreadPoolExecutor(max_workers=max(1, len(seed_list))) as ex:
        future_to_seed = {
            ex.submit(
                _run_keras_one,
                dataset, reasoner, paper_grounder, s,
                log_folder=log_folder, epochs=epochs, eval_fix=eval_fix,
                keras_root=keras_root,
            ): s
            for s in seed_list
        }
        for f in as_completed(future_to_seed):
            s = future_to_seed[f]
            try:
                r = f.result()
                r["seed"] = s
                runs.append(r)
            except Exception as e:
                runs.append({"seed": s, "error": str(e)})

    valid = [r for r in runs if "error" not in r and r.get("mrr") is not None]
    if valid:
        import statistics
        keys = ("mrr", "h1", "h3", "h10", "wall")
        agg = {}
        for k in keys:
            vals = [r[k] for r in valid if r.get(k) is not None]
            if vals:
                agg[k + "_mean"] = sum(vals) / len(vals)
                agg[k + "_std"] = (
                    statistics.stdev(vals) if len(vals) > 1 else None)
            else:
                agg[k + "_mean"] = None
                agg[k + "_std"] = None
    else:
        agg = {f"{k}_mean": None for k in ("mrr", "h1", "h3", "h10", "wall")}
    return agg, [r for r in runs if "error" in r]


def _is_tail_only(dataset: str) -> bool:
    """True iff the dataset uses TAIL-only corruption at eval time —
    the only configuration where the inflation formula applies. See
    keras-ns ``experiments/update_config.py``: ``'TAIL'`` for
    ``ablation*`` and ``countries*``; ``'HEAD_AND_TAIL'`` otherwise."""
    return any(dataset.startswith(p) for p in ("ablation", "countries"))


def _derive_infl(real_mrr: Optional[float],
                 real_h1: Optional[float],
                 real_h3: Optional[float],
                 real_h10: Optional[float],
                 dataset: str) -> dict:
    """Derive the inflated metrics from the un-inflated ones using the
    bug formula ``infl = (real + 100) / 2`` — valid only for TAIL-only
    datasets. For HEAD+TAIL datasets there is no inflation, so
    ``infl == real``.

    The fixed keras-ns padding (``ns_lib/utils.py: _PADDING_PREDICTION =
    -inf``) makes trivial single-positive entries always rank #1
    correctly, which is what the formula assumes. Residual deviations
    (≤1pp) come from non-deterministic tie-break on queries where the
    reasoner emits ``-FLT_MAX`` for *every* candidate; those are bounded
    and tracked separately.
    """
    if not _is_tail_only(dataset):
        return {"mrr_mean": real_mrr, "h1_mean": real_h1,
                "h3_mean": real_h3, "h10_mean": real_h10}
    def _f(x):
        return None if x is None else (x + 100.0) / 2.0
    return {"mrr_mean": _f(real_mrr), "h1_mean": _f(real_h1),
            "h3_mean": _f(real_h3), "h10_mean": _f(real_h10)}


def _run_keras_seeds(
    dataset: str, reasoner: str, paper_grounder: str,
    *, log_folder: Path, seeds: int, epochs: int,
    backends: dict[str, Path],
) -> dict:
    """Run each configured keras backend over ``seeds`` seeds.

    For every backend in ``backends`` (e.g. ``{"main": ..., "ijcai":
    ...}``) runs the keras-ns subprocess once per seed and aggregates
    per-backend metrics under prefixed keys. The ``main`` backend
    runs with ``KERAS_NS_FIX_EVAL_INFLATION=1`` (un-inflated MRR);
    ``ijcai`` ignores that env var and reproduces the original paper
    eval (inflated for TAIL-only datasets).

    Returns a dict with keys ``keras_{backend}_{metric}_{stat}``,
    e.g. ``keras_main_mrr_mean`` / ``keras_ijcai_mrr_mean``.
    """
    out: dict = {}
    for backend_name, backend_root in backends.items():
        sub = log_folder / backend_name
        agg, errs = _run_keras_seeds_one_variant(
            dataset, reasoner, paper_grounder,
            log_folder=sub, seeds=seeds, epochs=epochs,
            eval_fix=True,                 # only main honors this
            keras_root=backend_root,
        )
        for k, v in agg.items():
            out[f"keras_{backend_name}_{k}"] = v
        out[f"keras_{backend_name}_errors"] = errs
    return out


def run_one(
    dataset: str, reasoner: str, grounder: Optional[str],
    *, epochs: int, seeds: int,
    do_profile: bool, profile_only: bool,
    keras_log_folder: Optional[Path] = None,
    keras_only: bool = False,
    torch_ckpt_root: Optional[Path] = None,
    keras_backends: Optional[dict[str, Path]] = None,
) -> dict:
    """Train on ``seeds`` seeds, aggregate mean ± std, optionally profile.

    If ``keras_log_folder`` is provided, also drive the keras-ns
    runner side-by-side (CPU subprocess) — outputs go to that folder.
    Use ``keras_only=True`` to skip torch-ns and run keras only.
    """
    runs = []
    if not profile_only and not keras_only:
        # Parallel torch: spawn one subprocess per seed sharing the
        # GPU. ablation_d2/d3 sbr/dcr/r2n models are tiny (~55K-115K
        # params); 5 concurrent CUDA contexts comfortably fit in
        # 24 GB. The orchestrator subprocess (this same script) is
        # invoked with ``--torch-cell`` to run a single seed and
        # write metrics to JSON. Sequential fallback if
        # ``TORCH_PARALLEL=0`` (e.g. for debugging).
        from concurrent.futures import ThreadPoolExecutor, as_completed
        seed_list = list(range(PAPER_SEED_BASE, PAPER_SEED_BASE + seeds))
        if os.environ.get("TORCH_PARALLEL", "1") == "0" or len(seed_list) <= 1:
            for s in seed_list:
                cfg = _build_cfg(
                    dataset, reasoner, grounder, seed=s, epochs=epochs,
                    torch_ckpt_root=torch_ckpt_root,
                )
                try:
                    r = _train_one(cfg)
                    r["seed"] = s
                    runs.append(r)
                except Exception as e:
                    runs.append({"seed": s, "error": str(e)})
                _gpu_cleanup()
        else:
            with ThreadPoolExecutor(max_workers=len(seed_list)) as ex:
                future_to_seed = {
                    ex.submit(
                        _torch_seed_subprocess,
                        dataset, reasoner, grounder, s,
                        epochs=epochs, torch_ckpt_root=torch_ckpt_root,
                    ): s for s in seed_list
                }
                for f in as_completed(future_to_seed):
                    s = future_to_seed[f]
                    try:
                        r = f.result()
                        r["seed"] = s
                        runs.append(r)
                    except Exception as e:
                        runs.append({"seed": s, "error": str(e)})

    profile_metrics = {}
    if do_profile or profile_only:
        # Profiling uses a fresh, smaller-epoch cfg so it doesn't run
        # the full 100-epoch loop. Seed=0.
        cfg_prof = _build_cfg(
            dataset, reasoner, grounder, seed=0, epochs=1,
        )
        try:
            profile_metrics = _profile_one(cfg_prof)
        except Exception as e:
            profile_metrics = {"profile_error": str(e)}
        _gpu_cleanup()

    # Aggregate
    valid = [r for r in runs if "error" not in r]
    if valid:
        import statistics
        keys = ("mrr", "h1", "h3", "h10", "wall")
        agg = {}
        for k in keys:
            vals = [r[k] for r in valid]
            agg[k + "_mean"] = sum(vals) / len(vals)
            agg[k + "_std"] = statistics.stdev(vals) if len(vals) > 1 else None
    else:
        agg = {f"{k}_mean": None for k in ("mrr", "h1", "h3", "h10", "wall")}
        agg.update({f"{k}_std": None for k in ("mrr", "h1", "h3", "h10", "wall")})

    # Optional: drive keras-ns alongside (CPU subprocess). Runs keras
    # TWICE per cell: once with KERAS_NS_FIX_EVAL_INFLATION=1
    # (un-inflated → ``Keras`` column) and once with =0 (inflated,
    # paper-equivalent → ``Keras(infl)`` column). Both numbers are
    # measured directly rather than derived; the doubled wall-clock
    # is acceptable because keras runs on CPU and is fast.
    keras_metrics: dict = {}
    if (keras_log_folder is not None
            and keras_backends
            and not _keras_skip(dataset, reasoner, grounder or "BC01")):
        sub = keras_log_folder / f"{dataset}__{reasoner}__{grounder or 'BC01'}"
        sub.mkdir(parents=True, exist_ok=True)
        keras_metrics = _run_keras_seeds(
            dataset, reasoner, grounder or "BC01",
            log_folder=sub, seeds=seeds, epochs=epochs,
            backends=keras_backends,
        )

    return {
        "dataset": dataset, "reasoner": reasoner, "grounder": grounder,
        "seeds_run": [r["seed"] for r in runs],
        "errors": [r for r in runs if "error" in r],
        **agg, **profile_metrics, **keras_metrics,
    }


def _gpu_cleanup() -> None:
    """Same triple-clear pattern as run_3way_sweep — dynamo cache,
    cudagraph trees, allocator."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        try:
            from torch._inductor.cudagraph_trees import reset_cudagraph_trees
            reset_cudagraph_trees()
        except Exception:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────
# Run-bundle + parity report
# ─────────────────────────────────────────────────────────────────────
def _open_run_bundle(experiment_name: Optional[str], run_name: Optional[str]):
    """Tee stdout into output/runs/<experiment>/<run>/stdout.log. Identical
    contract to scripts/run_3way_sweep.py — line-buffered so ``tail -f``
    works while the sweep is running.
    """
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from kge_kernels.runs.context import _TeeStream

    repo_root = Path(__file__).resolve().parents[1]
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d-reasoner-parity")
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = repo_root / "output" / "runs" / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _tee():
        with (run_dir / "stdout.log").open("a", encoding="utf-8", buffering=1) as fh:
            orig_out, orig_err = sys.stdout, sys.stderr
            sys.stdout = _TeeStream(orig_out, fh)
            sys.stderr = _TeeStream(orig_err, fh)
            try:
                yield run_dir
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err

    return _tee()


def _delta(actual: Optional[float], baseline: Optional[float]) -> str:
    if actual is None or baseline is None:
        return "    —"
    d = actual - baseline
    return f"{d:+5.1f}"


def _render_report(rows: List[dict]) -> str:
    """Markdown parity report.

    Columns are dynamic — one ``Keras-{backend}`` column per backend
    actually run in this sweep (e.g. ``Keras-main``, ``Keras-ijcai``).
    The ``Paper`` column shows the published baseline as-is (inflated
    for TAIL-only datasets, raw otherwise).

    Δ columns: ``Torch − Keras-main``, plus ``Keras-{backend} − Paper``
    and ``Torch − Paper`` for per-system comparison against the paper
    baseline.
    """
    # Discover which backends were run (any row that has a
    # ``keras_{name}_mrr_mean`` key).
    backends = []
    for r in rows:
        for k in r.keys():
            if k.startswith("keras_") and k.endswith("_mrr_mean"):
                name = k[len("keras_"):-len("_mrr_mean")]
                if name and name not in backends:
                    backends.append(name)
    # Stable backend order — main first, then ijcai, then any others
    # in discovery order.
    _preferred = ["main", "ijcai"]
    backends.sort(key=lambda n: (
        _preferred.index(n) if n in _preferred else 99 + backends.index(n)
    ))

    headers = ["Dataset", "Reasoner", "Grounder", "Torch"]
    headers += [f"Keras-{n}" for n in backends]
    headers += ["Paper"]
    if backends:
        headers += [f"Δ Torch−Keras-{backends[0]}"]
    for n in backends:
        headers += [f"Δ Keras-{n}−Paper"]
    headers += ["Δ Torch−Paper"]

    seps = ["---"] * 3 + ["---:"] * (len(headers) - 3)

    lines = [
        "# Reasoner parity report",
        "",
        "Source baselines: `docs/reasoner_parity_baselines.md` "
        "(IJCAI '25 paper Table 1 / Table 2 / Figure 5).",
        "",
        "Numerical columns:",
        "* `Torch` — torch-ns MRR (un-inflated by construction).",
    ]
    for n in backends:
        if n == "main":
            lines.append(
                "* `Keras-main` — patched keras-ns "
                "(``KERAS_NS_FIX_EVAL_INFLATION=1``, u=0 grounder, "
                "-inf MRR padding). Un-inflated.")
        elif n == "ijcai":
            lines.append(
                "* `Keras-ijcai` — original IJCAI '25 paper code "
                "(no patches). Inflated for TAIL-only datasets.")
        else:
            lines.append(f"* `Keras-{n}` — keras-ns checkout `{n}`.")
    lines += [
        "* `Paper` — published baseline (inflated for TAIL-only).",
        "",
        "MRR in percent.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(seps) + "|",
    ]

    # Sort rows by dataset → reasoner → grounder so the table is
    # always in canonical order regardless of how cells were
    # scheduled.
    _DS_ORDER = {
        "ablation_d2": 0, "ablation_d3": 1,
        "countries_s2": 2, "countries_s3": 3,
        "family": 4, "wn18rr": 5,
    }
    _RZ_ORDER = {"no_reasoner": 0, "sbr": 1, "dcr": 2, "r2n": 3}
    _GR_ORDER = {None: 0, "BC01": 1, "BC12": 2, "BC13": 3}

    def _row_key(r):
        return (
            _DS_ORDER.get(r["dataset"], 99),
            _RZ_ORDER.get(r["reasoner"], 99),
            _GR_ORDER.get(r["grounder"], 99),
        )

    def _fmt(v):
        return f"{v:.1f}" if v is not None else "—"

    for r in sorted(rows, key=_row_key):
        bl = BASELINES.get((r["dataset"], r["reasoner"], r["grounder"]))
        paper_mrr = bl.mrr_mean if bl is not None else None
        torch_mrr = r.get("mrr_mean")
        keras_mrrs = {
            n: r.get(f"keras_{n}_mrr_mean") for n in backends
        }

        cells = [
            r["dataset"], r["reasoner"], r["grounder"] or "—",
            _fmt(torch_mrr),
        ]
        for n in backends:
            cells.append(_fmt(keras_mrrs[n]))
        cells.append(_fmt(paper_mrr))
        if backends:
            cells.append(_delta(torch_mrr, keras_mrrs[backends[0]]))
        for n in backends:
            cells.append(_delta(keras_mrrs[n], paper_mrr))
        cells.append(_delta(torch_mrr, paper_mrr))
        lines.append("| " + " | ".join(cells) + " |")
    # End of table.

    if any("ms_batch_train" in r for r in rows):
        lines += [
            "",
            "## Per-batch profiling",
            "",
            "| Dataset | Reasoner | Grounder | train ms/batch | train fwd ms | train peak MB | eval ms/batch | eval fwd ms | eval peak MB |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if "ms_batch_train" not in r:
                continue
            f = lambda k: f"{r[k]:.2f}" if r.get(k) is not None else "—"
            lines.append(
                f"| {r['dataset']} | {r['reasoner']} | {r['grounder'] or '—'} | "
                f"{f('ms_batch_train')} | {f('fwd_gpu_ms_batch')} | "
                f"{f('peak_mem_train_mb')} | {f('ms_batch_eval')} | "
                f"{f('fwd_gpu_ms_eval')} | {f('peak_mem_eval_mb')} |"
            )

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only include datasets matching these substrings.")
    parser.add_argument("--reasoner", nargs="*", default=None,
                        help="Only run these reasoners (sbr / dcr / r2n / no_reasoner).")
    parser.add_argument("--grounder", nargs="*", default=None,
                        help="Only run these paper grounders (BC01 / BC11 / BC12 / BC13).")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Override per-dataset seed count.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (default 100; smoke=3).")
    parser.add_argument("--smoke", action="store_true",
                        help="3-epoch quick run.")
    parser.add_argument("--profile", action="store_true",
                        help="Also run profile_train + profile_eval per config.")
    parser.add_argument("--profile_only", action="store_true",
                        help="Skip full training; only run profilers.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--keras", action="store_true",
        help="Also drive each configured keras backend per cell "
             "(CPU subprocess). Default backend set: ``main`` only. "
             "Use ``--keras-backends`` to override.")
    parser.add_argument(
        "--keras_only", action="store_true",
        help="Skip torch-ns runs and only drive keras backends. "
             "Implies ``--keras``.")
    parser.add_argument(
        "--keras-backends", nargs='+', default=["main"],
        choices=list(KERAS_BACKENDS.keys()),
        help="Which keras-ns checkouts to drive per cell. Each one "
             "produces its own ``Keras-{name}`` column in the report. "
             "``main`` is the patched copy (un-inflated eval, u=0); "
             "``ijcai`` is the original IJCAI '25 paper code "
             "(inflated eval, default ``u``).")
    # Internal: single-cell torch subprocess mode for parallel seeds.
    parser.add_argument(
        "--torch-cell", default=None,
        help="Internal use only. ``ds:reasoner:grounder:seed``. "
             "Runs one torch training and writes metrics to "
             "``--torch-cell-out``, then exits. Used by run_one to "
             "spawn parallel torch subprocesses on the GPU.")
    parser.add_argument("--torch-cell-out", default=None)
    parser.add_argument("--torch-cell-ckpt-root", default=None)
    args = parser.parse_args()

    # Single-cell mode: build cfg, train, write JSON, exit.
    if args.torch_cell is not None:
        ds, reasoner, gr_str, seed_str = args.torch_cell.split(":")
        grounder = gr_str if gr_str else None
        seed = int(seed_str)
        ckpt_root = (
            Path(args.torch_cell_ckpt_root)
            if args.torch_cell_ckpt_root else None
        )
        cfg = _build_cfg(
            ds, reasoner, grounder, seed=seed, epochs=args.epochs,
            torch_ckpt_root=ckpt_root,
        )
        metrics = _train_one(cfg)
        with open(args.torch_cell_out, "w") as fh:
            json.dump(metrics, fh)
        return 0

    if args.keras_only:
        args.keras = True

    if args.smoke:
        args.epochs = min(args.epochs, 3)

    # Canonical iteration order — show every (dataset, reasoner,
    # grounder) cell regardless of whether the paper reports a
    # baseline for it. The Paper column shows ``—`` when the
    # baseline is missing. ``no_reasoner`` is grounder-free
    # (only BC01 makes sense; the no_reasoner cell uses no rules).
    CANONICAL_DATASETS = [
        "ablation_d2", "ablation_d3",
        "countries_s2", "countries_s3",
        "family", "wn18rr",
    ]
    CANONICAL_REASONERS_ORDER = ["no_reasoner", "sbr", "dcr", "r2n"]
    CANONICAL_GROUNDERS = ["BC01", "BC12", "BC13"]

    selected = []
    for dataset in CANONICAL_DATASETS:
        if dataset not in DATASET_SPECS:
            continue
        if args.only and not any(s in dataset for s in args.only):
            continue
        for reasoner in CANONICAL_REASONERS_ORDER:
            if args.reasoner and reasoner not in args.reasoner:
                continue
            grounders = (
                [None] if reasoner == "no_reasoner"
                else CANONICAL_GROUNDERS
            )
            for grounder in grounders:
                if args.grounder and grounder is not None and \
                        grounder not in args.grounder:
                    continue
                seeds = (
                    args.seeds if args.seeds is not None
                    else DATASET_SPECS[dataset].seeds
                )
                selected.append((dataset, reasoner, grounder, seeds))

    print(f"\n{'='*110}")
    print(f"REASONER PARITY SWEEP — {len(selected)} configs, "
          f"epochs={args.epochs}, "
          f"profile={args.profile or args.profile_only}")
    print(f"{'='*110}\n")

    with _open_run_bundle(args.experiment_name, args.run_name) as run_dir:
        # Snapshot config.
        (run_dir / "config.json").write_text(json.dumps({
            "selected": [list(c) for c in selected],
            "epochs": args.epochs,
            "profile": args.profile,
            "profile_only": args.profile_only,
            "keras": args.keras,
            "keras_only": args.keras_only,
            "data_root": DATA_ROOT,
        }, indent=2) + "\n")

        keras_log_folder: Optional[Path] = None
        if args.keras:
            keras_log_folder = run_dir / "keras_logs"
            keras_log_folder.mkdir(parents=True, exist_ok=True)
            print(f"keras-ns logs → {keras_log_folder}\n", flush=True)

        # torch-ns checkpoints + per-cell run dirs go inside the bundle
        # too. Each (dataset, reasoner, grounder, seed) gets its own
        # subdir under ``torch_runs/``; the ``ModelCheckpoint`` callback
        # writes ``model.safetensors`` there.
        torch_ckpt_root = run_dir / "torch_runs"
        torch_ckpt_root.mkdir(parents=True, exist_ok=True)
        print(f"torch-ns runs → {torch_ckpt_root}\n", flush=True)

        rows = []
        for dataset, reasoner, grounder, seeds in selected:
            label = f"{dataset:14} | {reasoner:12} | {grounder or '—':5}"
            print(f"--- {label} ---", flush=True)
            row = run_one(
                dataset, reasoner, grounder,
                epochs=args.epochs, seeds=seeds,
                do_profile=args.profile,
                profile_only=args.profile_only,
                keras_log_folder=keras_log_folder,
                keras_only=args.keras_only,
                torch_ckpt_root=torch_ckpt_root,
                keras_backends=(
                    {n: KERAS_BACKENDS[n] for n in args.keras_backends}
                    if args.keras else None
                ),
            )
            rows.append(row)
            mrr = row.get("mrr_mean")
            wall = row.get("wall_mean")
            ms_batch = row.get("ms_batch_train")
            mrr_s = "  N/A" if mrr is None else f"{mrr:5.1f}"
            wall_s = "  N/A" if wall is None else f"{wall:5.0f}s"
            ms_s = "  N/A" if ms_batch is None else f"{ms_batch:5.2f}"
            # Per-backend keras summary — discover dynamically so the
            # display matches whatever ``--keras-backends`` selected.
            keras_parts = []
            for k in row.keys():
                if k.startswith("keras_") and k.endswith("_mrr_mean"):
                    name = k[len("keras_"):-len("_mrr_mean")]
                    val = row.get(k)
                    if val is not None:
                        keras_parts.append(f" keras-{name}={val:5.1f}")
            ks_s = "".join(keras_parts)
            print(
                f"    torch={mrr_s}{ks_s} wall={wall_s} train_ms_batch={ms_s}",
                flush=True,
            )
            for err in row.get("errors", []) or []:
                print(f"    ERROR seed={err.get('seed')}: {err.get('error')}", flush=True)

        # Persist manifest + parity report.
        from datetime import datetime, timezone
        (run_dir / "manifest.json").write_text(json.dumps({
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        }, indent=2, default=str) + "\n")
        (run_dir / "parity_report.md").write_text(_render_report(rows))
        print(f"\nParity report: {run_dir / 'parity_report.md'}")


if __name__ == "__main__":
    main()
