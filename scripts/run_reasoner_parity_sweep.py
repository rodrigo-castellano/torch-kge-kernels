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
    "BC01": "enum.fp_batch.w0.d1",  # depth=1, w=0; preground for ≥2 free-var rules
    "BC11": "enum.fp_batch.d1",     # depth=1, w=1
    "BC12": "enum.fp_batch.w1.d2",  # depth=2, w=1
    "BC13": "enum.fp_batch.w1.d3",  # depth=3, w=1
}


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
):
    """Construct ``NSTrainConfig`` for one parity row."""
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

    cfg = NSTrainConfig(
        dataset_name=dataset, data_path=DATA_ROOT,
        model_name=reasoner, kge="complex", grounder=grounder_str,
        kge_atom_embedding_size=100,
        num_rules=9999,
        reasoner_depth=None,
        resnet=spec.resnet,
        epochs=epochs, batch_size=train_bs,
        learning_rate=0.01, num_negatives=1,
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
        checkpoint_mode="none",
        output_root=os.path.join(NS_REPO, "output"),
        run_signature=f"parity_{dataset}_{reasoner}_{grounder or 'none'}_s{seed}",
        **overrides,
    )
    return cfg


def _train_one(cfg) -> dict:
    """Run a single training+eval and return metrics + wall-clock."""
    from torch_ns.experiment import pipeline as ns_pipeline
    t0 = time.perf_counter()
    train_m, valid_m, test_m, _ = ns_pipeline(cfg)
    wall = time.perf_counter() - t0
    return {
        "mrr":  test_m.get("MRR", 0.0) * 100.0,
        "h1":   test_m.get("Hits@1", 0.0) * 100.0,
        "h3":   test_m.get("Hits@3", 0.0) * 100.0,
        "h10":  test_m.get("Hits@10", 0.0) * 100.0,
        "wall": wall,
    }


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


def run_one(
    dataset: str, reasoner: str, grounder: Optional[str],
    *, epochs: int, seeds: int,
    do_profile: bool, profile_only: bool,
) -> dict:
    """Train on ``seeds`` seeds, aggregate mean ± std, optionally profile."""
    runs = []
    if not profile_only:
        for s in range(PAPER_SEED_BASE, PAPER_SEED_BASE + seeds):
            cfg = _build_cfg(
                dataset, reasoner, grounder, seed=s, epochs=epochs,
            )
            try:
                r = _train_one(cfg)
                r["seed"] = s
                runs.append(r)
            except Exception as e:
                runs.append({"seed": s, "error": str(e)})
            _gpu_cleanup()

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

    return {
        "dataset": dataset, "reasoner": reasoner, "grounder": grounder,
        "seeds_run": [r["seed"] for r in runs],
        "errors": [r for r in runs if "error" in r],
        **agg, **profile_metrics,
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
    """Markdown parity report. Per-row deltas vs IJCAI baselines.

    Three MRR columns per row:

    * ``Paper(infl)`` — published number from keras-ns ``main`` CSVs.
      For TAIL-only datasets this is inflated (``(real+1)/2``).
    * ``Paper(real)`` — un-inflated equivalent via
      :func:`_real_baseline`. For HEAD+TAIL datasets this matches
      ``Paper(infl)``. For TAIL-only datasets this is what
      ``dcr_r2n_with_neural_grounder`` produces, and also what
      torch-ns produces natively.
    * ``Ours``       — torch-ns raw test MRR.

    ΔMRR is taken vs ``Paper(real)`` so the comparison is
    apples-to-apples across all datasets.
    """
    lines = [
        "# Reasoner parity report",
        "",
        "Source baselines: `docs/reasoner_parity_baselines.md` "
        "(IJCAI '25 paper Table 1 / Table 2).",
        "",
        "**Inflation note.** Paper numbers for `ablation_d2`, `ablation_d3`, "
        "`countries_s2`, `countries_s3` were produced by keras-ns `main`, "
        "which inflates TAIL-only MRR as `(real + 1) / 2` "
        "(see `experiments/dataset.py:202-208 KGCEvalDataset.__getitem__` "
        "for the unconditional double-append). The bug is **fixed** in "
        "branch `dcr_r2n_with_neural_grounder` "
        "(`experiments/kge/dataset.py:191-200` — conditional "
        "`if c.head:` / `if c.tail:`), and torch-ns has the equivalent "
        "fix natively. Below: `Paper(infl)` is the published figure; "
        "`Paper(real)` = `2·Paper(infl) - 100` (un-inflated, what "
        "`dcr_r2n_with_neural_grounder` would produce on the same run); "
        "`Ours` is torch-ns raw. Δ = `Ours − Paper(real)`. For "
        "HEAD+TAIL datasets `Paper(real) = Paper(infl)`.",
        "",
        "MRR / Hits@k in percent.",
        "",
        "| Dataset | Reasoner | Grounder | Paper(infl) MRR | Paper(real) MRR | Ours MRR | ΔMRR | Ours H@1 | ΔH@1 | Ours H@3 | ΔH@3 | Ours H@10 | ΔH@10 | Train(s) | Eval(s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        bl = BASELINES.get((r["dataset"], r["reasoner"], r["grounder"]))
        if bl is None:
            continue
        bl_real = _real_baseline(bl, r["dataset"])

        def _fmt(v):
            return f"{v:.1f}" if v is not None else "—"

        if r.get("errors"):
            lines.append(
                f"| {r['dataset']} | {r['reasoner']} | "
                f"{r['grounder'] or '—'} | "
                f"{_fmt(bl.mrr_mean)} | {_fmt(bl_real.mrr_mean)} | "
                f"**ERROR** | — | — | — | — | — | — | — | — | — |"
            )
            continue

        lines.append(
            f"| {r['dataset']} | {r['reasoner']} | "
            f"{r['grounder'] or '—'} | "
            f"{_fmt(bl.mrr_mean)} | {_fmt(bl_real.mrr_mean)} | "
            f"{_fmt(r.get('mrr_mean'))} | {_delta(r.get('mrr_mean'), bl_real.mrr_mean)} | "
            f"{_fmt(r.get('h1_mean'))} | {_delta(r.get('h1_mean'), bl_real.h1)} | "
            f"{_fmt(r.get('h3_mean'))} | {_delta(r.get('h3_mean'), bl_real.h3)} | "
            f"{_fmt(r.get('h10_mean'))} | {_delta(r.get('h10_mean'), bl_real.h10)} | "
            f"{_fmt(r.get('wall_mean'))} | "
            f"{_fmt(r.get('ms_batch_eval'))} ms/batch |"
        )

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
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 3)

    selected = []
    for (dataset, reasoner, grounder), bl in BASELINES.items():
        if args.only and not any(s in dataset for s in args.only):
            continue
        if args.reasoner and reasoner not in args.reasoner:
            continue
        if args.grounder and grounder not in (args.grounder or [grounder]):
            continue
        seeds = args.seeds if args.seeds is not None else DATASET_SPECS[dataset].seeds
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
            "data_root": DATA_ROOT,
        }, indent=2) + "\n")

        rows = []
        for dataset, reasoner, grounder, seeds in selected:
            label = f"{dataset:14} | {reasoner:12} | {grounder or '—':5}"
            print(f"--- {label} ---", flush=True)
            row = run_one(
                dataset, reasoner, grounder,
                epochs=args.epochs, seeds=seeds,
                do_profile=args.profile,
                profile_only=args.profile_only,
            )
            rows.append(row)
            mrr = row.get("mrr_mean")
            wall = row.get("wall_mean")
            ms_batch = row.get("ms_batch_train")
            mrr_s = "  N/A" if mrr is None else f"{mrr:5.1f}"
            wall_s = "  N/A" if wall is None else f"{wall:5.0f}s"
            ms_s = "  N/A" if ms_batch is None else f"{ms_batch:5.2f}"
            print(
                f"    MRR={mrr_s} wall={wall_s} train_ms_batch={ms_s}",
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
