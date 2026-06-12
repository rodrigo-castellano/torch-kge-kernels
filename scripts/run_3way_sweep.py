#!/usr/bin/env python
"""2-way KGE comparison — tkk standalone vs ns KGE-only, in-process.

Two columns:

- ``tkk``: ``kge_kernels.training.pipeline`` — runs the unified
  ``train_epoch`` path; loss is BCE by default, NSSA via ``cfg.loss``.
- ``ns``:  ``torch_ns.train.main`` with ``model_name=no_reasoner``,
  which routes through ``train_loop`` → ``train_epoch`` (the shared
  static-buffer compiled path).

Both paths use the same tkk ``Sampler``, same ``KGEModel`` subclasses,
same ``evaluate()``. The ``ns`` column measures the full framework
overhead on a KGE-only model.
"""
import argparse

# Resolve the torch-ns checkout via its editable install. tkk-consolidation
# renamed the umbrella package from ``experiments``/``ns_lib`` to
# ``torch_ns``; ``find_spec`` here also exists to surface a clean error if
# torch-ns isn't installed in the active env.
import importlib.util
import os
import sys
import time

_spec = importlib.util.find_spec("torch_ns")
if _spec and _spec.submodule_search_locations:
    NS_REPO = os.path.dirname(next(iter(_spec.submodule_search_locations)))
else:
    NS_REPO = os.environ.get(
        "NS_REPO",
        os.path.expanduser("~/repos/torch-ns-swarm/tkk-consolidation"),
    )
    sys.path.insert(0, NS_REPO)

import torch  # noqa: E402  (sys.path setup above)

DATA_ROOT = os.path.expanduser("~/repos/data-swarm/main")
SEED = 0

CONFIGS = [
    # (dataset, model, corrupt_mode, domain_file, epochs, eval_negs)
    # corrupt_mode: "head" | "tail" | "both" (matches tkk's corruption_scheme)
    # Small datasets: exhaustive TAIL eval
    ("ablation_d3",  "complex", "tail", "domain2constants.txt", 300, 0),
    ("ablation_d3",  "rotate",  "tail", "domain2constants.txt", 300, 0),
    ("countries_s3", "complex", "tail", "domain2constants.txt", 300, 0),
    ("countries_s3", "rotate",  "tail", "domain2constants.txt", 300, 0),
    # Family: 500-neg sampled
    ("family",       "complex", "both", None,                   100, 500),
    ("family",       "rotate",  "both", None,                   100, 500),
    # wn18rr: 500-neg sampled
    ("wn18rr",       "complex", "both", None,                   100, 500),
    ("wn18rr",       "rotate",  "both", None,                   100, 500),
]

PAPER = {
    ("ablation_d3",  "complex"): 89.6,
    ("countries_s3", "complex"): 76.8,
    ("family",       "complex"): 85.9,
    ("wn18rr",       "complex"): 42.7,
}


def run_tkk(dataset, model, corrupt_mode, domain_file, epochs, eval_negs):
    from kge_kernels.training import TrainConfig, pipeline
    # corrupt_mode is already in tkk-canonical form ("head"/"tail"/"both").
    scheme = corrupt_mode
    dom = None
    if domain_file:
        candidate = os.path.join(DATA_ROOT, dataset, domain_file)
        dom = candidate if os.path.isfile(candidate) else None
    # ns's runner with ``--kge rotate`` maps to the ns-aligned RotatE
    # variant (rotate_ns) via ``_run_tkk_kge_only``. Match here so the
    # tkk standalone column compares against the same KGE model.
    tkk_model = "rotate_ns" if model == "rotate" else model
    cfg = TrainConfig(
        dataset=dataset, model=tkk_model, dim=200, gamma=12.0,
        lr=0.01, epochs=epochs, batch_size=256, neg_ratio=1, seed=SEED,
        weight_decay=0.0, loss="bce", scheduler="none",
        report_train_mrr=False, compile=False, amp=False,
        valid_eval_every=1, corruption_scheme=scheme,
        eval_num_corruptions=eval_negs, domain_file=dom,
    )
    t0 = time.perf_counter()
    art = pipeline(cfg)
    dt = time.perf_counter() - t0
    m = art.metrics or {}
    return m.get("test_mrr", 0) * 100, dt


def _make_ns_args(dataset, model, corrupt_mode, domain_file, epochs, eval_negs):
    """Build an NSTrainConfig matching the tkk-side cfg used in run_tkk()."""
    from torch_ns.config import NSTrainConfig
    return NSTrainConfig(
        dataset_name=dataset, data_path=DATA_ROOT,
        model_name='no_reasoner', kge=model,
        epochs=epochs, learning_rate=0.01,
        batch_size=256, num_negatives=1,
        lr_sched='none',
        loss='binary_crossentropy',
        corrupt_mode=corrupt_mode,
        test_negatives=eval_negs, valid_negatives=eval_negs,
        test_batch_size=4096, val_batch_size=-1,
        seed=SEED, seed_run_i=SEED,
        no_compile=True,
        domain_file=domain_file,
        valid_frequency=1,
        checkpoint_mode='none',
        output_root=os.path.join(NS_REPO, 'output'),
        early_stopping=False,
        num_rules=0, reasoner_depth=0,  # ensured by normalize_kge_only_args anyway
    )


def run_ns(dataset, model, corrupt_mode, domain_file, epochs, eval_negs):
    from torch_ns.experiment import pipeline as ns_pipeline
    args = _make_ns_args(dataset, model, corrupt_mode, domain_file,
                         epochs, eval_negs)
    t0 = time.perf_counter()
    train_m, valid_m, test_m, _ = ns_pipeline(args)
    dt = time.perf_counter() - t0
    mrr = test_m.get('MRR', 0) * 100
    return mrr, dt


def _open_run_bundle(experiment_name: str | None, run_name: str | None):
    """Create the output/runs/<experiment>/<run>/ directory and tee stdout+stderr
    into its stdout.log. Returns a context manager that restores the streams.

    Follows the shared logging contract (tkk / ns / DpRL): the sweep is one
    experiment; each invocation is one run. stdout.log is line-buffered so
    ``tail -f`` works while the sweep is running.
    """
    import json
    import sys
    from contextlib import contextmanager
    from datetime import datetime, timezone
    from pathlib import Path

    from kge_kernels.runs.context import _TeeStream  # type: ignore

    repo_root = Path(__file__).resolve().parents[1]
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d-3way-kge-sweep")
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = repo_root / "output" / "runs" / experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the config for reproducibility.
    (run_dir / "config.json").write_text(json.dumps({
        "experiment_name": experiment_name,
        "run_name": run_name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "data_root": DATA_ROOT,
        "configs": [list(c) for c in CONFIGS],
    }, indent=2) + "\n")

    @contextmanager
    def _tee():
        # ``buffering=1`` → line-buffered so tail -f is live.
        with (run_dir / "stdout.log").open("a", encoding="utf-8", buffering=1) as handle:
            orig_out, orig_err = sys.stdout, sys.stderr
            sys.stdout = _TeeStream(orig_out, handle)
            sys.stderr = _TeeStream(orig_err, handle)
            try:
                yield run_dir
            finally:
                sys.stdout, sys.stderr = orig_out, orig_err

    return _tee()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Dataset substrings to skip (e.g. 'wn18rr')")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only include datasets matching these substrings")
    parser.add_argument("--kge", nargs="*", default=None,
                        help="Only include KGE models in this list (e.g. complex)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Subdirectory under output/runs/. "
                             "Default: YYYYMMDD-3way-kge-sweep")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Subdirectory under experiment_name. "
                             "Default: timestamp")
    args = parser.parse_args()

    with _open_run_bundle(args.experiment_name, args.run_name) as run_dir:
        _run_sweep(args, run_dir)


def _run_sweep(args, run_dir):
    import json
    from datetime import datetime, timezone

    rows = []
    print(f"\n{'='*110}")
    print(f"3-WAY KGE COMPARISON (in-process) — seed={SEED}  log={run_dir}")
    print(f"{'='*110}\n")

    configs = CONFIGS
    if args.only:
        configs = [c for c in configs if any(s in c[0] for s in args.only)]
    if args.kge:
        configs = [c for c in configs if c[1] in args.kge]
    for ex in args.exclude:
        configs = [c for c in configs if ex not in c[0]]

    for dataset, model, corrupt_mode, domain_file, epochs, eval_negs in configs:
        label = f"{dataset} | {model}"
        print(f"--- {label} ---", flush=True)

        for name, runner in [
            ("tkk", lambda: run_tkk(dataset, model, corrupt_mode, domain_file, epochs, eval_negs)),
            ("ns",  lambda: run_ns(dataset, model, corrupt_mode, domain_file, epochs, eval_negs)),
        ]:
            print(f"  [{name}] ...", end="", flush=True)
            try:
                mrr, dt = runner()
                print(f" MRR={mrr:.1f}% ({dt:.0f}s)")
            except Exception as e:
                mrr, dt = 0.0, 0.0
                print(f" ERROR: {e}")
            rows.append(dict(dataset=dataset, model=model, path=name, mrr=mrr, time=dt))
            # Release GPU memory between sub-runs. Three things to clear:
            #  1. ``torch._dynamo.reset()``                  — dynamo cache
            #  2. ``cudagraph_trees.reset_cudagraph_trees()`` — the
            #     reduce-overhead CUDA-graph pools (this is the BIG one;
            #     without it a single run's capture persists 2-3 GB per
            #     compiled graph, and the 24 sub-runs stack past 24 GB)
            #  3. ``torch.cuda.empty_cache()``                — reserved
            #     but unallocated blocks in the caching allocator
            # With all three, ns family rotate doesn't OOM after tkk's
            # family rotate run left graph pools resident.
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
        print()

    # Print table
    print(f"\n{'='*80}")
    hdr = f"{'Dataset':<15} {'KGE':<8} {'Paper':>6}"
    for _ in ('tkk', 'ns'):
        hdr += f" {'MRR':>8} {'time':>6}"
    print(hdr)
    print("-" * 80)

    datasets_models = []
    for cfg in CONFIGS:
        dm = (cfg[0], cfg[1])
        if dm not in datasets_models:
            datasets_models.append(dm)

    for ds, kge in datasets_models:
        paper = PAPER.get((ds, kge))
        p_str = f"{paper:.1f}" if paper else "—"
        line = f"{ds:<15} {kge:<8} {p_str:>6}"
        for path in ('tkk', 'ns'):
            r = [x for x in rows if x['dataset'] == ds and x['model'] == kge and x['path'] == path]
            if r:
                line += f" {r[0]['mrr']:>7.1f}% {r[0]['time']:>5.0f}s"
            else:
                line += f" {'—':>8} {'—':>6}"
        print(line)

    # Write a compact manifest at run root so the log bundle is self-describing.
    (run_dir / "manifest.json").write_text(json.dumps({
        "status": "completed",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
