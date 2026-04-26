# CLAUDE.md

## Running Experiments

- Use `python -u` or `PYTHONUNBUFFERED=1` for long runs.
- Do not use `conda run`; it hides live output. Activate the env (or call `$CONDA_PREFIX/bin/python`) directly.

## Logging Experiments

This section is the shared contract for tkk, torch-ns, and DpRL. It is copy-pasted identically across those repos; keep them in sync. tkk owns the framework (`kge_kernels.logging.run_experiment` + `RunContext`); consumers wire it up through an `ExperimentSpec` adapter. Any new training entrypoint that writes to `output/runs/` MUST go through `run_experiment` so every run produces the same bundle.

Output root: `output/` at the repo root for tkk and torch-ns, `kge_experiments/output/` for DpRL (kept nested for historical reasons).

Layout:

- `output/runs/<experiment_name>/<run_name>/` is the canonical run bundle.
- Name `<experiment_name>` as `YYYYMMDD-<topic>` (e.g. `20260414-ablation-d2-5seeds`) so runs sort chronologically and two agents can't silently collide on the same group.
- Stream stdout to a live log file from the first line — not only after the run finishes. `RunContext.stdout_capture()` already line-buffers into `stdout.log`; use it instead of rolling your own log file. A long sweep must be tail-able (`tail -f <log>`) while it runs; document the path you use so users can watch it.
- Each run stores `manifest.json`, `config.json`, `stdout.log`, `events.jsonl`, `metrics.json`, `model.safetensors` (or the config-chosen single model file), `model_info.json`, and optional `artifacts/`.
- `metrics.json` stores split metrics (`train`, `val`, `test`); `events.jsonl` is the lifecycle timeline; `manifest.json` is the final run summary.
- Save exactly one run-local model, controlled by `LoggingConfig.model.mode`: `none`, `last`, or `best`.
- `model_info.json` records the save mode, metric name/value, and global step of the saved model.
- Put auxiliary per-run files — entity/relation id maps, vocabularies, debug dumps, plots — under `artifacts/`, not at the run root. The root is reserved for the canonical bundle files above.
- `output/registry/<experiment_name>/<run_name>/` is a promoted copy of the same run bundle, plus optional promotion metadata.
- `output/legacy/` is for imported historical legacy artifacts only; do not write new runs there.

Rules:

- `runs/` is the source of truth; do not create ad-hoc log directories elsewhere (no top-level checkpoint dirs, no `/tmp/...` save_dirs for real experiments).
- Promote runs by copying the full run bundle into `registry/`; keep the same experiment and run names.
- Promotion into `registry/` is manual only; training code must not auto-promote successful runs.
- Reports are optional and manual; training code must not auto-generate `report.md`.
- Keep legacy imports and archives under `output/legacy/`, not under `runs/`.
