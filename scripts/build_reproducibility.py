"""Aggregate every (dataset × reasoner × grounder × backend) test result we have on disk
into a single reproducibility.json + a Markdown summary table.

Backends scanned:
- torch-ns      : stdout.log + model_info.json + config.json
- keras-main    : experiments.csv + manifest.json
- keras-ijcai   : indiv_runs/_ind_log-...csv

For each cell we keep the LATEST run (by filesystem timestamp). Output:
- docs/reproducibility.json (the fixed file to test against)
- docs/reproducibility.md   (human-readable table)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/home/castellanoontiv/repos/torch-kge-kernels-swarm/tkk-consolidation")
RUNS_ROOT = REPO_ROOT / "output" / "runs"
IJCAI_RUNS = Path("/home/castellanoontiv/repos/keras-ns-swarm/ijcai/experiments/runs")
DOCS = REPO_ROOT / "docs"

PAPER = {
    ("ablation_d2", "no_reasoner", None):  98.4,
    ("ablation_d2", "sbr", "BC01"):  32.2, ("ablation_d2", "sbr", "BC12"):  96.8, ("ablation_d2", "sbr", "BC13"):  97.4,
    ("ablation_d2", "dcr", "BC01"):  33.8, ("ablation_d2", "dcr", "BC12"):  94.0, ("ablation_d2", "dcr", "BC13"):  95.0,
    ("ablation_d2", "r2n", "BC11"):  71.0, ("ablation_d2", "r2n", "BC12"):  97.2, ("ablation_d2", "r2n", "BC13"):  98.0,
    ("ablation_d3", "no_reasoner", None):  94.8,
    ("ablation_d3", "sbr", "BC01"):  34.8, ("ablation_d3", "sbr", "BC12"):  32.0, ("ablation_d3", "sbr", "BC13"):  86.8,
    ("ablation_d3", "dcr", "BC01"):  34.0, ("ablation_d3", "dcr", "BC12"):  50.0, ("ablation_d3", "dcr", "BC13"):  86.7,
    ("ablation_d3", "r2n", "BC01"):  71.0, ("ablation_d3", "r2n", "BC12"):  74.0, ("ablation_d3", "r2n", "BC13"):  96.6,
    ("countries_s2", "no_reasoner", None): 98.5,
    ("countries_s2", "sbr", "BC01"): 99.5, ("countries_s2", "sbr", "BC12"): 99.5, ("countries_s2", "sbr", "BC13"): 99.5,
    ("countries_s2", "dcr", "BC01"): 99.5, ("countries_s2", "dcr", "BC12"): 99.0, ("countries_s2", "dcr", "BC13"): 97.0,
    ("countries_s2", "r2n", "BC01"): 99.0, ("countries_s2", "r2n", "BC12"): 99.0, ("countries_s2", "r2n", "BC13"): 99.0,
    ("countries_s3", "no_reasoner", None): 88.4,
    ("countries_s3", "sbr", "BC01"): 95.3, ("countries_s3", "sbr", "BC12"): 96.8, ("countries_s3", "sbr", "BC13"): 97.7,
    ("countries_s3", "dcr", "BC01"): 93.5, ("countries_s3", "dcr", "BC12"): 96.9, ("countries_s3", "dcr", "BC13"): 97.6,
    ("countries_s3", "r2n", "BC01"): 90.7, ("countries_s3", "r2n", "BC12"): 88.9, ("countries_s3", "r2n", "BC13"): 89.5,
    ("family", "no_reasoner", None): 85.9,
    ("family", "sbr", "BC01"): 86.9, ("family", "sbr", "BC12"): 87.7,
    ("family", "dcr", "BC01"): 90.1, ("family", "dcr", "BC12"): 90.1,
    ("family", "r2n", "BC01"): 94.0, ("family", "r2n", "BC12"): 91.8,
    ("wn18rr", "no_reasoner", None): 42.7,
    ("wn18rr", "sbr", "BC01"): 44.0, ("wn18rr", "sbr", "BC12"): 44.7,
    ("wn18rr", "dcr", "BC01"): 44.2, ("wn18rr", "dcr", "BC12"): 45.6,
    ("wn18rr", "r2n", "BC01"): 44.2, ("wn18rr", "r2n", "BC12"): 44.1,
}
PAPER_WALL = {
    # (dataset, reasoner, grounder): seconds (training + test) from paper
    ("family", "sbr", "BC01"): 9067 + 6209, ("family", "sbr", "BC12"): 43355 + 27448,
    ("family", "dcr", "BC01"): 16480 + 7659, ("family", "dcr", "BC12"): 16295 + 7517,
    ("family", "r2n", "BC01"): 9573 + 6616, ("family", "r2n", "BC12"): 48809 + 28249,
    ("wn18rr", "sbr", "BC01"): 21941 + 1910, ("wn18rr", "sbr", "BC12"): 67852 + 6666,
    ("wn18rr", "dcr", "BC01"): 26133 + 2338, ("wn18rr", "dcr", "BC12"): 74627 + 6944,
    ("wn18rr", "r2n", "BC01"): 20614 + 2183, ("wn18rr", "r2n", "BC12"): 72213 + 7353,
}
INFLATED_DATASETS = {"ablation_d2", "ablation_d3", "countries_s2", "countries_s3"}
KERAS_GROUNDER_MAP = {"backward_0_1": "BC01", "backward_1_2": "BC12", "backward_1_3": "BC13", "backward_1_1": "BC11"}


def paper_real(dataset: str, infl: Optional[float]) -> Optional[float]:
    if infl is None:
        return None
    if dataset in INFLATED_DATASETS:
        return max(0.0, 2 * infl - 100)
    return infl


def ts_iso(ts: str) -> str:
    """Convert YYYYMMDDTHHMMSS → ISO 8601."""
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%S").isoformat()
    except ValueError:
        return ts


# ───────────────────────────────────────────────────────────────────────────
# torch-ns parser
# ───────────────────────────────────────────────────────────────────────────

CELL_HEADER = re.compile(r"^---\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+---", re.MULTILINE)
RESULT_RE = re.compile(r"torch=\s*([0-9.]+)\s+wall=\s*([0-9]+)s")
TEST_BLOCK = re.compile(
    r"Test Results:\s*\n\s*MRR:\s*([0-9.]+)\s*\n\s*Hits@1:\s*([0-9.]+)\s*\n\s*Hits@3:\s*([0-9.]+)\s*\n\s*Hits@10:\s*([0-9.]+)",
    re.MULTILINE,
)
EXH_BLOCK = re.compile(
    r"Exhaustive test \(k=None.*?\):\s*\n\s*MRR:\s*([0-9.]+)\s*\n\s*Hits@1:\s*([0-9.]+)\s*\n\s*Hits@3:\s*([0-9.]+)\s*\n\s*Hits@10:\s*([0-9.]+)",
    re.MULTILINE,
)
EXH_TIME_RE = re.compile(r"Exhaustive inference time:\s*([0-9.]+)s")
TRAIN_TIME_RE = re.compile(r"Training completed in ([0-9.]+)s")
SAMPLED_INF_RE = re.compile(r"Inference time:\s*([0-9.]+)s")


def parse_torch_stdout(stdout_path: Path) -> list[dict]:
    text = stdout_path.read_text(errors="ignore")
    headers = list(CELL_HEADER.finditer(text))
    rows = []
    for i, h in enumerate(headers):
        body_start = h.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]
        cell = {
            "dataset": h.group(1).strip(),
            "reasoner": h.group(2).strip(),
            "grounder": h.group(3).strip(),
        }
        if (m := RESULT_RE.search(body)):
            cell["wall_seconds"] = int(m.group(2))
        if (m := TRAIN_TIME_RE.search(body)):
            cell["train_seconds"] = float(m.group(1))
        if (m := SAMPLED_INF_RE.search(body)):
            cell["test_inference_seconds"] = float(m.group(1))
        if (m := TEST_BLOCK.search(body)):
            cell["sampled_mrr"] = float(m.group(1))
            cell["sampled_h1"] = float(m.group(2))
            cell["sampled_h3"] = float(m.group(3))
            cell["sampled_h10"] = float(m.group(4))
        if (m := EXH_BLOCK.search(body)):
            cell["exhaustive_mrr"] = float(m.group(1))
            cell["exhaustive_h1"] = float(m.group(2))
            cell["exhaustive_h3"] = float(m.group(3))
            cell["exhaustive_h10"] = float(m.group(4))
        if (m := EXH_TIME_RE.search(body)):
            cell["exhaustive_inference_seconds"] = float(m.group(1))
        rows.append(cell)
    return rows


def scan_torch() -> list[dict]:
    """Walk output/runs/<exp>/<ts>/ and emit one record per (dataset, rsn, gnd) cell."""
    records = []
    for exp_dir in sorted(RUNS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            stdout = ts_dir / "stdout.log"
            if not stdout.exists():
                continue
            cells = parse_torch_stdout(stdout)
            run_dt_iso = ts_iso(ts_dir.name)
            for c in cells:
                cell_dir = ts_dir / "torch_runs" / f"{c['dataset']}__{c['reasoner']}__{c['grounder']}__seed0"
                ckpt = cell_dir / "model.pt"
                model_info = cell_dir / "model_info.json"
                cfg = ts_dir / "config.json"
                rec = {
                    "backend": "torch-ns",
                    "experiment": exp_dir.name,
                    "run_timestamp": run_dt_iso,
                    "seed": 0,
                    **c,
                    "checkpoint": str(ckpt) if ckpt.exists() else None,
                    "model_info_path": str(model_info) if model_info.exists() else None,
                    "config_path": str(cfg) if cfg.exists() else None,
                    "stdout_log": str(stdout),
                }
                # If model_info exists, attach best_val + best_epoch
                if model_info.exists():
                    try:
                        mi = json.loads(model_info.read_text())
                        rec["best_val_mrr"] = mi.get("best_value")
                        rec["best_epoch"] = mi.get("best_epoch")
                    except Exception:
                        pass
                # Only emit if cell has SOME final metric
                if rec.get("sampled_mrr") is not None or rec.get("exhaustive_mrr") is not None:
                    records.append(rec)
    return records


# ───────────────────────────────────────────────────────────────────────────
# keras parser (works for keras-main runs we launched via parity-sweep + ijcai)
# ───────────────────────────────────────────────────────────────────────────


def parse_keras_experiments_csv(csv_path: Path) -> list[dict]:
    """Parse a keras experiments/experiments.csv. Returns a LIST of dicts —
    one per row (i.e. one per run/seed).
    """
    text = csv_path.read_text(errors="ignore")
    lines = text.splitlines()
    if not lines:
        return []
    sep = ";"
    start = 0
    if lines[0].startswith("sep="):
        sep = lines[0].split("=", 1)[1].strip()
        start = 1
    if len(lines) <= start + 1:
        return []
    headers = lines[start].split(sep)
    rows = []
    for line in lines[start + 1:]:
        values = line.split(sep)
        if len(values) >= len(headers) // 2:  # tolerate trailing-column rows
            rows.append(dict(zip(headers, values)))
    return rows


def f(v):
    """Coerce a value to float. Handles list-formatted '[mean, std]' strings
    (keras experiments.csv aggregates one row per (sig, seed) into [mean, std])."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    # Try to parse as a [x, y] list — keras aggregator wraps single-seed values too
    s = str(v).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
        first = s.split(",")[0].strip()
        # Handle 'np.float64(0.919)' or 'np.float32(0.919)'
        m = re.match(r"np\.(float\d+|int\d+)\((-?[0-9.eE+-]+)\)", first)
        if m:
            try:
                return float(m.group(2))
            except (TypeError, ValueError):
                return None
        try:
            return float(first)
        except (TypeError, ValueError):
            return None
    # Handle bare 'np.float64(...)' values
    m = re.match(r"np\.(float\d+|int\d+)\((-?[0-9.eE+-]+)\)", s)
    if m:
        try:
            return float(m.group(2))
        except (TypeError, ValueError):
            return None
    return None


def scan_keras_row(d: dict, csv_path: Path, backend: str) -> Optional[dict]:
    """Build a record from one row of experiments/experiments.csv."""
    if not d:
        return None
    dataset = d.get("dataset_name") or ""
    if dataset == "kinship_family":
        dataset = "family"
    reasoner = d.get("model_name")
    grounder_raw = d.get("grounder")
    grounder = KERAS_GROUNDER_MAP.get(grounder_raw, grounder_raw)
    if not dataset or not reasoner or grounder is None:
        return None
    test_mrr = f(d.get("test_task_mrr"))
    if test_mrr is None:
        return None
    rec = {
        "backend": backend,
        "experiment": csv_path.parts[-5] if len(csv_path.parts) >= 5 else csv_path.parent.name,
        "run_timestamp": datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "seed": int(f(d.get("seed", "0")) or 0),
        "dataset": dataset,
        "reasoner": reasoner,
        "grounder": grounder,
        "experiments_csv": str(csv_path),
        "ckpt_folder": d.get("ckpt_folder"),
        "rules_file": d.get("rules_file"),
        "test_negatives": d.get("test_negatives"),
        "valid_negatives": d.get("valid_negatives"),
        "valid_size": d.get("valid_size"),
        "train_seconds": f(d.get("time_train")),
        "test_inference_seconds": f(d.get("time_inference")),
        "test_task_mrr": test_mrr,
        "test_task_h1": f(d.get("test_task_hits@1")),
        "test_task_h3": f(d.get("test_task_hits@3")),
        "test_task_h10": f(d.get("test_task_hits@10")),
        "valid_task_mrr": f(d.get("valid_task_mrr")),
    }
    # If test_negatives is None or "" → exhaustive
    is_exh = d.get("test_negatives") in ("None", "", None, "0")
    if is_exh:
        rec["exhaustive_mrr"] = rec["test_task_mrr"]
        rec["exhaustive_h1"] = rec["test_task_h1"]
        rec["exhaustive_h3"] = rec["test_task_h3"]
        rec["exhaustive_h10"] = rec["test_task_h10"]
    else:
        rec["sampled_mrr"] = rec["test_task_mrr"]
        rec["sampled_h1"] = rec["test_task_h1"]
        rec["sampled_h3"] = rec["test_task_h3"]
        rec["sampled_h10"] = rec["test_task_h10"]
    # wall_seconds = train + test inference (approximation)
    if rec["train_seconds"] is not None and rec["test_inference_seconds"] is not None:
        rec["wall_seconds"] = int(rec["train_seconds"] + rec["test_inference_seconds"])
    return rec


def scan_keras() -> list[dict]:
    """Pull keras results from experiments/experiments.csv files (one row per run).

    Paths:
    - output/runs/.../experiments/experiments.csv  → keras-main (path contains '/main/'
      subdir typically; we classify by csv path heuristics below)
    - ~/repos/keras-ns-swarm/ijcai/experiments/runs/experiments/experiments.csv → keras-ijcai
    """
    records = []
    # keras-main: any experiments/experiments.csv inside output/runs/
    for csv_path in RUNS_ROOT.rglob("experiments/experiments.csv"):
        backend = "keras-ijcai" if "/ijcai/" in str(csv_path) else "keras-main"
        for row in parse_keras_experiments_csv(csv_path):
            rec = scan_keras_row(row, csv_path, backend)
            if rec:
                records.append(rec)
    # keras-ijcai (in the ijcai repo)
    ijcai_csv = IJCAI_RUNS / "experiments" / "experiments.csv"
    if ijcai_csv.exists():
        for row in parse_keras_experiments_csv(ijcai_csv):
            rec = scan_keras_row(row, ijcai_csv, "keras-ijcai")
            if rec:
                records.append(rec)
    return records


# ───────────────────────────────────────────────────────────────────────────
# Aggregator + report
# ───────────────────────────────────────────────────────────────────────────


def latest_per_key(records: list[dict]) -> dict[tuple, dict]:
    """Pick latest by run_timestamp per (dataset, reasoner, grounder, backend)."""
    latest: dict[tuple, dict] = {}
    for r in records:
        key = (r["dataset"], r["reasoner"], r["grounder"], r["backend"])
        if key not in latest or r["run_timestamp"] > latest[key]["run_timestamp"]:
            latest[key] = r
    return latest


def attach_paper(rec: dict):
    dataset, reasoner, grounder = rec["dataset"], rec["reasoner"], rec["grounder"]
    paper = PAPER.get((dataset, reasoner, grounder))
    rec["paper_mrr"] = paper
    rec["paper_real_mrr"] = paper_real(dataset, paper)
    rec["inflated_paper"] = dataset in INFLATED_DATASETS
    primary = rec.get("exhaustive_mrr") or rec.get("sampled_mrr")
    if primary is not None:
        rec["primary_mrr_pct"] = round(primary * 100, 2)
        rec["primary_metric_kind"] = "exhaustive" if "exhaustive_mrr" in rec and rec["exhaustive_mrr"] is not None else "sampled"
        if rec["paper_real_mrr"] is not None:
            rec["delta_vs_paper_real_pp"] = round(rec["primary_mrr_pct"] - rec["paper_real_mrr"], 2)
    paper_wall = PAPER_WALL.get((dataset, reasoner, grounder))
    if paper_wall:
        rec["paper_wall_seconds"] = paper_wall


def main():
    print("Scanning torch-ns runs ...", flush=True)
    torch_recs = scan_torch()
    print(f"  → {len(torch_recs)} torch cell-runs")

    print("Scanning keras runs ...", flush=True)
    keras_recs = scan_keras()
    print(f"  → {len(keras_recs)} keras cell-runs")

    all_recs = torch_recs + keras_recs
    latest = latest_per_key(all_recs)
    for k, rec in latest.items():
        attach_paper(rec)

    # Sort: (dataset, reasoner, grounder, backend)
    backend_order = ["torch-ns", "keras-main", "keras-ijcai"]
    def sort_key(k):
        d, r, g, b = k
        try:
            bi = backend_order.index(b)
        except ValueError:
            bi = 99
        return (d, r, g or "", bi)

    sorted_keys = sorted(latest.keys(), key=sort_key)

    DOCS.mkdir(exist_ok=True)
    out_json = DOCS / "reproducibility.json"

    # Build the output structure: a flat list of records
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper_source": "IJCAI'25 baselines from BASELINES table in tkk:scripts/run_reasoner_parity_sweep.py",
        "inflation_note": (
            "ablation_d{2,3} and countries_s{2,3} are TAIL-only-corruption datasets; "
            "paper baselines are inflated by the keras-ns main KGCEvalDataset bug. "
            "paper_real_mrr = max(0, 2*paper_mrr - 100). "
            "family + wn18rr are HEAD+TAIL → paper_real_mrr == paper_mrr."
        ),
        "torch_protocol": "1 seed, ComplEx, BCE loss, 100 epochs, early stopping patience=50. Eval: exhaustive over relation domain for family + countries + ablation; 1000 sampled negatives for wn18rr.",
        "keras_protocol": "1 seed each, ComplEx. keras-main launched here with --test_negatives None (exhaustive). keras-ijcai uses ijcai-runner defaults (exhaustive for family, 1000 sampled for wn18rr).",
        "cells": [latest[k] for k in sorted_keys],
    }
    out_json.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote: {out_json}")

    # Build a tight markdown summary
    md_lines = []
    md_lines.append("# Reproducibility — multi-backend reasoner parity\n")
    md_lines.append(f"_Generated: {output['generated_at']}_\n")
    md_lines.append("\n**Paper source**: " + output["paper_source"] + "\n")
    md_lines.append("\n**Inflation note**: " + output["inflation_note"] + "\n")
    md_lines.append("\n**Source of truth**: `docs/reproducibility.json` — one record per (dataset, reasoner, grounder, backend) keeping the LATEST run on disk. Each record contains the checkpoint path, stdout log, config path, MRR/Hits@k, wall time, and run timestamp.\n")
    md_lines.append("\n## Test MRR (%)\n")
    md_lines.append("- Exhaustive eval unless marked `*` (sampled — 100 or 1000 negs per dataset).")
    md_lines.append("- `Paper` column shows the paper-reported number; for TAIL-only datasets the `(real)` value in parens is the un-inflated baseline (`paper(real) = max(0, 2·paper - 100)`).")
    md_lines.append("- `Δ torch` is `torch.exh_mrr - paper(real)` (signed). Positive = torch beats paper.\n")
    md_lines.append("| Dataset | Reasoner | Grounder | torch-ns | keras-main | keras-ijcai | Paper (real) | Δ torch |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    # Group by (dataset, reasoner, grounder), one row per cell
    by_cell: dict[tuple, dict[str, dict]] = {}
    for k, rec in latest.items():
        d, r, g, b = k
        by_cell.setdefault((d, r, g), {})[b] = rec
    for cell_key in sorted(by_cell.keys()):
        d, r, g = cell_key
        backends = by_cell[cell_key]
        paper = PAPER.get((d, r, g))
        paper_disp = f"{paper:.1f}" if paper is not None else "—"
        if d in INFLATED_DATASETS and paper is not None:
            real = paper_real(d, paper)
            paper_disp = f"{paper:.1f} ({real:.1f})"
        delta = "—"
        if "torch-ns" in backends and "delta_vs_paper_real_pp" in backends["torch-ns"]:
            delta = f"{backends['torch-ns']['delta_vs_paper_real_pp']:+.2f}"
        # Eval kind flag — annotate columns that ran on SAMPLED instead of exhaustive
        def _annot(rec):
            if rec is None or not rec:
                return "—"
            mrr = rec.get("primary_mrr_pct", "—")
            kind = rec.get("primary_metric_kind", "")
            return f"{mrr}{'*' if kind == 'sampled' else ''}"
        torch_disp = _annot(backends.get("torch-ns"))
        kmain_disp = _annot(backends.get("keras-main"))
        kijcai_disp = _annot(backends.get("keras-ijcai"))
        md_lines.append(f"| {d} | {r} | {g} | {torch_disp} | {kmain_disp} | {kijcai_disp} | {paper_disp} | {delta} |")

    # Timing table (torch only — keras varies wildly with CPU)
    md_lines.append("\n## Wall time (seconds, torch-ns end-to-end vs paper)\n")
    md_lines.append("| Dataset | Reasoner | Grounder | torch wall | paper wall | speedup |")
    md_lines.append("|---|---|---|---:|---:|---:|")
    for cell_key in sorted(by_cell.keys()):
        d, r, g = cell_key
        backends = by_cell[cell_key]
        torch_rec = backends.get("torch-ns", {})
        torch_wall = torch_rec.get("wall_seconds")
        paper_wall = PAPER_WALL.get((d, r, g))
        sp = "—"
        if torch_wall and paper_wall:
            sp = f"{paper_wall/torch_wall:.1f}×"
        md_lines.append(f"| {d} | {r} | {g} | {torch_wall or '—'} | {paper_wall or '—'} | {sp} |")

    out_md = DOCS / "reproducibility.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote: {out_md}")

    # Brief stdout summary
    print("\nLatest cells per backend:")
    counts = {}
    for k in latest:
        counts[k[3]] = counts.get(k[3], 0) + 1
    for b, c in counts.items():
        print(f"  {b}: {c} cells")


if __name__ == "__main__":
    main()
