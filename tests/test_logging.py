from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kge_kernels.logging import LoggingConfig, RegistryConfig, ReportConfig, run_experiment


@dataclass
class DummyConfig:
    seed: int = 7
    signature: str = "dummy"
    logging: LoggingConfig = field(default_factory=LoggingConfig)


class DummySpec:
    def resolve_config(self, raw_config: Any) -> DummyConfig:
        return raw_config

    def logging_config(self, config: DummyConfig) -> LoggingConfig:
        return config.logging

    def family(self, config: DummyConfig) -> str:
        return "train"

    def signature(self, config: DummyConfig) -> str:
        return config.signature

    def run(self, ctx, config: DummyConfig):
        ctx.log_event("inside_run")
        ctx.log_metrics({"valid/mrr": 0.75}, step=3, split="val")
        ctx.save_model(lambda path: path.write_bytes(b"weights"), saved_as="best", metric_name="valid/mrr", metric_value=0.75)
        if config.logging.report.enabled:
            ctx.write_report("# Report\n\nDone.", metadata={"writer": config.logging.report.writer})
        return {"valid/mrr": 0.75}


def test_run_layout_and_registry(tmp_path: Path):
    cfg = DummyConfig()
    cfg.logging.output.output_root = str(tmp_path)
    cfg.logging.registry = RegistryConfig(enabled=True, promote_on_success=True, model_name="dummy-model")
    cfg.logging.report = ReportConfig(enabled=True, writer="agent")

    summary = run_experiment(cfg, DummySpec())
    assert summary["valid/mrr"] == 0.75

    runs_root = tmp_path / "runs" / "train"
    run_dirs = list(runs_root.rglob("manifest.json"))
    assert len(run_dirs) == 1

    manifest = json.loads(run_dirs[0].read_text())
    assert manifest["run"]["status"] == "completed"

    run_root = run_dirs[0].parent
    assert (run_root / "model" / "model.safetensors").exists()
    assert (run_root / "logs" / "events.jsonl").exists()
    metrics = json.loads((run_root / "logs" / "metrics.json").read_text())
    assert "val" in metrics
    assert len(metrics["val"]) == 1
    assert metrics["val"][0]["valid/mrr"] == 0.75
    assert metrics["val"][0]["step"] == 3
    assert (tmp_path / "reports" / "train").exists()
    assert list((tmp_path / "reports" / "train").rglob("report.md"))

    registry_model = tmp_path / "registry" / "dummy-model" / "v001" / "model.safetensors"
    assert registry_model.exists()
    assert (tmp_path / "registry" / "dummy-model" / "v001" / "metrics.json").exists()
    assert (tmp_path / "registry" / "dummy-model" / "v001" / "stdout.log").exists()
