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
    manual_promote: bool = False
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
        if config.manual_promote:
            promoted = ctx.promote_model()
            if promoted is not None:
                ctx.log_event("model_promoted", registry_path=str(promoted))
        return {"valid/mrr": 0.75}


def test_run_layout_and_registry(tmp_path: Path):
    cfg = DummyConfig()
    cfg.logging.output.output_root = str(tmp_path)

    summary = run_experiment(cfg, DummySpec())
    assert summary["valid/mrr"] == 0.75

    runs_root = tmp_path / "runs" / "train"
    run_dirs = list(runs_root.rglob("manifest.json"))
    assert len(run_dirs) == 1

    manifest = json.loads(run_dirs[0].read_text())
    assert manifest["run"]["status"] == "completed"

    run_root = run_dirs[0].parent
    assert (run_root / "config.json").exists()
    assert (run_root / "stdout.log").exists()
    assert (run_root / "events.jsonl").exists()
    assert (run_root / "metrics.json").exists()
    assert (run_root / "model.safetensors").exists()
    assert (run_root / "model_info.json").exists()
    metrics = json.loads((run_root / "metrics.json").read_text())
    assert "val" in metrics
    assert len(metrics["val"]) == 1
    assert metrics["val"][0]["valid/mrr"] == 0.75
    assert metrics["val"][0]["step"] == 3
    assert (tmp_path / "runs" / "train" / "campaign.json").exists()
    assert not (tmp_path / "runs" / "train" / "report.md").exists()
    assert not (tmp_path / "registry" / "train" / run_root.name).exists()


def test_optional_report_and_manual_promotion(tmp_path: Path):
    cfg = DummyConfig(manual_promote=True)
    cfg.logging.output.output_root = str(tmp_path)
    cfg.logging.registry = RegistryConfig(enabled=True, promote_on_success=False, model_name="dummy-model")
    cfg.logging.report = ReportConfig(enabled=True, writer="agent")

    run_experiment(cfg, DummySpec())

    run_dirs = list((tmp_path / "runs" / "train").rglob("manifest.json"))
    assert len(run_dirs) == 1
    run_root = run_dirs[0].parent

    assert (tmp_path / "runs" / "train" / "report.md").exists()
    registry_root = tmp_path / "registry" / "train" / run_root.name
    assert (registry_root / "model.safetensors").exists()
    assert (registry_root / "config.json").exists()
    assert (registry_root / "metrics.json").exists()
    assert (registry_root / "stdout.log").exists()
    assert (registry_root / "promotion.json").exists()
