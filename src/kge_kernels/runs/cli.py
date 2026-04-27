"""Unified CLI driver for running experiments.

Consumers write a typed ``@dataclass`` config (``ExpConfig``) and a
``run_experiment(ctx, cfg) -> dict | None`` function. :func:`run_cli`
parses the standard ``--set KEY=VALUE`` / ``--grid KEY=V1,V2`` CLI,
expands a Cartesian-product grid + seed loop, builds a fresh
:class:`RunContext` per run, calls the user's ``run_experiment``
inside ``ctx.stdout_capture()``, then persists the returned metrics
into the canonical run bundle.

Naming conventions:
  * ``ExpConfig`` — user's dataclass; methods ``family()`` /
    ``signature()`` / ``logging_config()`` are duck-typed if present.
  * ``run_experiment(ctx, cfg) → dict[str, dict]`` — the user's work.
    Splits like ``"train"`` / ``"val"`` / ``"test"`` are common; any
    keys are accepted and persisted to ``metrics.json``.
"""
from __future__ import annotations

import argparse
import ast
import copy
import traceback
from dataclasses import MISSING, fields, is_dataclass
from itertools import product
from typing import Any, Callable, Iterable, Mapping, Optional, Type

from .config import LoggingConfig
from .context import RunContext
from .layout import build_run_id, build_run_paths


BOOLEAN_TRUE = {"true", "t", "yes", "y", "on", "1"}
BOOLEAN_FALSE = {"false", "f", "no", "n", "off", "0"}


# ─────────────────────────────────────────────────────────────────────
# CLI parsing helpers (pure)
# ─────────────────────────────────────────────────────────────────────

def parse_assignment(entry: str) -> tuple[str, str]:
    if "=" not in entry:
        raise ValueError(f"Assignments must be in key=value format, got '{entry}'.")
    key, raw = entry.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid assignment '{entry}'.")
    return key, raw.strip()


def parse_scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        lowered = text.lower()
        if lowered in BOOLEAN_TRUE:
            return True
        if lowered in BOOLEAN_FALSE:
            return False
        if lowered in {"none", "null"}:
            return None
    return text


def coerce_config_value(key: str, value: Any, defaults: Mapping[str, Any]) -> Any:
    default = _resolve_default(defaults, key)

    if isinstance(default, list):
        if isinstance(value, (list, tuple)):
            return [copy.deepcopy(v) for v in value]
        return [value]

    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in BOOLEAN_TRUE:
                return True
            if lowered in BOOLEAN_FALSE:
                return False
            raise ValueError(f"Cannot parse boolean for '{key}': {value}")
        return bool(value)

    if isinstance(default, int) and not isinstance(default, bool):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str)):
            return int(value)

    if isinstance(default, float):
        if value is None:
            return None
        if isinstance(value, (int, float, str)):
            return float(value)

    if isinstance(default, str):
        if value is None:
            return None
        return str(value)

    return copy.deepcopy(value)


def _resolve_default(defaults: Mapping[str, Any], key: str) -> Any:
    parts = key.split(".")
    current: Any = defaults
    for part in parts:
        if isinstance(current, Mapping):
            if part not in current:
                raise ValueError(f"Unknown configuration key '{key}'.")
            current = current[part]
            continue
        if is_dataclass(current):
            if not hasattr(current, part):
                raise ValueError(f"Unknown configuration key '{key}'.")
            current = getattr(current, part)
            continue
        raise ValueError(f"Unknown configuration key '{key}'.")
    return current


def _assign(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, Mapping):
            if part not in current:
                raise ValueError(f"Unknown configuration key '{key}'.")
            current = current[part]
            continue
        if is_dataclass(current):
            if not hasattr(current, part):
                raise ValueError(f"Unknown configuration key '{key}'.")
            current = getattr(current, part)
            continue
        raise ValueError(f"Unknown configuration key '{key}'.")
    leaf = parts[-1]
    stored = copy.deepcopy(value) if isinstance(value, (list, dict)) else value
    if isinstance(current, dict):
        current[leaf] = stored
        return
    if hasattr(current, leaf):
        setattr(current, leaf, stored)
        return
    raise ValueError(f"Unknown configuration key '{key}'.")


def _defaults_from_dataclass(config_cls: Type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(config_cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
    return out


def build_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set lr=0.001 --set seed='[0,1]'.",
    )
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Grid sweep, e.g. --grid dataset=family,fb15k237.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Optional shortcut: extras_handler may interpret --eval.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Optional shortcut: extras_handler may interpret --profile.",
    )
    return parser


def expand_run_configs(
    base: dict[str, Any], grid: Mapping[str, list[Any]]
) -> list[dict[str, Any]]:
    if not grid:
        return [base]
    keys = sorted(grid.keys())
    out: list[dict[str, Any]] = []
    for combo in product(*(grid[key] for key in keys)):
        cfg = copy.deepcopy(base)
        for key, value in zip(keys, combo):
            _assign(cfg, key, value)
        out.append(cfg)
    return out


# ─────────────────────────────────────────────────────────────────────
# Per-run lifecycle (was logging/runner.py:run_experiment)
# ─────────────────────────────────────────────────────────────────────

def _resolve_metadata(cfg: Any, *, family_fn, signature_fn, logging_config_fn) -> tuple[str, str, LoggingConfig]:
    """Compute run-bundle metadata (family / signature / logging) from cfg.

    Resolution order: explicit ``*_fn`` override → method on ``cfg`` →
    sensible default. Methods on the config are duck-typed.
    """
    if family_fn is not None:
        family = family_fn(cfg)
    elif hasattr(cfg, "family") and callable(cfg.family):
        family = cfg.family()
    else:
        family = "experiment"

    if signature_fn is not None:
        signature = signature_fn(cfg)
    elif hasattr(cfg, "signature") and callable(cfg.signature):
        signature = cfg.signature()
    else:
        signature = "default"

    if logging_config_fn is not None:
        logging_cfg = logging_config_fn(cfg)
    elif hasattr(cfg, "logging_config") and callable(cfg.logging_config):
        logging_cfg = cfg.logging_config()
    elif hasattr(cfg, "logging") and isinstance(getattr(cfg, "logging"), LoggingConfig):
        logging_cfg = cfg.logging
    else:
        logging_cfg = LoggingConfig()

    return str(family), str(signature), logging_cfg


def _seed_from_config(cfg: Any) -> int:
    if isinstance(cfg, Mapping):
        return int(cfg.get("seed", 0))
    return int(getattr(cfg, "seed", 0))


def run_one(
    raw_config: Any,
    *,
    config_cls: Type,
    run_experiment: Callable[[RunContext, Any], Optional[Mapping[str, Any]]],
    family_fn: Optional[Callable[[Any], str]] = None,
    signature_fn: Optional[Callable[[Any], str]] = None,
    logging_config_fn: Optional[Callable[[Any], LoggingConfig]] = None,
) -> Mapping[str, Any]:
    """Run a single experiment lifecycle.

    Public per-run entry point. :func:`run_cli` is the multi-run CLI
    driver layered on top of this; tests and library callers that want
    one config to one run can call ``run_one`` directly.

    ``raw_config`` may be either a dict / mapping (will be filtered into
    ``config_cls(**kwargs)``) or an existing ``config_cls`` instance
    (used directly).
    """
    if isinstance(raw_config, config_cls):
        cfg = raw_config
    else:
        valid = {f.name for f in fields(config_cls) if f.init}
        cfg = config_cls(**{k: v for k, v in dict(raw_config).items() if k in valid})

    family, signature, logging_cfg = _resolve_metadata(
        cfg,
        family_fn=family_fn,
        signature_fn=signature_fn,
        logging_config_fn=logging_config_fn,
    )
    seed = _seed_from_config(cfg)
    run_id, started_at = build_run_id(signature=signature, seed=seed)
    paths = build_run_paths(
        logging_cfg.output.output_root,
        family=family,
        run_id=run_id,
        model_filename=logging_cfg.model.filename,
        report_filename=logging_cfg.report.filename,
    )
    ctx = RunContext(
        logging=logging_cfg,
        family=family,
        signature=signature,
        seed=seed,
        run_id=run_id,
        started_at=started_at,
        paths=paths,
        resolved_config=cfg,
    )

    try:
        ctx.log_event("run_started")
        with ctx.stdout_capture():
            result = run_experiment(ctx, cfg)
        summary: dict[str, Any] = dict(result) if result else {}
        ctx.log_event("run_completed")
        ctx.finish(status="completed", final_metrics=summary)
        return summary
    except Exception as exc:
        error_summary = {
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        ctx.log_event("run_failed", error=str(exc))
        ctx.finish(status="failed", final_metrics=error_summary, error=str(exc))
        raise


# ─────────────────────────────────────────────────────────────────────
# Multi-run CLI driver — the public entry point
# ─────────────────────────────────────────────────────────────────────

def run_cli(
    *,
    config_cls: Type,
    run_experiment: Callable[[RunContext, Any], Optional[Mapping[str, Any]]],
    description: str = "",
    family_fn: Optional[Callable[[Any], str]] = None,
    signature_fn: Optional[Callable[[Any], str]] = None,
    logging_config_fn: Optional[Callable[[Any], LoggingConfig]] = None,
    extra_parser_setup: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    extras_handler: Optional[Callable[[argparse.Namespace, dict], None]] = None,
    grid_exclude: Iterable[str] = ("seed",),
) -> None:
    """Run the unified CLI for an experiment package.

    Args:
        config_cls: The user's typed ``@dataclass`` config (``ExpConfig``).
            Defaults are read from its fields; ``--set`` overrides are
            type-coerced against those defaults.
        run_experiment: Callable ``(ctx: RunContext, cfg: config_cls) ->
            dict | None`` — the actual work. Returns a metrics dict that
            gets persisted into ``metrics.json`` (typically keyed by
            split: ``{"train": {...}, "val": {...}, "test": {...}}``).
        description: argparse description string.
        family_fn / signature_fn / logging_config_fn: optional overrides
            that compute run-bundle metadata from a constructed config.
            Default: duck-typed methods on the config (``cfg.family()`` /
            ``cfg.signature()`` / ``cfg.logging_config()``); else
            sensible fallbacks (``"experiment"`` / ``"default"`` /
            ``LoggingConfig()``).
        extra_parser_setup: optional callable that adds repo-specific
            CLI flags to the parser before ``parse_args()``.
        extras_handler: optional callable ``(args, base_config: dict) ->
            None`` fired ONCE on the base config (after ``--set``,
            before grid expansion) so it can mutate base-level fields
            from CLI extras (``--eval`` shortcuts, ``num_seeds`` →
            seed-list expansion, etc.).
        grid_exclude: field names that should NOT be auto-lifted into
            the grid even if their base value is a list. Default
            ``("seed",)`` — seeds iterate as a separate loop so each
            grid combo runs every seed.

    Lifecycle per run:
      1. Parse CLI; build base config from defaults; apply ``--set``;
         call ``extras_handler``.
      2. Build grid (auto-lifted lists + ``--grid``); expand to a list
         of raw configs.
      3. For each (combo × seed):
         a. Construct ``cfg = config_cls(**raw_config)``.
         b. Resolve family / signature / logging via the override
            chain.
         c. Build :class:`RunPaths` and :class:`RunContext`.
         d. Inside ``ctx.stdout_capture()``: call
            ``run_experiment(ctx, cfg)``.
         e. Persist the returned metrics + manifest.
    """
    parser = build_parser(description=description)
    if extra_parser_setup is not None:
        extra_parser_setup(parser)
    args = parser.parse_args()

    defaults = _defaults_from_dataclass(config_cls)
    base = copy.deepcopy(defaults)

    for entry in args.set:
        key, raw = parse_assignment(entry)
        value = coerce_config_value(key, parse_scalar(raw), defaults)
        _assign(base, key, value)

    if extras_handler is not None:
        extras_handler(args, base)

    grid: dict[str, list[Any]] = {}
    exclude = set(grid_exclude)
    for key, value in list(base.items()):
        if isinstance(value, list) and key not in exclude:
            grid[key] = value

    for entry in args.grid:
        key, raw_values = parse_assignment(entry)
        candidates = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not candidates:
            raise ValueError(f"No values supplied for grid entry '{entry}'.")
        grid[key] = [
            coerce_config_value(key, parse_scalar(c), defaults) for c in candidates
        ]

    run_configs = expand_run_configs(base, grid)

    seed_value = base.get("seed", 0)
    seeds = list(seed_value) if isinstance(seed_value, list) else [seed_value]
    has_seed_run_i = "seed_run_i" in defaults

    for raw_config in run_configs:
        for seed in seeds:
            cfg_dict = copy.deepcopy(raw_config)
            cfg_dict["seed"] = seed
            if has_seed_run_i:
                cfg_dict["seed_run_i"] = seed
            run_one(
                cfg_dict,
                config_cls=config_cls,
                run_experiment=run_experiment,
                family_fn=family_fn,
                signature_fn=signature_fn,
                logging_config_fn=logging_config_fn,
            )


__all__ = [
    "BOOLEAN_FALSE",
    "BOOLEAN_TRUE",
    "build_parser",
    "coerce_config_value",
    "expand_run_configs",
    "parse_assignment",
    "parse_scalar",
    "run_cli",
    "run_one",
]
