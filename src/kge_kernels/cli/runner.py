"""Generic CLI driver around :func:`kge_kernels.logging.run_experiment`.

:func:`run_cli` is a 1-call wrapper that gives any consumer repo the
DpRL-style ``--set`` / ``--grid`` CLI surface, parameterized over the
repo's own typed :class:`@dataclass` config type and
:class:`ExperimentSpec` adapter:

  * Reads defaults from ``dataclasses.fields(config_cls)``.
  * Adds ``--set KEY=VALUE`` (override one field) and ``--grid KEY=V1,V2``
    (Cartesian product over fields).
  * Lifts list-valued base fields into the grid automatically (so
    ``dataset_name=['a','b']`` in defaults sweeps both).
  * Expands every (combo × seed) pair into a concrete raw config dict.
  * Calls :func:`run_experiment(raw_config, spec)` per run.

The driver lives here in tkk so the three consumer repos (tkk standalone
KGE-pretrain, torch-ns, DpRL) share one canonical CLI surface and one
upgrade path. tkk doesn't execute the consumer-specific training loops;
it just owns the parser + grid + run-lifecycle helper they all import.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import MISSING, fields, is_dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Optional, Type

from ..logging import ExperimentSpec, run_experiment


BOOLEAN_TRUE = {"true", "t", "yes", "y", "on", "1"}
BOOLEAN_FALSE = {"false", "f", "no", "n", "off", "0"}


def parse_assignment(entry: str) -> tuple[str, str]:
    """Split a ``key=value`` CLI assignment into the two halves."""
    if "=" not in entry:
        raise ValueError(f"Assignments must be in key=value format, got '{entry}'.")
    key, raw = entry.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid assignment '{entry}'.")
    return key, raw.strip()


def parse_scalar(text: str) -> Any:
    """Best-effort literal conversion of a CLI string."""
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
    """Coerce a parsed override to match the type of the default for ``key``."""
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
    """Read defaults out of a dataclass into a dict."""
    out: dict[str, Any] = {}
    for f in fields(config_cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
        else:
            # Required field with no default — leave unset; user must --set it.
            pass
    return out


def build_parser(description: str = "") -> argparse.ArgumentParser:
    """Build the standard tkk CLI parser (--set / --grid / --eval / --profile)."""
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
        help="Optional shortcut: spec may interpret --eval (e.g. timesteps=0).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Optional shortcut: spec may interpret --profile (e.g. enable cprofile).",
    )
    return parser


def expand_run_configs(
    base: dict[str, Any], grid: Mapping[str, list[Any]]
) -> list[dict[str, Any]]:
    """Cartesian-product expand a grid into concrete raw configs."""
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


def run_cli(
    *,
    config_cls: Type,
    spec: ExperimentSpec,
    description: str = "",
    extra_parser_setup: Optional[Any] = None,
    grid_exclude: Iterable[str] = ("seed",),
) -> None:
    """Run the standard tkk CLI driver for a consumer repo.

    Args:
        config_cls: The repo's ``@dataclass`` config type. Defaults are
            read from its fields; types of ``--set`` overrides are
            coerced against those defaults.
        spec: The repo's :class:`ExperimentSpec` instance — receives one
            ``run_experiment`` invocation per (combo × seed) run.
        description: Argparse description string.
        extra_parser_setup: Optional callable ``(parser) -> None`` that
            adds repo-specific CLI flags. The resulting ``args`` is
            passed to ``spec.consume_extra(args, base_config)`` if the
            spec defines that method (otherwise extra flags are ignored).
        grid_exclude: Field names to exclude from auto-lifting list-valued
            base fields into the grid (default: ``"seed"`` — the seed
            field is iterated separately so each grid combo runs every
            seed).

    Mechanics:
        1. Parse CLI: ``--set`` overrides → applied to base; ``--grid``
           builds an explicit grid spec; list-valued base fields (other
           than excluded) are auto-lifted into the grid.
        2. Cartesian-product expand → list of raw config dicts.
        3. For each (config × seed) pair, call
           :func:`run_experiment(raw_config, spec)`.
    """
    parser = build_parser(description=description)
    if extra_parser_setup is not None:
        extra_parser_setup(parser)
    args = parser.parse_args()

    defaults = _defaults_from_dataclass(config_cls)
    base = copy.deepcopy(defaults)

    # Apply --set overrides.
    for entry in args.set:
        key, raw = parse_assignment(entry)
        value = coerce_config_value(key, parse_scalar(raw), defaults)
        _assign(base, key, value)

    # Optional repo hook to consume parser extras (--eval / --profile flags,
    # ``num_seeds`` → seed-list expansion, etc.). Runs ONCE on the base
    # config before grid expansion so it can mutate base-level fields.
    extras_hook = getattr(spec, "consume_extras", None)
    if extras_hook is not None:
        extras_hook(args, base)

    # Build grid: list-valued base entries (auto-lifted) + explicit --grid entries.
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

    # Iterate seeds: pull seed list out of base; default to [base['seed']].
    seed_value = base.get("seed", 0)
    seeds = list(seed_value) if isinstance(seed_value, list) else [seed_value]
    has_seed_run_i = "seed_run_i" in defaults

    for raw_config in run_configs:
        for seed in seeds:
            cfg = copy.deepcopy(raw_config)
            cfg["seed"] = seed
            if has_seed_run_i:
                cfg["seed_run_i"] = seed
            run_experiment(cfg, spec)


__all__ = [
    "BOOLEAN_FALSE",
    "BOOLEAN_TRUE",
    "build_parser",
    "coerce_config_value",
    "expand_run_configs",
    "parse_assignment",
    "parse_scalar",
    "run_cli",
]
