"""Configuration helpers shared by the policies: dataclass <-> dict conversion and experiment overrides.

Each policy defines its own config dataclasses (policy/sac_dls/config.py, policy/policyik/config.py) on top of
_ConfigMixin, and its own override sections (e.g. env / sac / train). Robot-specific values live in
configs/<robot>.yaml (RobotSpec).
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml


def coerce_value(default: Any, value: Any, where: str) -> Any:
    """Convert a YAML / command-line value to the type of `default` (e.g. YAML reads `1e-4` as a string)."""
    try:
        if isinstance(default, bool):
            if isinstance(value, bool):
                return value
        elif isinstance(default, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                return int(value)
        elif isinstance(default, float):
            if not isinstance(value, bool):
                return float(value)
        elif isinstance(default, tuple):
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            if isinstance(value, (list, tuple)):
                if default:
                    return tuple(coerce_value(default[0], v, where) for v in value)
                return tuple(value)
        elif isinstance(default, str):
            return str(value)
        else:
            return value
    except (TypeError, ValueError):
        pass
    raise ValueError(f"{where}: expected {type(default).__name__}, got {value!r}")


class _ConfigMixin:
    """Dict conversion helpers shared by the config dataclasses."""

    @classmethod
    def _coerced(cls, d: dict[str, Any], strict: bool, section: str) -> dict[str, Any]:
        defaults = {f.name: f.default for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(d) - set(defaults)
        if unknown and strict:
            raise ValueError(f"Unknown {section} config keys {sorted(unknown)}; available: {list(defaults)}")
        return {k: coerce_value(defaults[k], v, f"{section}.{k}") for k, v in d.items() if k in defaults}

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        """Build from a (possibly older) saved config, ignoring unknown fields."""
        return cls(**cls._coerced(d, strict=False, section=cls.__name__))

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any], section: str | None = None):
        """Defaults with `overrides` applied; unknown keys are an error, values are type-converted."""
        return cls(**cls._coerced(overrides, strict=True, section=section or cls.__name__))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------------------------------
# Experiment overrides: YAML file with optional `env:`, `sac:` and `train:` sections, plus `--set`
# ---------------------------------------------------------------------------------------------------


def _check_sections(d: dict[str, Any], source: str, sections: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    unknown = set(d) - set(sections)
    if unknown:
        raise ValueError(f"{source}: unknown sections {sorted(unknown)}; allowed: {list(sections)}")
    out: dict[str, dict[str, Any]] = {}
    for k, v in d.items():
        if v is None:
            continue
        if not isinstance(v, dict):
            raise ValueError(f"{source}: section '{k}' must be a mapping")
        out[k] = dict(v)
    return out


def load_experiment(path: str | Path, sections: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Read an experiment YAML with the given top-level sections (e.g. env / sac / train)."""
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a mapping with sections {list(sections)}")
    return _check_sections(d, str(path), sections)


def parse_set_args(items: list[str] | None, sections: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """Parse ['env.reward_scale=0.3', 'sac.hidden_dims=[256,256]'] into {'env': {...}, 'sac': {...}}."""
    out: dict[str, dict[str, Any]] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot or not name:
            raise ValueError(f"--set expects SECTION.KEY=VALUE (e.g. env.reward_scale=0.3), got {item!r}")
        out.setdefault(section, {})[name] = yaml.safe_load(raw) if raw.strip() else raw
    return _check_sections(out, "--set", sections)


def merge_overrides(*layers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge override dicts section by section; later layers win."""
    out: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for section, values in layer.items():
            out.setdefault(section, {}).update(values)
    return out
