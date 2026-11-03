"""Method configuration: IK environment / reward (IKConfig) and SAC hyperparameters (SACConfig).

The dataclasses below are the single source of truth for every default value. Change a default here
only when you mean to change the baseline for all future runs; for experiments, override fields from
an experiment YAML (`train.py --exp configs/exp/<name>.yaml`) or the command line
(`train.py --set env.reward_scale=0.3 sac.gamma=0.97`). The values actually used are stored with
every run (config.json) and checkpoint, so old runs stay reproducible when defaults change.

Robot-specific values live in configs/<robot>.yaml (RobotSpec), not here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ik_rl.obs import DEFAULT_OBS_FEATURES, obs_dim, validate_obs_features

if TYPE_CHECKING:
    from ik_rl.robot import RobotSpec

# Sections of an experiment YAML / `--set` key prefixes
EXP_SECTIONS = ("env", "sac", "train")


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


@dataclass
class IKConfig(_ConfigMixin):
    """Robot-independent IK environment, action and reward configuration."""

    max_episode_steps: int = 65
    max_delta_q: float = 0.18  # rad per step when far from target
    min_scale_ratio: float = 0.012  # ~2.1 mrad fine step near target
    pos_scale_ref: float = 0.12
    rot_scale_ref: float = 0.40

    home_reset_prob: float = 0.25
    chain_reset_prob: float = 0.30

    # Accuracy thresholds
    term_pos_tol: float = 0.002  # 2.0 mm
    term_rot_tol: float = 0.02  # 0.02 rad (~1.15 deg)
    eval_pos_tol: float = 0.005  # 5.0 mm
    eval_rot_tol: float = 0.05  # 0.05 rad (~2.86 deg)

    # Reward weights. Position / rotation errors are normalized so both share one kernel shape:
    #   coarse terms use e_c = d / {pos,rot}_scale_ref, fine terms use e_f = d / term_{pos,rot}_tol.
    reward_scale: float = 0.25
    lin_weight: float = 0.25  # -w * e_c        (global guidance far from target)
    coarse_weight: float = 1.0  # w * exp(-g * e_c) (length = scale_ref / g)
    coarse_gain: float = 2.0
    fine_weight: float = 1.5  # w * exp(-g * e_f) (length = tol / g = 2.5 x tolerance)
    fine_gain: float = 0.4
    success_bonus: float = 1.0  # paid every step while within term_{pos,rot}_tol (both at once)
    action_penalty: float = 0.015  # L2 on executed joint step dq / max_delta_q (goal-independent)
    smooth_penalty: float = 0.04  # L2 on action rate (a - a_prev)
    collision_penalty: float = 1.0

    # Observation feature groups (see ik_rl/obs.py: OBS_FEATURES); stored with checkpoints
    obs_features: tuple[str, ...] = DEFAULT_OBS_FEATURES

    def __post_init__(self) -> None:
        self.obs_features = validate_obs_features(self.obs_features)

    def obs_dim(self, n_joints: int) -> int:
        return obs_dim(n_joints, self.obs_features)

    @classmethod
    def for_robot(cls, robot: RobotSpec, **kwargs: Any) -> IKConfig:
        """Defaults with the robot's `env_overrides` applied (explicit kwargs win)."""
        return cls.from_overrides({**dict(robot.env_overrides), **kwargs}, section="env")


@dataclass
class SACConfig(_ConfigMixin):
    """Hyperparameters for Soft Actor-Critic on 6D Inverse Kinematics.

    train.py sets obs_dim / act_dim from the robot and observation, batch_size from --batch-size, and
    target_entropy to -n_joints unless it is overridden.
    """

    obs_dim: int = 112
    act_dim: int = 7
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.96
    tau: float = 0.005
    init_alpha: float = 0.05
    min_alpha: float = 1e-4
    target_entropy: float = -7.0
    batch_size: int = 1024


# ---------------------------------------------------------------------------------------------------
# Experiment overrides: YAML file with optional `env:`, `sac:` and `train:` sections, plus `--set`
# ---------------------------------------------------------------------------------------------------


def _check_sections(d: dict[str, Any], source: str) -> dict[str, dict[str, Any]]:
    unknown = set(d) - set(EXP_SECTIONS)
    if unknown:
        raise ValueError(f"{source}: unknown sections {sorted(unknown)}; allowed: {list(EXP_SECTIONS)}")
    out: dict[str, dict[str, Any]] = {}
    for k, v in d.items():
        if v is None:
            continue
        if not isinstance(v, dict):
            raise ValueError(f"{source}: section '{k}' must be a mapping")
        out[k] = dict(v)
    return out


def load_experiment(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read an experiment YAML: {'env': {...IKConfig fields}, 'sac': {...}, 'train': {...train.py options}}."""
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a mapping with sections {list(EXP_SECTIONS)}")
    return _check_sections(d, str(path))


def parse_set_args(items: list[str] | None) -> dict[str, dict[str, Any]]:
    """Parse ['env.reward_scale=0.3', 'sac.hidden_dims=[256,256]'] into {'env': {...}, 'sac': {...}}."""
    out: dict[str, dict[str, Any]] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot or not name:
            raise ValueError(f"--set expects SECTION.KEY=VALUE (e.g. env.reward_scale=0.3), got {item!r}")
        out.setdefault(section, {})[name] = yaml.safe_load(raw) if raw.strip() else raw
    return _check_sections(out, "--set")


def merge_overrides(*layers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge override dicts section by section; later layers win."""
    out: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for section, values in layer.items():
            out.setdefault(section, {}).update(values)
    return out
