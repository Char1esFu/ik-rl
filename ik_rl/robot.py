"""Robot specification loaded from configs/<robot>.yaml: everything robot-specific the IK pipeline needs."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_ROBOT = "franka"


@dataclass(frozen=True)
class RobotSpec:
    """Robot-specific configuration for the shared IK environment, training and tools.

    Positions are in meters, angles in radians, quaternions in [w, x, y, z].
    See configs/*.yaml for documented examples.
    """

    name: str  # short id used on the command line and in runs/<name>/
    display_name: str
    scene_xml: Path
    arm_joints: tuple[str, ...]  # joints controlled by the policy (in order)
    ee_body: str  # body the TCP site is attached to
    ee_offset: tuple[float, float, float]  # TCP position in ee_body frame
    home_q: tuple[float, ...]
    sample_q_low: tuple[float, ...]  # joint box for sampling targets / start configurations
    sample_q_high: tuple[float, ...]
    workspace_low: tuple[float, float, float]  # TCP box a sampled target must lie in
    workspace_high: tuple[float, float, float]
    ee_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # TCP orientation in ee_body
    fixed_joints: tuple[tuple[str, float], ...] = ()  # non-arm joints held fixed, e.g. gripper fingers
    start_pos_min: tuple[float, float, float] = (0.05, float("-inf"), 0.05)  # TCP lower bound for starts
    home_noise_std: float = 0.25
    camera_lookat: tuple[float, float, float] = (0.32, 0.0, 0.42)
    camera_distance: float = 1.6
    # Viewer figure-8 demo (visualize.py): position center / amplitude, orientation as extrinsic roll-pitch-yaw
    # center / amplitude in degrees. fig8_radial_yaw adds the target's azimuth around the first arm joint's
    # axis to the yaw, keeping the approach direction in the arm's vertical plane (needed for 5-DoF arms).
    fig8_center: tuple[float, float, float] = (0.44, 0.0, 0.43)
    fig8_amplitude: tuple[float, float, float] = (0.08, 0.22, 0.11)
    fig8_rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    fig8_rpy_amplitude_deg: tuple[float, float, float] = (18.0, 15.0, 25.0)
    fig8_radial_yaw: bool = False
    env_overrides: tuple[tuple[str, Any], ...] = ()  # IKConfig field overrides

    @property
    def n_joints(self) -> int:
        return len(self.arm_joints)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotSpec:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown robot config keys: {sorted(unknown)}")
        kw: dict[str, Any] = {}
        for k, v in d.items():
            if k == "scene_xml":
                p = Path(v)
                kw[k] = p if p.is_absolute() else PROJECT_ROOT / p
            elif k in ("fixed_joints", "env_overrides"):
                kw[k] = tuple((str(name), val) for name, val in dict(v or {}).items())
            elif isinstance(v, list):
                kw[k] = tuple(v)
            else:
                kw[k] = v
        spec = cls(**kw)
        n = spec.n_joints
        for k in ("home_q", "sample_q_low", "sample_q_high"):
            if len(getattr(spec, k)) != n:
                raise ValueError(f"{spec.name}: '{k}' must have {n} entries (one per arm joint)")
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotSpec:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-friendly representation (stored with each training run)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "scene_xml":
                try:
                    v = str(Path(v).relative_to(PROJECT_ROOT))
                except ValueError:
                    v = str(v)
            elif f.name in ("fixed_joints", "env_overrides"):
                v = dict(v)
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out


def available_robots() -> list[str]:
    """Robot names with a config file in configs/."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def get_robot(name_or_path: str | Path) -> RobotSpec:
    """Load a robot by name (configs/<name>.yaml) or from a path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml"):
        return RobotSpec.from_yaml(p)
    path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise KeyError(f"Unknown robot '{name_or_path}'. Available: {', '.join(available_robots())}")
    return RobotSpec.from_yaml(path)
