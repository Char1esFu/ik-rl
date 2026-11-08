"""SAC+DLS method configuration: IK environment / reward (IKConfig) and SAC hyperparameters (SACConfig).

The dataclasses below are the single source of truth for every default value. Change a default here
only when you mean to change the baseline for all future runs; for experiments, override fields from
an experiment YAML (`train.py --method sac+dls --exp configs/exp/<name>.yaml`) or the command line
(`train.py --method sac+dls --set env.reward_scale=0.3 sac.gamma=0.97`). The values actually used are stored
with every run (config.json) and checkpoint, so old runs stay reproducible when defaults change.

Robot-specific values live in configs/<robot>.yaml (RobotSpec), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from policy.sac_dls.obs import DEFAULT_OBS_FEATURES, obs_dim, validate_obs_features
from utils.config import _ConfigMixin

if TYPE_CHECKING:
    from core.robot import RobotSpec

# Sections of an experiment YAML / `--set` key prefixes
EXP_SECTIONS = ("env", "sac", "train")


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

    # Observation feature groups (see policy/sac_dls/obs.py: OBS_FEATURES); stored with checkpoints
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
