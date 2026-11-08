"""PolicyIK configuration: environment / observation / reward (PolicyIKConfig) and PPO (PPOConfig).

Defaults follow the simulation recipe of PolicyIK (Liu et al., "PolicyIK: High-Precision Inverse Kinematics
for Robot Arms with Reinforcement Learning", arXiv:2611.05372; code github.com/Tinaliu0123/PolicyIK,
simulation/policyik_sim/tasks/*.py and agents/rl_games_ppo_cfg.yaml), adapted to this repository's kinematic
setting (see README.md here, "Deviations from the paper", for what was changed and why).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from utils.config import _ConfigMixin


@dataclass
class PolicyIKConfig(_ConfigMixin):
    """Robot-independent PolicyIK environment, action, observation and reward configuration."""

    # Timing: policy rate 30 Hz; every reward term is multiplied by dt (as Isaac Lab's reward manager does)
    dt: float = 1.0 / 30.0
    episode_steps: int = 360  # 12 s episodes ...
    goal_resample_steps: int = 120  # ... with a new goal every 4 s, without resetting the arm
    home_reset_prob: float = 0.25  # share of episode starts near the home pose (as this repo's SAC env)

    # Success tolerance used by the progress / dwell rewards and the error history (1 mm / 0.2 deg)
    tol_pos: float = 0.001
    tol_rot: float = math.radians(0.2)

    # Action: accumulated joint reference z_t = clip(z_{t-1} + eta * beta(e_p, e_R) * a_t, q_min, q_max)
    # Gain ramp: wider and lower than PolicyIK's (beta_min 0.1, ramps 0.5-9.31 mm / 0.415-2.328 deg). In PolicyIK
    # the PD-tracked arm low-pass filters the exploration noise in the reference; a kinematic arm executes it
    # directly. TCP noise per step ~ sigma * eta * beta * L * sqrt(n) (~0.024 beta m for the Franka at
    # sigma 0.3, PolicyIK's final std), so with the paper's ramp the noise exceeds the error itself near the goal
    # and training rollouts never enter the tolerance (Franka / UR5e stalled at 7-10 mm; the 3x smaller SO-101
    # still learned). The ramp below keeps noise < 0.5 x error and a floor of ~0.3 mm / 0.03 deg.
    action_scale: float = 0.0625  # eta (rad per step at full gain)
    gain_min: float = 0.012  # beta_min (paper: 0.1)
    gain_pos_lo: float = 0.0005  # position ramp of the gain (m): beta_min below lo, 1 above hi
    gain_pos_hi: float = 0.06  # (paper: 0.00931)
    gain_rot_lo: float = math.radians(0.415)  # orientation ramp (rad)
    gain_rot_hi: float = 0.3  # (paper: 2.328 deg)
    clip_actions: float = 100.0  # PolicyIK does not squash the Gaussian action (rl_games clip_actions=100)

    # Observation scales
    obs_kp_size: float = 0.30  # keypoints at 0.30 m along each TCP axis (goal - current offsets, 9-D)
    coarse_pos: float = 0.01  # coarse pose channel: clip(e / scale, -1, 1), 1 cm / 20 deg
    coarse_rot: float = math.radians(20.0)
    fine_pos: float = 0.0005  # fine pose channel: 0.5 mm / 0.25 deg
    fine_rot: float = math.radians(0.25)
    hist_scale: float = 20.0  # error history h_t = clip([b_{t-1}, rho_t - b_{t-1}] / 20, 0, 1)
    obs_noise: float = 0.01  # training only: uniform noise on the joint position / velocity channels

    # Reward (weights before the dt factor)
    w_pos_l2: float = -0.2  # -0.2 * e_p                  (OpenArm reach default)
    w_pos_fine: float = 0.1  # 0.1 * (1 - tanh(e_p / 0.1)) (OpenArm reach default)
    pos_fine_std: float = 0.1
    reward_kp_size: float = 0.45  # tracking keypoints at kappa * (e_i - 1/3) (centered)
    w_kp_wide: float = 1.0  # 1 - tanh(eps / 0.20), eps = mean keypoint distance
    kp_wide_std: float = 0.20
    w_kp_sharp: float = 1.0  # 1 - tanh(eps / 0.05)
    kp_sharp_std: float = 0.05
    w_improve: float = 26.831  # [U(rho_t) - U(b_{t-1})]_+, U(rho) = (1 + tanh((1 - rho) / beta_u)) / 2
    improve_beta: float = 0.5
    w_potential: float = 1.0  # gamma * Phi_t - Phi_{t-1}, Phi = -min(cap, rho)
    potential_gamma: float = 0.99
    potential_cap: float = 50.0
    w_backslide: float = 0.2  # -min(cap, [rho_t - b_{t-1}]_+)
    backslide_cap: float = 3.0
    w_dwell: float = 1.0  # 1[rho_t <= 1]
    w_joint_margin: float = 1.0  # clip(1 - min_j m_j / eps, 0, 1), only while rho > gate
    joint_margin_eps: float = 0.05
    joint_margin_gate: float = 10.0
    # Motion costs (squared action rate, squared joint velocity), raised after a number of env steps
    w_action_rate: float = 5e-4
    w_joint_vel: float = 5e-4
    w_action_rate_final: float = 0.025
    w_joint_vel_final: float = 0.005
    motion_curriculum_steps: int = 4500
    # Kinematic adaptation: colliding steps are rejected (the arm stays); PolicyIK's physics has no such term
    w_collision: float = 1.0

    # This repository's benchmark tolerances (evaluation/benchmark.py reads them from solver.cfg)
    term_pos_tol: float = 0.002
    term_rot_tol: float = 0.02
    eval_pos_tol: float = 0.005
    eval_rot_tol: float = 0.05

    def obs_dim(self, n_joints: int) -> int:
        # q - q_home, qdot, prev action (3n) + goal pos 3 + keypoints 9 + 4 pose channels 12 + history 2
        return 3 * n_joints + 26


@dataclass
class PPOConfig(_ConfigMixin):
    """PPO hyperparameters (rl_games a2c_continuous settings of PolicyIK)."""

    hidden_dims: tuple[int, ...] = (256, 256)  # shared actor-critic trunk, ELU
    init_log_std: float = 0.0  # state-independent log std
    horizon: int = 24
    mini_epochs: int = 5
    minibatch_size: int = 24576
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-3
    kl_threshold: float = 0.01  # adaptive learning rate
    lr_min: float = 1e-6
    lr_max: float = 1e-2
    e_clip: float = 0.2
    clip_value: bool = True
    critic_coef: float = 2.0
    entropy_coef: float = 0.01
    bounds_loss_coef: float = 1e-4
    grad_norm: float = 1.0
    # Deployed policy: exponentially weighted average of the last checkpoints
    save_every: int = 100  # epochs
    average_last: int = 20
    average_half_life: float = 400.0  # epochs
