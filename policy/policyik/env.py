"""Vectorized PolicyIK training environment (thousands of kinematic arms; one batched FK call per step).

As in PolicyIK's Isaac Lab task: start configurations and goals are drawn from precomputed pools of
collision-free configurations (goals: TCP inside the workspace), an episode lasts `episode_steps` and the
goal is resampled every `goal_resample_steps` without resetting the arm (so later goals start from where the
arm ended up). Pools use the same acceptance rules as this repository's SAC environment and evaluation
targets. Everything (pools, state, observation, reward) stays on the kinematics backend's device.
"""

from __future__ import annotations

import torch

from core.batch_kinematics import BatchKinematics, sample_valid_configs
from policy.policyik.config import PolicyIKConfig
from policy.policyik.core import PolicyIKBatch


class PolicyIKVecEnv:
    def __init__(
        self,
        bk: BatchKinematics,
        cfg: PolicyIKConfig,
        num_envs: int,
        seed: int = 0,
        pool_size: int = 200_000,
    ) -> None:
        self.bk = bk
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = bk.device
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.n = bk.n
        self.obs_dim = cfg.obs_dim(self.n)
        # Start pool (with the home-pose share) and goal pool
        self.start_q, self.start_p, self.start_R = sample_valid_configs(
            bk, pool_size, self.gen, home_prob=cfg.home_reset_prob
        )
        _, self.goal_p, self.goal_R = sample_valid_configs(bk, pool_size, self.gen, workspace=True)
        self.batch = PolicyIKBatch(bk, cfg)
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.goal_step = torch.zeros_like(self.ep_step)
        self.common_step = 0  # environment steps (for the motion-cost curriculum)

    def _idx(self, pool_len: int, k: int) -> torch.Tensor:
        return torch.randint(0, pool_len, (k,), generator=self.gen, device=self.device)

    def reset(self) -> torch.Tensor:
        N = self.num_envs
        g = self._idx(len(self.goal_p), N)
        self.batch.reset(self.start_q[self._idx(len(self.start_q), N)], self.goal_p[g], self.goal_R[g])
        # stagger episode clocks so resets / goal changes are spread over time
        self.ep_step = torch.randint(0, self.cfg.episode_steps, (N,), generator=self.gen, device=self.device)
        self.goal_step = self.ep_step % self.cfg.goal_resample_steps
        return self.batch.obs(self.gen)

    def motion_weights(self) -> tuple[float, float]:
        c = self.cfg
        if self.common_step > c.motion_curriculum_steps:
            return c.w_action_rate_final, c.w_joint_vel_final
        return c.w_action_rate, c.w_joint_vel

    def step(self, actions: torch.Tensor):
        """Returns obs, reward, time_out (B,), noise-free obs before the reset of every row, and info.

        Rows with time_out == True have been reset; their pre-reset obs is the bootstrap state s_T.
        """
        c = self.cfg
        out = self.batch.step(actions, motion_weights=self.motion_weights())
        self.common_step += 1
        self.ep_step += 1
        self.goal_step += 1
        time_out = self.ep_step >= c.episode_steps
        final_obs = self.batch.obs(None)

        rows = torch.nonzero(time_out).squeeze(1)
        if len(rows):
            i = self._idx(len(self.start_q), len(rows))
            self.batch.reset_rows(rows, self.start_q[i], self.start_p[i], self.start_R[i])
            self.ep_step[rows] = 0
        rows = torch.nonzero(time_out | (self.goal_step >= c.goal_resample_steps)).squeeze(1)
        if len(rows):
            g = self._idx(len(self.goal_p), len(rows))
            self.batch.set_goal(rows, self.goal_p[g], self.goal_R[g])
            self.goal_step[rows] = 0
        info = {"e_p": out["e_p"], "e_r": out["e_r"], "in_tol": out["rho"] <= 1.0, "collided": out["collided"],
                "terms": out["terms"]}
        return self.batch.obs(self.gen), out["reward"], time_out, final_obs, info
