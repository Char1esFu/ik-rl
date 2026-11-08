"""SAC+DLS policy as a batched closed-loop controller (hold protocol, viewer); same step as rollout.py."""

from __future__ import annotations

import numpy as np

from core.batch_kinematics import BatchKinematics
from core.controller import KinematicController
from policy.sac_dls.config import IKConfig
from policy.sac_dls.env import action_to_delta_q
from policy.sac_dls.obs import compute_obs_batch


class SACController(KinematicController):
    name = "sac"

    def __init__(self, agent, bk: BatchKinematics, cfg: IKConfig) -> None:
        super().__init__(bk)
        self.agent = agent
        self.cfg = cfg
        self.horizon = cfg.max_episode_steps  # the training episode length (65)

    def reset(self, q0, p_goal, R_goal) -> None:
        super().reset(q0, p_goal, R_goal)
        self.prev_act = np.zeros_like(self.q)
        self.prev_col = np.zeros(len(self.q), dtype=bool)

    def step(self, smooth: float = 1.0, scale: float = 1.0) -> None:
        k = self.kin
        obs = compute_obs_batch(
            self.q, self.p, self.R, self.J, self.prev_act, self.prev_col, self.p_goal, self.R_goal,
            k.q_min, k.q_max, self.cfg.obs_features,
        )
        acts = self.agent.select_action(obs, deterministic=True)
        if smooth != 1.0 or scale != 1.0:
            acts = (scale * (smooth * acts + (1.0 - smooth) * self.prev_act)).astype(np.float32)
        dq = action_to_delta_q(acts, self.p, self.R, self.p_goal, self.R_goal, self.cfg)
        self.prev_col = self._move(dq)
        self.prev_act = np.clip(acts, -1.0, 1.0)

    def activity(self) -> np.ndarray:
        return np.abs(self.prev_act).max(axis=1)
