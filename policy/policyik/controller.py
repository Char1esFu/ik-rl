"""PolicyIK policy as a batched closed-loop controller (solver, hold protocol, viewer)."""

from __future__ import annotations

import numpy as np

from core.batch_kinematics import BatchKinematics
from core.controller import Controller
from policy.policyik.config import PolicyIKConfig
from policy.policyik.core import PolicyIKBatch


class PolicyIKController(Controller):
    """PPO policy on PolicyIKBatch (torch, on the backend's device); q / p / R are numpy views for evaluation."""

    name = "policyik"

    def __init__(self, agent, bk: BatchKinematics, cfg: PolicyIKConfig, device: str | None = None) -> None:
        """device: where the batch state lives (default: the backend's device; "cpu" for a few rows)."""
        self.agent = agent
        self.cfg = cfg
        self.batch = PolicyIKBatch(bk, cfg, device=device)

    def reset(self, q0, p_goal, R_goal) -> None:
        self.batch.reset(q0, p_goal, R_goal)

    def set_goal(self, rows, p_goal, R_goal) -> None:
        self.batch.set_goal(rows, p_goal, R_goal)

    def step(self, smooth: float = 1.0, scale: float = 1.0) -> None:
        a = self.agent.act_tensor(self.batch.obs(None))
        if smooth != 1.0 or scale != 1.0:
            a = scale * (smooth * a + (1.0 - smooth) * self.batch.prev_action)
        self.batch.step(a)

    def activity(self) -> np.ndarray:
        """Executed joint step / action scale eta: the gain-scaled action. Near the goal the gain is tiny, so the
        raw action can stay large while the arm no longer moves."""
        dq = self.batch.qdot * self.cfg.dt
        return (dq.abs().max(dim=1).values / self.cfg.action_scale).cpu().numpy()

    def __getattr__(self, k):  # q, p, R, p_goal, R_goal live in the batch
        if k in ("q", "p", "R", "p_goal", "R_goal"):
            return getattr(self.batch, k).cpu().numpy()
        raise AttributeError(k)
