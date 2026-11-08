"""PolicyIKSolver: the same solve() / solve_batch() API as SACIKSolver, driven by a trained PolicyIK (PPO) policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.batch_kinematics import CPU_MAX_BATCH, BatchKinematics
from core.controller import run_best
from core.robot import RobotSpec
from core.solution import IKSolution, checkpoint_robot, parse_target_rot, refine_branches, start_configs
from policy.policyik.config import PolicyIKConfig
from policy.policyik.controller import PolicyIKController
from policy.policyik.ppo import PPOAgent
from utils.math_utils import rotmat_to_quat


class PolicyIKSolver:
    """Iterative kinematic 6D IK with a PolicyIK (PPO) policy; the robot is read from the checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cuda", backend: str | None = None) -> None:
        """backend: kinematics backend ("warp" on CUDA by default, or "cpu"); small batches always run on the CPU."""
        agent, meta = PPOAgent.load(checkpoint_path, device=device)
        self._setup(agent, checkpoint_robot(meta), PolicyIKConfig.from_dict(meta["env_config"]), backend)

    @classmethod
    def from_agent(cls, agent: PPOAgent, robot: RobotSpec, cfg: PolicyIKConfig, bk: BatchKinematics | None = None):
        solver = cls.__new__(cls)
        solver._setup(agent, robot, cfg, None, bk)
        return solver

    def _setup(self, agent, robot, cfg, backend, bk=None) -> None:
        self.agent = agent
        self.robot = robot
        self.cfg = cfg
        self.bk = bk if bk is not None else BatchKinematics(robot, backend=backend)
        self.kin = self.bk.kin
        if agent.obs_dim != cfg.obs_dim(self.kin.n):
            raise ValueError(f"Checkpoint obs_dim={agent.obs_dim} does not match robot '{robot.name}'")
        self.controller = PolicyIKController(agent, self.bk, cfg)
        # A few rows (single solve() calls): state, kinematics and policy on the CPU avoid GPU launch / sync latency
        cpu_agent = PPOAgent(agent.obs_dim, agent.act_dim, agent.cfg, device="cpu")
        cpu_agent.net.load_state_dict({k: v.cpu() for k, v in agent.net.state_dict().items()})
        cpu_agent.net.eval()
        self.cpu_controller = PolicyIKController(cpu_agent, self.bk, cfg, device="cpu")
        self.data = self.bk._data  # MjData for DLS refinement / start sampling
        self.rng = np.random.default_rng(0)

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return self.kin.fk(self.data, q)

    def start_configs(self, q_init: np.ndarray | None, num_restarts: int) -> np.ndarray:
        """Start configurations (k, n): q_init (if given), the home pose, then random collision-free ones."""
        return start_configs(self.kin, self.data, self.rng, q_init, num_restarts)

    _target_rot = staticmethod(parse_target_rot)

    def solve(self, target_pos, target_quat=None, target_rot=None, q_init=None, **kwargs) -> IKSolution:
        """Solve one target (target_quat [w,x,y,z] or target_rot [3,3]); keyword options as solve_batch."""
        return self.solve_batch(
            np.asarray(target_pos, dtype=np.float32).reshape(1, 3),
            target_quat=None if target_quat is None else np.asarray(target_quat).reshape(1, 4),
            target_rot=None if target_rot is None else np.asarray(target_rot).reshape(1, 3, 3),
            q_init=None if q_init is None else np.asarray(q_init).reshape(1, -1),
            **kwargs,
        )[0]

    @property
    def q_min(self):
        return self.kin.q_min

    @property
    def q_max(self):
        return self.kin.q_max

    def solve_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_steps: int = 160,
        pos_tol: float = 0.001,
        rot_tol: float = float(np.radians(0.2)),
        num_restarts: int = 1,
        refine: bool = False,
        refine_pos_tol: float = 1e-4,
        refine_rot_tol: float = float(np.radians(1e-3)),
        refine_iters: int = 30,
    ) -> list[IKSolution]:
        """Solve N targets in one batch (N x num_restarts branches); stops per target once within tolerance."""
        p_goal = np.asarray(target_pos, dtype=np.float32).reshape(-1, 3)
        count = p_goal.shape[0]
        R_goal = self._target_rot(target_quat, target_rot, count)
        q_inits = [None] * count if q_init is None else list(np.asarray(q_init, dtype=np.float32).reshape(count, -1))
        q_start = np.stack([self.start_configs(qi, num_restarts) for qi in q_inits])
        if q_start.shape[0] * q_start.shape[1] <= CPU_MAX_BATCH:
            # tiny CPU tensors: intra-op threads only contend (Franka solve() 68 ms -> 12 ms with one thread)
            threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                res = run_best(self.cpu_controller, q_start, p_goal, R_goal, max_steps, pos_tol, rot_tol)
            finally:
                torch.set_num_threads(threads)
        else:
            res = run_best(self.controller, q_start, p_goal, R_goal, max_steps, pos_tol, rot_tol)
        out = []
        for i in range(count):
            q, p, R = res.q[i], res.p[i], res.R[i]
            d_pos, d_rot, iters = float(res.pos_err[i]), float(res.rot_err[i]), 0
            ok = bool(res.success[i])
            if refine:  # same refinement as SACIKSolver (core/solution.py)
                q, p, R, d_pos, d_rot, iters, ok = refine_branches(
                    self.kin, self.data, (q, p, R, d_pos, d_rot), res.branch_q[i], res.branch_score[i],
                    p_goal[i], R_goal[i], refine_pos_tol, refine_rot_tol, refine_iters,
                )
            out.append(IKSolution(q, p, rotmat_to_quat(R), d_pos, d_rot, int(res.steps[i]), ok, iters))
        return out

