"""SAC+DLS: 6D IK solver API driven by a trained SAC policy (any robot), with optional DLS refinement."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from core.kinematics import RobotKinematics
from core.robot import RobotSpec, get_robot
from core.solution import IKSolution, checkpoint_robot, parse_target_rot, refine_branches, start_configs
from policy.sac_dls.config import IKConfig
from policy.sac_dls.rollout import rollout_groups
from policy.sac_dls.sac import SACAgent
from utils.math_utils import rotmat_to_quat

__all__ = ["SACIKSolver", "IKSolution", "checkpoint_robot"]


class SACIKSolver:
    """Iterative kinematic 6D IK solver using a trained Soft Actor-Critic policy.

    The robot is read from the checkpoint metadata unless given explicitly.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        robot: RobotSpec | str | None = None,
        env_cfg: IKConfig | None = None,
        device: str = "cuda",
    ) -> None:
        agent, meta = SACAgent.load(checkpoint_path, device=device)
        if robot is None:
            robot = checkpoint_robot(meta)
        if env_cfg is None:
            env_cfg = IKConfig.from_dict(meta["env_config"]) if "env_config" in meta else None
        self._setup(agent, robot, env_cfg)

    @classmethod
    def from_agent(cls, agent: SACAgent, robot: RobotSpec | str, env_cfg: IKConfig | None = None) -> SACIKSolver:
        """Solver around an in-memory agent (e.g. during training; it sees the agent's live weights)."""
        solver = cls.__new__(cls)
        solver._setup(agent, robot, env_cfg)
        return solver

    def _setup(self, agent: SACAgent, robot: RobotSpec | str, env_cfg: IKConfig | None) -> None:
        self.agent = agent
        self.robot = get_robot(robot) if isinstance(robot, str) else robot
        self.cfg = env_cfg if env_cfg is not None else IKConfig.for_robot(self.robot)
        self.kin = RobotKinematics(self.robot)
        expected = self.cfg.obs_dim(self.kin.n)
        if self.agent.cfg.obs_dim != expected:
            raise ValueError(
                f"Checkpoint obs_dim={self.agent.cfg.obs_dim} does not match robot '{self.robot.name}' "
                f"with obs_features={list(self.cfg.obs_features)} (obs_dim={expected})."
            )
        self.model = self.kin.model
        self.data = mujoco.MjData(self.model)
        self.q_min = self.kin.q_min
        self.q_max = self.kin.q_max
        self.rng = np.random.default_rng(0)

    def forward_kinematics(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return self.kin.fk(self.data, q)

    def start_configs(self, q_init: np.ndarray | None, num_restarts: int) -> np.ndarray:
        """Start configurations (k, n): q_init (if given), the home pose, then random collision-free ones."""
        return start_configs(self.kin, self.data, self.rng, q_init, num_restarts)

    _target_rot = staticmethod(parse_target_rot)

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_steps: int = 80,
        pos_tol: float = 0.005,
        rot_tol: float = 0.05,
        num_restarts: int = 1,
        refine: bool = False,
        refine_pos_tol: float = 1e-4,
        refine_rot_tol: float = float(np.radians(1e-3)),
        refine_iters: int = 30,
    ) -> IKSolution:
        """Solve 6D IK for (target_pos, target_quat [w,x,y,z] or target_rot [3,3]).

        The policy iterates until within (pos_tol, rot_tol). With refine=True, damped least squares then
        refines the best pose to (refine_pos_tol, refine_rot_tol) (default 0.1 mm / 0.001 deg).
        """
        return self.solve_batch(
            np.asarray(target_pos, dtype=np.float32).reshape(1, 3),
            target_quat=None if target_quat is None else np.asarray(target_quat).reshape(1, 4),
            target_rot=None if target_rot is None else np.asarray(target_rot).reshape(1, 3, 3),
            q_init=None if q_init is None else np.asarray(q_init).reshape(1, -1),
            max_steps=max_steps,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            num_restarts=num_restarts,
            refine=refine,
            refine_pos_tol=refine_pos_tol,
            refine_rot_tol=refine_rot_tol,
            refine_iters=refine_iters,
        )[0]

    def solve_batch(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray | None = None,
        target_rot: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_steps: int = 80,
        pos_tol: float = 0.005,
        rot_tol: float = 0.05,
        num_restarts: int = 1,
        refine: bool = False,
        refine_pos_tol: float = 1e-4,
        refine_rot_tol: float = float(np.radians(1e-3)),
        refine_iters: int = 30,
    ) -> list[IKSolution]:
        """Solve N targets at once: target_pos (N, 3), target_quat (N, 4) or target_rot (N, 3, 3), q_init (N, n).

        All N x num_restarts branches run as one GPU batch; each target stops when it is solved. Start
        configurations are generated per target in order, so random restarts match N calls of solve().
        """
        p_goal = np.asarray(target_pos, dtype=np.float32).reshape(-1, 3)
        count = p_goal.shape[0]
        R_goal = self._target_rot(target_quat, target_rot, count)
        q_inits = [None] * count if q_init is None else list(np.asarray(q_init, dtype=np.float32).reshape(count, -1))
        q_start = np.stack([self.start_configs(qi, num_restarts) for qi in q_inits])
        res = rollout_groups(
            self.agent, self.kin, self.data, self.cfg, q_start, p_goal, R_goal, max_steps, pos_tol, rot_tol
        )
        solutions = []
        for i in range(count):
            q, p, R = res.q[i], res.p[i], res.R[i]
            d_pos, d_rot, iters = float(res.pos_err[i]), float(res.rot_err[i]), 0
            ok = bool(res.success[i])
            if refine:
                q, p, R, d_pos, d_rot, iters, ok = refine_branches(
                    self.kin, self.data, (q, p, R, d_pos, d_rot), res.branch_q[i], res.branch_score[i],
                    p_goal[i], R_goal[i], refine_pos_tol, refine_rot_tol, refine_iters,
                )
            solutions.append(
                IKSolution(
                    qpos=q,
                    ee_pos=p,
                    ee_quat=rotmat_to_quat(R),
                    pos_err_m=d_pos,
                    rot_err_rad=d_rot,
                    steps=int(res.steps[i]),
                    success=ok,
                    refine_iters=iters,
                )
            )
        return solutions
