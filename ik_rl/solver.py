"""High-level 6D inverse kinematics solver API powered by a trained SAC policy (any robot)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ik_rl.config import IKConfig
from ik_rl.env import RobotKinematics, action_to_delta_q
from ik_rl.math_utils import quat_to_rotmat, rotmat_to_quat, so3_error_vec
from ik_rl.obs import compute_obs_batch
from ik_rl.robot import DEFAULT_ROBOT, RobotSpec, get_robot
from ik_rl.sac import SACAgent


@dataclass
class IKSolution:
    """Result returned by SACIKSolver.solve()."""

    qpos: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    steps: int
    success: bool


def checkpoint_robot(meta: dict) -> RobotSpec:
    """Robot a checkpoint was trained for: the stored spec if present, else configs/<name>.yaml.

    Checkpoints without robot metadata are Franka.
    """
    if "robot_spec" in meta:
        return RobotSpec.from_dict(meta["robot_spec"])
    return get_robot(meta.get("robot", DEFAULT_ROBOT))


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
        self.agent, meta = SACAgent.load(checkpoint_path, device=device)
        if robot is None:
            robot = checkpoint_robot(meta)
        self.robot = get_robot(robot) if isinstance(robot, str) else robot
        if env_cfg is not None:
            self.cfg = env_cfg
        elif "env_config" in meta:
            self.cfg = IKConfig.from_dict(meta["env_config"])
        else:
            self.cfg = IKConfig.for_robot(self.robot)

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
    ) -> IKSolution:
        """Solve 6D IK for (target_pos, target_quat [w,x,y,z] or target_rot [3,3])."""
        p_goal = np.asarray(target_pos, dtype=np.float32).reshape(3)
        if target_rot is not None:
            R_goal = np.asarray(target_rot, dtype=np.float32).reshape(3, 3)
        elif target_quat is not None:
            R_goal = quat_to_rotmat(np.asarray(target_quat, dtype=np.float32).reshape(4))
        else:
            raise ValueError("Must provide either target_quat [w,x,y,z] or target_rot [3,3].")

        kin = self.kin
        n = kin.n
        k = max(1, int(num_restarts))
        qs = np.zeros((k, n), dtype=np.float32)
        ps = np.zeros((k, 3), dtype=np.float32)
        Rs = np.zeros((k, 3, 3), dtype=np.float32)
        Js = np.zeros((k, 6, n), dtype=np.float32)
        prev_acts = np.zeros((k, n), dtype=np.float32)
        prev_cols = np.zeros(k, dtype=bool)

        for i in range(k):
            if i == 0 and q_init is not None:
                q0 = np.clip(np.asarray(q_init, dtype=np.float32), self.q_min, self.q_max)
            elif i == 0 or (i == 1 and q_init is not None):
                q0 = kin.home_q.copy()
            else:
                for _ in range(100):
                    cand = kin.sample_box(self.rng)
                    p_c, _, _, col = self.forward_kinematics(cand)
                    if not col and kin.valid_start(p_c):
                        q0 = cand
                        break
                else:
                    q0 = kin.home_q.copy()
            p0, R0, J0, _ = self.forward_kinematics(q0)
            qs[i], ps[i], Rs[i], Js[i] = q0, p0, R0, J0

        p_goals = np.broadcast_to(p_goal[None], (k, 3)).copy()
        R_goals = np.broadcast_to(R_goal[None], (k, 3, 3)).copy()

        best_q = qs[0].copy()
        best_p = ps[0].copy()
        best_R = Rs[0].copy()
        best_pos_err = float(np.linalg.norm(p_goal - best_p))
        _, rot_arr = so3_error_vec(best_R[None], R_goal[None])
        best_rot_err = float(rot_arr[0])
        best_score = best_pos_err + 0.2 * best_rot_err
        steps_used = 0

        for step in range(1, max_steps + 1):
            d_pos = np.linalg.norm(p_goals - ps, axis=-1)
            _, d_rot = so3_error_vec(Rs, R_goals)
            scores = d_pos + 0.2 * d_rot
            min_idx = int(np.argmin(scores))
            if scores[min_idx] < best_score:
                best_score = float(scores[min_idx])
                best_q = qs[min_idx].copy()
                best_p = ps[min_idx].copy()
                best_R = Rs[min_idx].copy()
                best_pos_err = float(d_pos[min_idx])
                best_rot_err = float(d_rot[min_idx])
                steps_used = step - 1

            if (best_pos_err < pos_tol) and (best_rot_err < rot_tol):
                break

            obs = compute_obs_batch(
                qs, ps, Rs, Js, prev_acts, prev_cols, p_goals, R_goals, self.q_min, self.q_max, self.cfg.obs_features
            )
            acts = self.agent.select_action(obs, deterministic=True)
            dqs = action_to_delta_q(acts, ps, Rs, p_goals, R_goals, self.cfg)
            q_cands = np.clip(qs + dqs, self.q_min, self.q_max)

            cols = np.zeros(k, dtype=bool)
            for i in range(k):
                p_c, R_c, J_c, col = self.forward_kinematics(q_cands[i])
                if not col:
                    qs[i], ps[i], Rs[i], Js[i] = q_cands[i], p_c, R_c, J_c
                else:
                    cols[i] = True
            prev_acts = acts.copy()
            prev_cols = cols

        d_pos = np.linalg.norm(p_goals - ps, axis=-1)
        _, d_rot = so3_error_vec(Rs, R_goals)
        scores = d_pos + 0.2 * d_rot
        min_idx = int(np.argmin(scores))
        if scores[min_idx] < best_score:
            best_q = qs[min_idx].copy()
            best_p = ps[min_idx].copy()
            best_R = Rs[min_idx].copy()
            best_pos_err = float(d_pos[min_idx])
            best_rot_err = float(d_rot[min_idx])
            steps_used = max_steps

        return IKSolution(
            qpos=best_q,
            ee_pos=best_p,
            ee_quat=rotmat_to_quat(best_R),
            pos_err_m=best_pos_err,
            rot_err_rad=best_rot_err,
            steps=steps_used,
            success=(best_pos_err < pos_tol) and (best_rot_err < rot_tol),
        )
