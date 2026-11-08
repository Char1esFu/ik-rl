"""Solver building blocks shared by every policy's IK solver (policy/*/solver.py).

IKSolution (the result type), the robot of a checkpoint, start configurations for multi-start solving,
target-orientation parsing and the DLS refinement of the best branch (refine=True).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from core.kinematics import RobotKinematics
from core.refine import dls_refine
from core.robot import DEFAULT_ROBOT, RobotSpec, get_robot
from utils.math_utils import quat_to_rotmat

ROT_SCORE_WEIGHT = 0.2  # branch ranking score: d_pos [m] + 0.2 * d_rot [rad]


@dataclass
class IKSolution:
    """Result of solve(): best configuration and its TCP pose / errors."""

    qpos: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    pos_err_m: float
    rot_err_rad: float
    steps: int  # policy steps
    success: bool  # within (pos_tol, rot_tol), or the refinement tolerances if refine=True
    refine_iters: int = 0  # DLS iterations (0 if not refined)


def checkpoint_robot(meta: dict) -> RobotSpec:
    """Robot a checkpoint was trained for: the stored spec if present, else configs/<name>.yaml.

    Checkpoints without robot metadata are Franka.
    """
    if "robot_spec" in meta:
        return RobotSpec.from_dict(meta["robot_spec"])
    return get_robot(meta.get("robot", DEFAULT_ROBOT))


def start_configs(
    kin: RobotKinematics, data: mujoco.MjData, rng: np.random.Generator, q_init: np.ndarray | None, num_restarts: int
) -> np.ndarray:
    """Start configurations (k, n): q_init (if given), the home pose, then random collision-free ones."""
    k = max(1, int(num_restarts))
    qs = np.zeros((k, kin.n), dtype=np.float32)
    for i in range(k):
        if i == 0 and q_init is not None:
            q0 = np.clip(np.asarray(q_init, dtype=np.float32), kin.q_min, kin.q_max)
        elif i == 0 or (i == 1 and q_init is not None):
            q0 = kin.home_q.copy()
        else:
            for _ in range(100):
                cand = kin.sample_box(rng)
                p_c, _, _, col = kin.fk(data, cand)
                if not col and kin.valid_start(p_c):
                    q0 = cand
                    break
            else:
                q0 = kin.home_q.copy()
        qs[i] = q0
    return qs


def parse_target_rot(target_quat: np.ndarray | None, target_rot: np.ndarray | None, count: int) -> np.ndarray:
    """Target rotations (count, 3, 3) from rotation matrices or [w, x, y, z] quaternions."""
    if target_rot is not None:
        return np.asarray(target_rot, dtype=np.float32).reshape(count, 3, 3)
    if target_quat is not None:
        quats = np.asarray(target_quat, dtype=np.float32).reshape(count, 4)
        return np.stack([quat_to_rotmat(qt) for qt in quats])
    raise ValueError("Must provide either target_quat [w,x,y,z] or target_rot [3,3].")


def refine_branches(
    kin: RobotKinematics,
    data: mujoco.MjData,
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    branch_q: np.ndarray,
    branch_score: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    pos_tol: float,
    rot_tol: float,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, bool]:
    """DLS-refine the best pose; if it does not reach the tolerance, try the branches' final configurations.

    Refines the best (q, p, R, d_pos, d_rot); if that does not reach (pos_tol, rot_tol) (e.g. a near-solution in
    another IK branch, like SO-101's mirrored pan + pi configuration ~0.3 mm off), the other branches' final
    configurations are refined in order of their error, keeping the best result.
    Returns (q, p, R, d_pos, d_rot, total DLS iterations, success).
    """
    q, p, R, d_pos, d_rot = best
    iters = 0
    for seed in [q] + [branch_q[j] for j in np.argsort(branch_score)]:
        r = dls_refine(kin, data, seed, p_goal, R_goal, pos_tol, rot_tol, max_iters)
        if r is not None and r.pos_err + ROT_SCORE_WEIGHT * r.rot_err < d_pos + ROT_SCORE_WEIGHT * d_rot:
            q, p, R, d_pos, d_rot = r.q, r.p, r.R, r.pos_err, r.rot_err
        iters += r.iters if r is not None else 0
        if d_pos < pos_tol and d_rot < rot_tol:
            break
    return q, p, R, d_pos, d_rot, iters, (d_pos < pos_tol) and (d_rot < rot_tol)
