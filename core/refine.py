"""Damped-least-squares refinement of a near solution (float64 MuJoCo kinematics).

The "DLS" of SAC+DLS; PolicyIKSolver uses the same refinement with refine=True.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from core.kinematics import RobotKinematics
from utils.math_utils import so3_error_vec


@dataclass
class RefineResult:
    q: np.ndarray  # (n,) float32
    p: np.ndarray  # (3,)
    R: np.ndarray  # (3, 3)
    pos_err: float
    rot_err: float
    iters: int


def dls_refine(
    kin: RobotKinematics,
    data: mujoco.MjData,
    q0: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    pos_tol: float = 1e-4,
    rot_tol: float = float(np.radians(1e-3)),
    max_iters: int = 30,
    damping: float = 1e-3,
    max_step: float = 0.2,
) -> RefineResult | None:
    """Damped-least-squares refinement of a near solution (float64 kinematics).

    dq = J^T (J J^T + damping^2 I)^-1 [p_goal - p; w_err], clipped to the joint limits and to `max_step` rad
    per joint; a step that does not reduce the error is halved (up to 6 times), otherwise the refinement
    stops. Returns None if the refined configuration collides (the caller keeps its unrefined solution).
    """
    m = kin.model
    sid = kin.ee_site_id
    q_min = kin.q_min.astype(np.float64)
    q_max = kin.q_max.astype(np.float64)
    p_goal = np.asarray(p_goal, dtype=np.float64)
    R_goal = np.asarray(R_goal, dtype=np.float64)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    def pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[kin.arm_qadr] = q
        data.qpos[kin.fixed_qadr] = kin.fixed_val
        mujoco.mj_kinematics(m, data)
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()

    def error(p: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, float, float]:
        w, ang = so3_error_vec(R[None], R_goal[None], out_dtype=np.float64)
        e = np.concatenate([p_goal - p, w[0]])
        return e, float(np.linalg.norm(e[:3])), float(ang[0])

    q = np.clip(np.asarray(q0, dtype=np.float64), q_min, q_max)
    p, R = pose(q)
    e, d_pos, d_rot = error(p, R)
    iters = 0
    while iters < max_iters and not (d_pos < pos_tol and d_rot < rot_tol):
        iters += 1
        mujoco.mj_comPos(m, data)  # required by mj_jacSite
        mujoco.mj_jacSite(m, data, jacp, jacr, sid)
        J = np.vstack([jacp[:, kin.arm_dadr], jacr[:, kin.arm_dadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
        dq *= min(1.0, max_step / max(float(np.abs(dq).max()), 1e-12))
        norm0 = float(np.linalg.norm(e))
        for _ in range(7):
            q_new = np.clip(q + dq, q_min, q_max)
            p_new, R_new = pose(q_new)
            e_new, dp_new, dr_new = error(p_new, R_new)
            if float(np.linalg.norm(e_new)) < norm0:
                break
            dq *= 0.5
        else:
            pose(q)  # restore data
            break
        q, p, R, e, d_pos, d_rot = q_new, p_new, R_new, e_new, dp_new, dr_new

    q32 = q.astype(np.float32)
    _, _, _, collided = kin.fk(data, q32)
    if collided:
        return None
    p, R = pose(q32.astype(np.float64))  # report the error of the returned (float32) configuration
    _, d_pos, d_rot = error(p, R)
    return RefineResult(q32, p.astype(np.float32), R.astype(np.float32), d_pos, d_rot, iters)
