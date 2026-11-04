"""Batched policy rollout: the iterative IK loop shared by the solver and evaluation.

A rollout runs G independent IK problems ("groups", one target each) with k start configurations
("branches") per group in one batch: one policy forward pass per step for all active rows. Single-start
solving is G = 1, k = 1; multi-start solving is G = 1, k > 1; evaluation solves many targets at once
(G = N). A group stops as soon as its best branch is within tolerance; the result per group is the best
(position + 0.2 * rotation error) pose seen over all its branches and steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from ik_rl.config import IKConfig
from ik_rl.env import RobotKinematics, action_to_delta_q
from ik_rl.math_utils import so3_error_vec
from ik_rl.obs import compute_obs_batch

if TYPE_CHECKING:
    from ik_rl.sac import SACAgent

ROT_SCORE_WEIGHT = 0.2  # branch ranking score: d_pos [m] + 0.2 * d_rot [rad]


@dataclass
class RolloutResult:
    """Best pose per group (G groups)."""

    q: np.ndarray  # (G, n)
    p: np.ndarray  # (G, 3)
    R: np.ndarray  # (G, 3, 3)
    pos_err: np.ndarray  # (G,) meters
    rot_err: np.ndarray  # (G,) radians
    steps: np.ndarray  # (G,) policy steps until the best pose was reached
    success: np.ndarray  # (G,) best pose within (pos_tol, rot_tol)
    branch_q: np.ndarray  # (G, k, n) final configuration of every branch (e.g. alternative seeds for refinement)
    branch_score: np.ndarray  # (G, k) final d_pos + 0.2 * d_rot of every branch


def rollout_groups(
    agent: SACAgent,
    kin: RobotKinematics,
    data: mujoco.MjData,
    cfg: IKConfig,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    max_steps: int,
    pos_tol: float,
    rot_tol: float,
) -> RolloutResult:
    """Deterministic policy rollout for q_start (G, k, n) towards p_goal (G, 3) / R_goal (G, 3, 3)."""
    G, k, n = q_start.shape
    B = G * k
    q = q_start.reshape(B, n).astype(np.float32).copy()
    p = np.zeros((B, 3), dtype=np.float32)
    R = np.zeros((B, 3, 3), dtype=np.float32)
    J = np.zeros((B, 6, n), dtype=np.float32)
    for i in range(B):
        p[i], R[i], J[i], _ = kin.fk(data, q[i])
    prev_act = np.zeros((B, n), dtype=np.float32)
    prev_col = np.zeros(B, dtype=bool)
    p_goals = np.repeat(np.asarray(p_goal, dtype=np.float32).reshape(G, 3), k, axis=0)
    R_goals = np.repeat(np.asarray(R_goal, dtype=np.float32).reshape(G, 3, 3), k, axis=0)

    # Best so far per group, initialized with the group's first branch. Errors are kept as float64 copies of
    # the float32 values and compared to the tolerances in float64; scores are compared in float32.
    best_q = q.reshape(G, k, n)[:, 0].copy()
    best_p = p.reshape(G, k, 3)[:, 0].copy()
    best_R = R.reshape(G, k, 3, 3)[:, 0].copy()
    best_pos = np.zeros(G, dtype=np.float64)
    best_rot = np.zeros(G, dtype=np.float64)
    best_score = np.zeros(G, dtype=np.float32)
    for g in range(G):
        best_pos[g] = float(np.linalg.norm(p_goals[g * k] - best_p[g]))
        _, rot = so3_error_vec(best_R[g][None], R_goals[g * k][None])
        best_rot[g] = float(rot[0])
        best_score[g] = best_pos[g] + ROT_SCORE_WEIGHT * best_rot[g]
    steps = np.zeros(G, dtype=np.int64)
    done = np.zeros(G, dtype=bool)

    def record_best(step_value: int) -> None:
        d_pos = np.linalg.norm(p_goals - p, axis=-1)
        _, d_rot = so3_error_vec(R, R_goals)
        scores = (d_pos + ROT_SCORE_WEIGHT * d_rot).reshape(G, k)
        idx = np.argmin(scores, axis=1)
        smin = scores[np.arange(G), idx]
        for g in np.flatnonzero((smin < best_score) & ~done):
            row = g * k + idx[g]
            best_score[g] = smin[g]
            best_q[g], best_p[g], best_R[g] = q[row], p[row], R[row]
            best_pos[g], best_rot[g] = float(d_pos[row]), float(d_rot[row])
            steps[g] = step_value

    for step in range(1, max_steps + 1):
        record_best(step - 1)
        done |= (best_pos < pos_tol) & (best_rot < rot_tol)
        if done.all():
            break
        rows = np.flatnonzero(np.repeat(~done, k))
        obs = compute_obs_batch(
            q[rows], p[rows], R[rows], J[rows], prev_act[rows], prev_col[rows], p_goals[rows], R_goals[rows],
            kin.q_min, kin.q_max, cfg.obs_features,
        )
        acts = agent.select_action(obs, deterministic=True)
        dqs = action_to_delta_q(acts, p[rows], R[rows], p_goals[rows], R_goals[rows], cfg)
        q_r, p_r, R_r, J_r = q[rows], p[rows], R[rows], J[rows]
        cols = kin.apply_joint_step(data, q_r, p_r, R_r, J_r, dqs)
        q[rows], p[rows], R[rows], J[rows] = q_r, p_r, R_r, J_r
        prev_act[rows] = acts
        prev_col[rows] = cols
    else:
        record_best(max_steps)  # state after the last step

    d_pos = np.linalg.norm(p_goals - p, axis=-1)
    _, d_rot = so3_error_vec(R, R_goals)
    return RolloutResult(
        branch_q=q.reshape(G, k, n).copy(),
        branch_score=(d_pos + ROT_SCORE_WEIGHT * d_rot).reshape(G, k),
        q=best_q,
        p=best_p,
        R=best_R,
        pos_err=best_pos,
        rot_err=best_rot,
        steps=steps,
        success=(best_pos < pos_tol) & (best_rot < rot_tol),
    )


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
