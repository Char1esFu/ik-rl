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

    return RolloutResult(
        q=best_q,
        p=best_p,
        R=best_R,
        pos_err=best_pos,
        rot_err=best_rot,
        steps=steps,
        success=(best_pos < pos_tol) & (best_rot < rot_tol),
    )
