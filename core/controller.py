"""Batched closed-loop IK controller interface and the best-pose rollout used by solvers.

A controller moves B arms towards B goals one control step at a time:
    reset(q0, p_goal, R_goal); set_goal(rows, p_goal, R_goal); step(); attributes q, p, R (current state)
Implementations: policy/policyik/controller.py (PolicyIKController), policy/sac_dls/controller.py
(SACController, same step as policy/sac_dls/rollout.py) and evaluation/baselines.py (DLSController). All use
the same kinematic step rules (joint limits, colliding moves rejected), so evaluation/hold.py compares them on
identical terms. run_best is the solver protocol (best pose, early stop) for any controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.batch_kinematics import BatchKinematics
from core.solution import ROT_SCORE_WEIGHT
from utils.math_utils import so3_error_vec


def pose_errors(p: np.ndarray, R: np.ndarray, p_g: np.ndarray, R_g: np.ndarray):
    """numpy float64 errors for evaluation: e_p (B,), e_R (B,), TCP-frame position / rotation errors (B, 3)."""
    dp = (np.asarray(p_g) - np.asarray(p)).astype(np.float64)
    w_world, ang = so3_error_vec(R, R_g, out_dtype=np.float64)
    Rt = np.swapaxes(np.asarray(R), -1, -2).astype(np.float64)
    return np.linalg.norm(dp, axis=-1), ang, np.einsum("bij,bj->bi", Rt, dp), np.einsum("bij,bj->bi", Rt, w_world)


class Controller:
    name = "controller"
    q: np.ndarray
    p: np.ndarray
    R: np.ndarray
    horizon: int = 120  # step budget for one target in interactive use (the viewer)

    def reset(self, q0: np.ndarray, p_goal: np.ndarray, R_goal: np.ndarray) -> None:
        """Place the arms at q0 (B, n) with goals; clears the controller's memory (previous action, ...)."""
        raise NotImplementedError

    def set_goal(self, rows: np.ndarray, p_goal: np.ndarray, R_goal: np.ndarray) -> None:
        """New goals for `rows` without moving the arms (previous actions are kept)."""
        raise NotImplementedError

    def step(self, smooth: float = 1.0, scale: float = 1.0) -> None:
        """One control step. Policies execute a = scale * (smooth * a_policy + (1 - smooth) * a_prev): smooth < 1
        low-pass filters the actions (moving goals in the viewer), scale < 1 damps them."""
        raise NotImplementedError

    def activity(self) -> np.ndarray:
        """Per-row size of the last action, max |a| (about 0 once the arm has settled)."""
        raise NotImplementedError

    def errors(self) -> tuple[np.ndarray, np.ndarray]:
        e_p, e_r, _, _ = pose_errors(self.p, self.R, self.p_goal, self.R_goal)
        return e_p, e_r


class KinematicController(Controller):
    """Controller that moves q itself (numpy state; RobotKinematics.apply_joint_step semantics)."""

    def __init__(self, bk: BatchKinematics) -> None:
        self.bk = bk
        self.kin = bk.kin

    def reset(self, q0, p_goal, R_goal) -> None:
        self.q = np.clip(np.asarray(q0, dtype=np.float32), self.kin.q_min, self.kin.q_max).copy()
        self.p, self.R, self.J, _ = self.bk.fk(self.q)
        self.p_goal = np.asarray(p_goal, dtype=np.float32).copy()
        self.R_goal = np.asarray(R_goal, dtype=np.float32).copy()

    def set_goal(self, rows, p_goal, R_goal) -> None:
        self.p_goal[rows] = p_goal
        self.R_goal[rows] = R_goal

    def _move(self, dq: np.ndarray) -> np.ndarray:
        """q <- clip(q + dq) unless colliding (RobotKinematics.apply_joint_step semantics); returns collided."""
        z = np.clip(self.q + dq, self.kin.q_min, self.kin.q_max)
        p, R, J, col = self.bk.fk(z)
        ok = ~col
        self.q[ok], self.p[ok], self.R[ok], self.J[ok] = z[ok], p[ok], R[ok], J[ok]
        return col


@dataclass
class BestResult:
    q: np.ndarray  # (G, n) best configuration per group
    p: np.ndarray
    R: np.ndarray
    pos_err: np.ndarray  # (G,)
    rot_err: np.ndarray
    steps: np.ndarray  # step at which the best pose was reached
    success: np.ndarray
    branch_q: np.ndarray  # (G, k, n) final configuration of every branch
    branch_score: np.ndarray  # (G, k)


def run_best(
    ctrl: Controller,
    q_start: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    max_steps: int,
    pos_tol: float,
    rot_tol: float,
) -> BestResult:
    """This repository's solver protocol for any controller (cf. policy.sac_dls.rollout.rollout_groups).

    G groups x k branches; a group stops as soon as its best pose is within tolerance; the result is the best
    (d_pos + 0.2 d_rot) pose seen over all branches and steps. Finished groups keep stepping in the batch
    but their result is frozen.
    """
    G, k, n = q_start.shape
    pg = np.repeat(np.asarray(p_goal, dtype=np.float32).reshape(G, 3), k, axis=0)
    Rg = np.repeat(np.asarray(R_goal, dtype=np.float32).reshape(G, 3, 3), k, axis=0)
    ctrl.reset(q_start.reshape(G * k, n), pg, Rg)
    best_score = np.full(G, np.inf)
    best_q = np.zeros((G, n), dtype=np.float32)
    best_p = np.zeros((G, 3), dtype=np.float32)
    best_R = np.zeros((G, 3, 3), dtype=np.float32)
    best_pos = np.zeros(G)
    best_rot = np.zeros(G)
    steps = np.zeros(G, dtype=np.int64)
    done = np.zeros(G, dtype=bool)
    ar = np.arange(G)
    for step in range(max_steps + 1):
        if step > 0:
            ctrl.step()
        e_p, e_r, _, _ = pose_errors(ctrl.p, ctrl.R, pg, Rg)
        score = (e_p + ROT_SCORE_WEIGHT * e_r).reshape(G, k)
        j = score.argmin(axis=1)
        s = score[ar, j]
        upd = (s < best_score) & ~done
        r = ar[upd] * k + j[upd]
        best_score[upd] = s[upd]
        best_q[upd], best_p[upd], best_R[upd] = ctrl.q[r], ctrl.p[r], ctrl.R[r]
        best_pos[upd], best_rot[upd] = e_p[r], e_r[r]
        steps[upd] = step
        done |= (best_pos < pos_tol) & (best_rot < rot_tol)
        if done.all():
            break
    e_p, e_r, _, _ = pose_errors(ctrl.p, ctrl.R, pg, Rg)
    return BestResult(
        best_q, best_p, best_R, best_pos, best_rot, steps, (best_pos < pos_tol) & (best_rot < rot_tol),
        ctrl.q.reshape(G, k, n).copy(), (e_p + ROT_SCORE_WEIGHT * e_r).reshape(G, k),
    )
