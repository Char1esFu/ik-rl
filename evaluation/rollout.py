"""Fixed-horizon rollout of any controller (hold protocol): per-step errors and TCP poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.controller import Controller, pose_errors


@dataclass
class HorizonLog:
    """Per-step errors of a fixed-horizon attempt (B rows, T steps) and the TCP poses needed by the metrics."""

    pos_t: np.ndarray  # (B, T) m, error to the (original) goal after each step
    rot_t: np.ndarray  # (B, T) rad
    q_final: np.ndarray  # (B, n)
    p_hist: np.ndarray  # (B, T, 3) TCP positions
    R_hist: np.ndarray  # (B, T, 3, 3)


def run_horizon(
    ctrl: Controller,
    q0: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    steps: int,
    transient: tuple[np.ndarray, np.ndarray, int] | None = None,
) -> HorizonLog:
    """Fixed-horizon attempt. With transient=(p_tr, R_tr, k), the goal is (p_tr, R_tr) for the first k steps.

    Errors are always measured against (p_goal, R_goal).
    """
    B = len(q0)
    rows = np.arange(B)
    if transient is not None:
        ctrl.reset(q0, transient[0], transient[1])
    else:
        ctrl.reset(q0, p_goal, R_goal)
    pos_t = np.zeros((B, steps))
    rot_t = np.zeros((B, steps))
    p_hist = np.zeros((B, steps, 3), dtype=np.float32)
    R_hist = np.zeros((B, steps, 3, 3), dtype=np.float32)
    for t in range(steps):
        if transient is not None and t == transient[2]:
            ctrl.set_goal(rows, p_goal, R_goal)
        ctrl.step()
        e_p, e_r, _, _ = pose_errors(ctrl.p, ctrl.R, p_goal, R_goal)
        pos_t[:, t], rot_t[:, t] = e_p, e_r
        p_hist[:, t], R_hist[:, t] = ctrl.p, ctrl.R
    return HorizonLog(pos_t, rot_t, ctrl.q.copy(), p_hist, R_hist)
