"""Classical baseline controllers for comparisons (not trained methods)."""

from __future__ import annotations

import numpy as np

from core.batch_kinematics import BatchKinematics
from core.controller import KinematicController
from utils.math_utils import so3_error_vec


class DLSController(KinematicController):
    """dq = J^T (J J^T + lambda^2 I)^-1 [p_goal - p; w_err], scaled to at most `max_step` rad per joint."""

    name = "dls"

    def __init__(self, bk: BatchKinematics, damping: float = 0.05, max_step: float = 0.0625) -> None:
        super().__init__(bk)
        self.damping = damping
        self.max_step = max_step

    def reset(self, q0, p_goal, R_goal) -> None:
        super().reset(q0, p_goal, R_goal)
        self._activity = np.zeros(len(self.q))

    def activity(self) -> np.ndarray:
        return self._activity

    def step(self, smooth: float = 1.0, scale: float = 1.0) -> None:  # smooth: no action memory, ignored
        w, _ = so3_error_vec(self.R, self.R_goal, out_dtype=np.float64)
        e = np.concatenate([(self.p_goal - self.p).astype(np.float64), w], axis=1)
        J = self.J.astype(np.float64)
        A = J @ np.swapaxes(J, 1, 2) + self.damping**2 * np.eye(6)
        dq = np.einsum("bji,bj->bi", J, np.linalg.solve(A, e[..., None])[..., 0])
        dq *= scale * np.minimum(1.0, self.max_step / np.maximum(np.abs(dq).max(axis=1), 1e-12))[:, None]
        self._activity = np.abs(dq).max(axis=1) / self.max_step
        self._move(dq.astype(np.float32))
