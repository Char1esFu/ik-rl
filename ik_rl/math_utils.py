"""Numerically stable SO(3) rotation utilities (MuJoCo wxyz quaternion convention)."""

from __future__ import annotations

import numpy as np


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo quaternion (..., 4) [w, x, y, z] to rotation matrix (..., 3, 3)."""
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.clip(norm, 1e-12, None)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # Standard unit-quaternion rotation matrix, written row by row
    rot = np.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))  # fmt: skip
    return rot.astype(np.float32)


def rotmat_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (..., 3, 3) to MuJoCo quaternion (..., 4) [w, x, y, z]."""
    R = np.asarray(rot, dtype=np.float64)
    orig_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    n = R_flat.shape[0]
    q = np.zeros((n, 4), dtype=np.float64)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]

    m0 = trace > 0.0
    if np.any(m0):
        s = np.sqrt(trace[m0] + 1.0) * 2.0
        q[m0, 0] = 0.25 * s
        q[m0, 1] = (R_flat[m0, 2, 1] - R_flat[m0, 1, 2]) / s
        q[m0, 2] = (R_flat[m0, 0, 2] - R_flat[m0, 2, 0]) / s
        q[m0, 3] = (R_flat[m0, 1, 0] - R_flat[m0, 0, 1]) / s

    m1 = (~m0) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    if np.any(m1):
        s = np.sqrt(1.0 + R_flat[m1, 0, 0] - R_flat[m1, 1, 1] - R_flat[m1, 2, 2]) * 2.0
        q[m1, 0] = (R_flat[m1, 2, 1] - R_flat[m1, 1, 2]) / s
        q[m1, 1] = 0.25 * s
        q[m1, 2] = (R_flat[m1, 0, 1] + R_flat[m1, 1, 0]) / s
        q[m1, 3] = (R_flat[m1, 0, 2] + R_flat[m1, 2, 0]) / s

    m2 = (~m0) & (~m1) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    if np.any(m2):
        s = np.sqrt(1.0 + R_flat[m2, 1, 1] - R_flat[m2, 0, 0] - R_flat[m2, 2, 2]) * 2.0
        q[m2, 0] = (R_flat[m2, 0, 2] - R_flat[m2, 2, 0]) / s
        q[m2, 1] = (R_flat[m2, 0, 1] + R_flat[m2, 1, 0]) / s
        q[m2, 2] = 0.25 * s
        q[m2, 3] = (R_flat[m2, 1, 2] + R_flat[m2, 2, 1]) / s

    m3 = (~m0) & (~m1) & (~m2)
    if np.any(m3):
        s = np.sqrt(1.0 + R_flat[m3, 2, 2] - R_flat[m3, 0, 0] - R_flat[m3, 1, 1]) * 2.0
        q[m3, 0] = (R_flat[m3, 1, 0] - R_flat[m3, 0, 1]) / s
        q[m3, 1] = (R_flat[m3, 0, 2] + R_flat[m3, 2, 0]) / s
        q[m3, 2] = (R_flat[m3, 1, 2] + R_flat[m3, 2, 1]) / s
        q[m3, 3] = 0.25 * s

    # Canonicalize w >= 0
    sign = np.where(q[:, 0:1] < 0.0, -1.0, 1.0)
    q = q * sign
    q = q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)
    return q.reshape(orig_shape + (4,)).astype(np.float32)


def rotmat_to_r6d(rot: np.ndarray) -> np.ndarray:
    """Extract continuous 6D rotation representation (first 2 columns flattened)."""
    return np.concatenate([rot[..., :, 0], rot[..., :, 1]], axis=-1).astype(np.float32)


def so3_error_vec(R_cur: np.ndarray, R_goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-frame SO(3) axis-angle error vector and geodesic angle [0, pi].

    R_err = R_goal @ R_cur^T so that exp([omega]_x) @ R_cur = R_goal.
    Uses atan2(||vee(R_err - R_err^T)||, tr(R_err) - 1) for full precision near 0.
    """
    orig_shape = R_cur.shape[:-2]
    Rc = np.asarray(R_cur, dtype=np.float64).reshape(-1, 3, 3)
    Rg = np.asarray(R_goal, dtype=np.float64).reshape(-1, 3, 3)
    R_err = np.einsum("nij,nkj->nik", Rg, Rc)
    vx = R_err[:, 2, 1] - R_err[:, 1, 2]
    vy = R_err[:, 0, 2] - R_err[:, 2, 0]
    vz = R_err[:, 1, 0] - R_err[:, 0, 1]
    v = np.stack([vx, vy, vz], axis=-1)  # ||v|| = 2 * sin(theta)

    norm_v = np.linalg.norm(v, axis=-1)
    trace_m1 = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2] - 1.0  # 2 * cos(theta)
    angle = np.arctan2(norm_v, trace_m1)  # exact in [0, pi]

    # General case: sin(theta) not near 0
    scale = np.where(norm_v > 1e-6, angle / np.where(norm_v > 1e-6, norm_v, 1.0), 0.5)
    omega = v * scale[:, None]

    # Near pi singularity: norm_v <= 1e-6 and trace_m1 < 0
    near_pi = (norm_v <= 1e-6) & (trace_m1 < 0.0)
    if np.any(near_pi):
        R_pi = R_err[near_pi] + np.eye(3, dtype=np.float64)  # 2 * u * u^T
        diag = np.diagonal(R_pi, axis1=-2, axis2=-1)
        k = np.argmax(diag, axis=-1)
        cols = R_pi[np.arange(R_pi.shape[0]), :, k]
        u = cols / np.clip(np.linalg.norm(cols, axis=-1, keepdims=True), 1e-12, None)
        omega[near_pi] = np.pi * u

    return omega.reshape(orig_shape + (3,)).astype(np.float32), angle.reshape(orig_shape).astype(np.float32)
