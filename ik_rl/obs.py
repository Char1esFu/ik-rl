"""Observation feature groups and the batched observation builder.

OBS_FEATURES is the schema of the observation: each row has a matching branch in compute_obs_batch, so it
lives here next to that code (not in a config file). Which groups are used is chosen by IKConfig.obs_features.
To add a feature: add a row to OBS_FEATURES and a branch in compute_obs_batch.
"""

from __future__ import annotations

import numpy as np

from ik_rl.math_utils import rotmat_to_r6d, so3_error_vec

# Observation feature groups, in the order they are concatenated:
#   name: (dims per arm joint, fixed dims)
OBS_FEATURES: dict[str, tuple[int, int]] = {
    "q_norm": (1, 0),  # joint angles normalized to [-1, 1] by the joint limits
    "sin_cos_q": (2, 0),  # sin(q), cos(q)
    "position": (0, 14),  # p_cur, p_goal, dp, dp/|dp|, |dp|, log10|dp|/4
    "orientation": (0, 20),  # 6D rot cur/goal, w_err, w_err/|w_err|, |w_err|/pi, log10|w_err|/4
    "jacobian": (6, 0),  # 0.5 * TCP Jacobian (6 x n)
    "jt_direction": (1, 0),  # tanh(J^T [u_pos; u_rot]): Jacobian-transpose direction to the target
    "prev_action": (1, 0),
    "prev_collided": (0, 1),
}
DEFAULT_OBS_FEATURES: tuple[str, ...] = tuple(OBS_FEATURES)


def validate_obs_features(features: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Check feature names and return them in canonical (concatenation) order."""
    unknown = set(features) - set(OBS_FEATURES)
    if unknown:
        raise ValueError(f"Unknown observation features {sorted(unknown)}; available: {list(OBS_FEATURES)}")
    return tuple(f for f in OBS_FEATURES if f in features)


def obs_dim(n_joints: int, features: tuple[str, ...] = DEFAULT_OBS_FEATURES) -> int:
    """Observation size for an n-joint arm with the given feature groups (see OBS_FEATURES)."""
    return sum(OBS_FEATURES[f][0] * n_joints + OBS_FEATURES[f][1] for f in validate_obs_features(features))


def compute_obs_batch(
    q: np.ndarray,
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    J_cur: np.ndarray,
    prev_action: np.ndarray,
    prev_collided: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    features: tuple[str, ...] = DEFAULT_OBS_FEATURES,
) -> np.ndarray:
    """Construct the observation from the selected feature groups (see OBS_FEATURES)."""
    features = validate_obs_features(features)
    batch = q.shape[0]

    dp = p_goal - p_cur
    d_pos = np.linalg.norm(dp, axis=-1, keepdims=True)
    u_pos = dp / np.maximum(d_pos, 1e-6)
    rot_err_vec, d_rot_1d = so3_error_vec(R_cur, R_goal)
    d_rot = d_rot_1d[:, None]
    u_rot = rot_err_vec / np.maximum(d_rot, 1e-6)

    blocks: list[np.ndarray] = []
    for name in features:
        if name == "q_norm":
            blocks.append(2.0 * (q - q_min) / (q_max - q_min) - 1.0)
        elif name == "sin_cos_q":
            blocks += [np.sin(q), np.cos(q)]
        elif name == "position":
            log_d_pos = np.log10(np.clip(d_pos, 1e-4, 10.0)) / 4.0
            blocks += [p_cur, p_goal, dp, u_pos, d_pos, log_d_pos]
        elif name == "orientation":
            log_d_rot = np.log10(np.clip(d_rot, 1e-4, 10.0)) / 4.0
            blocks += [rotmat_to_r6d(R_cur), rotmat_to_r6d(R_goal), rot_err_vec, u_rot, d_rot / np.pi, log_d_rot]
        elif name == "jacobian":
            blocks.append((0.5 * J_cur).reshape(batch, -1))
        elif name == "jt_direction":
            unit_twist = np.concatenate([u_pos, u_rot], axis=-1)
            blocks.append(np.tanh(np.einsum("nij,ni->nj", J_cur, unit_twist)))
        elif name == "prev_action":
            blocks.append(prev_action)
        elif name == "prev_collided":
            blocks.append(prev_collided.astype(np.float32).reshape(batch, 1))
    return np.concatenate(blocks, axis=-1).astype(np.float32)

