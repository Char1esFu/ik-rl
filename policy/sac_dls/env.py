"""SAC+DLS: MuJoCo 6D IK training environments (single + vectorized), action scaling and reward.

Configuration lives in policy/sac_dls/config.py (IKConfig), the observation in policy/sac_dls/obs.py, the robot
model / kinematics in core/kinematics.py.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from core.kinematics import RobotKinematics
from core.robot import RobotSpec
from policy.sac_dls.config import IKConfig
from policy.sac_dls.obs import compute_obs_batch
from utils.math_utils import quat_to_rotmat, rotmat_to_quat, so3_error_vec


def delta_q_scale(
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    cfg: IKConfig,
) -> np.ndarray:
    """Gain-schedule factor in [min_scale_ratio, 1] from the observable 6D pose error."""
    d_pos = np.linalg.norm(p_goal - p_cur, axis=-1)
    _, d_rot = so3_error_vec(R_cur, R_goal)
    return np.clip(
        np.maximum(d_pos / cfg.pos_scale_ref, d_rot / cfg.rot_scale_ref),
        cfg.min_scale_ratio,
        1.0,
    ).astype(np.float32)


def action_to_delta_q(
    action: np.ndarray,
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    cfg: IKConfig,
) -> np.ndarray:
    """Smooth linear gain-scheduled joint increment scaled by observable 6D pose error."""
    a = np.clip(action, -1.0, 1.0)
    scale = delta_q_scale(p_cur, R_cur, p_goal, R_goal, cfg)
    return (cfg.max_delta_q * scale[..., None] * a).astype(np.float32)


def compute_reward_batch(
    p_next: np.ndarray,
    R_next: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    dq: np.ndarray,
    collided: np.ndarray,
    p_goal: np.ndarray,
    R_goal: np.ndarray,
    cfg: IKConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """RL-for-IK reward: normalized pose kernels + success bonus + effort/rate/collision penalties.

    `dq` is the attempted joint increment (pre joint-limit clip) produced by `action`.
    """
    d_pos = np.linalg.norm(p_goal - p_next, axis=-1).astype(np.float32)
    _, d_rot = so3_error_vec(R_next, R_goal)

    def pose_kernel(e_coarse: np.ndarray, e_fine: np.ndarray) -> np.ndarray:
        return (
            -cfg.lin_weight * e_coarse
            + cfg.coarse_weight * np.exp(-cfg.coarse_gain * e_coarse)
            + cfg.fine_weight * np.exp(-cfg.fine_gain * e_fine)
        )

    # Same kernel shape for position & rotation -> balanced precision pressure at tolerance
    r_pos = pose_kernel(d_pos / cfg.pos_scale_ref, d_pos / cfg.term_pos_tol)
    r_rot = pose_kernel(d_rot / cfg.rot_scale_ref, d_rot / cfg.term_rot_tol)

    not_col = ~collided.astype(bool)
    tight = not_col & (d_pos < cfg.term_pos_tol) & (d_rot < cfg.term_rot_tol)
    r_succ = cfg.success_bonus * tight.astype(np.float32)

    # Effort on executed joint motion (not on normalized action, whose meaning is gain-scheduled)
    r_act = -cfg.action_penalty * np.sum(np.square(dq / cfg.max_delta_q), axis=-1)
    r_smooth = -cfg.smooth_penalty * np.sum(np.square(action - prev_action), axis=-1)
    r_col = -cfg.collision_penalty * collided.astype(np.float32)

    reward = cfg.reward_scale * (r_pos + r_rot + r_succ + r_act + r_smooth + r_col)
    # Fixed-horizon formulation (terminated=False) so policy learns to hold motionless at target
    terminated = np.zeros_like(collided, dtype=bool)
    return reward.astype(np.float32), terminated, d_pos, d_rot


class IKEnv(gym.Env):
    """Single Gymnasium 6D inverse kinematics environment for any RobotSpec."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        robot: RobotSpec | str,
        cfg: IKConfig | None = None,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        super().__init__()
        self.kin = RobotKinematics(robot)
        self.robot = self.kin.robot
        self.cfg = cfg or IKConfig.for_robot(self.robot)
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height

        self.model = self.kin.model
        self.data = mujoco.MjData(self.model)
        self.ee_site_id = self.kin.ee_site_id
        self.target_mocap_id = self.kin.target_mocap_id
        self.q_min = self.kin.q_min
        self.q_max = self.kin.q_max
        n = self.kin.n

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.obs_dim(n),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        self.q_cur = self.kin.home_q.copy()
        self.p_cur = np.zeros(3, dtype=np.float32)
        self.R_cur = np.eye(3, dtype=np.float32)
        self.J_cur = np.zeros((6, n), dtype=np.float32)
        self.prev_action = np.zeros(n, dtype=np.float32)
        self.prev_collided = False
        self.p_goal = np.zeros(3, dtype=np.float32)
        self.R_goal = np.eye(3, dtype=np.float32)
        self.step_count = 0

    def _fk_jac_collision(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return self.kin.fk(self.data, q)

    def sample_reachable_pose(
        self, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a reachable, collision-free 6D pose inside the robot's workspace."""
        gen = rng if rng is not None else self.np_random
        return self.kin.sample_target(self.data, gen)

    def sample_initial_q(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a collision-free initial joint configuration."""
        gen = rng if rng is not None else self.np_random
        return self.kin.sample_start(self.data, gen, self.cfg.home_reset_prob)

    def _sync_target_mocap(self) -> None:
        self.data.mocap_pos[self.target_mocap_id] = self.p_goal
        self.data.mocap_quat[self.target_mocap_id] = rotmat_to_quat(self.R_goal)

    def _get_obs(self) -> np.ndarray:
        return compute_obs_batch(
            self.q_cur[None],
            self.p_cur[None],
            self.R_cur[None],
            self.J_cur[None],
            self.prev_action[None],
            np.array([self.prev_collided], dtype=bool),
            self.p_goal[None],
            self.R_goal[None],
            self.q_min,
            self.q_max,
            self.cfg.obs_features,
        )[0]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "target_pos" in options and ("target_rot" in options or "target_quat" in options):
            self.p_goal = np.asarray(options["target_pos"], dtype=np.float32).copy()
            if "target_rot" in options:
                self.R_goal = np.asarray(options["target_rot"], dtype=np.float32).copy()
            else:
                self.R_goal = quat_to_rotmat(np.asarray(options["target_quat"], dtype=np.float32))
        else:
            _, self.p_goal, self.R_goal = self.sample_reachable_pose()

        if "q_init" in options and options["q_init"] is not None:
            self.q_cur = np.clip(np.asarray(options["q_init"], dtype=np.float32), self.q_min, self.q_max)
        else:
            self.q_cur = self.sample_initial_q()

        self.p_cur, self.R_cur, self.J_cur, _ = self._fk_jac_collision(self.q_cur)
        self.prev_action.fill(0.0)
        self.prev_collided = False
        self._sync_target_mocap()
        mujoco.mj_kinematics(self.model, self.data)
        self.step_count = 0

        d_pos = float(np.linalg.norm(self.p_goal - self.p_cur))
        _, d_rot_arr = so3_error_vec(self.R_cur[None], self.R_goal[None])
        d_rot = float(d_rot_arr[0])

        info = {
            "pos_err": d_pos,
            "rot_err": d_rot,
            "is_success": (d_pos < self.cfg.eval_pos_tol) and (d_rot < self.cfg.eval_rot_tol),
            "is_tight_success": (d_pos < self.cfg.term_pos_tol) and (d_rot < self.cfg.term_rot_tol),
            "collided": False,
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_act = self.prev_action.copy()

        dq = action_to_delta_q(
            act[None],
            self.p_cur[None],
            self.R_cur[None],
            self.p_goal[None],
            self.R_goal[None],
            self.cfg,
        )[0]
        collided = bool(
            self.kin.apply_joint_step(
                self.data, self.q_cur[None], self.p_cur[None], self.R_cur[None], self.J_cur[None], dq[None],
                restore_on_collision=True,
            )[0]
        )

        self.prev_action = act.copy()
        self.prev_collided = collided
        self._sync_target_mocap()

        rew_arr, term_arr, d_pos_arr, d_rot_arr = compute_reward_batch(
            self.p_cur[None],
            self.R_cur[None],
            act[None],
            prev_act[None],
            dq[None],
            np.array([collided], dtype=bool),
            self.p_goal[None],
            self.R_goal[None],
            self.cfg,
        )

        reward = float(rew_arr[0])
        terminated = bool(term_arr[0])
        truncated = self.step_count >= self.cfg.max_episode_steps
        d_pos = float(d_pos_arr[0])
        d_rot = float(d_rot_arr[0])

        info = {
            "pos_err": d_pos,
            "rot_err": d_rot,
            "is_success": (not collided) and (d_pos < self.cfg.eval_pos_tol) and (d_rot < self.cfg.eval_rot_tol),
            "is_tight_success": (not collided)
            and (d_pos < self.cfg.term_pos_tol)
            and (d_rot < self.cfg.term_rot_tol),
            "collided": collided,
            "qpos": self.q_cur.copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info

    def make_camera(self, azimuth: float = 135.0, elevation: float = -22.0) -> mujoco.MjvCamera:
        """Default free camera framing this robot's workspace."""
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, cam)
        cam.lookat[:] = self.robot.camera_lookat
        cam.distance = self.robot.camera_distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def render(self, camera_azimuth: float = 135.0, camera_elevation: float = -22.0) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._sync_target_mocap()
        mujoco.mj_forward(self.model, self.data)

        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[1] = 1

        cam = self.make_camera(camera_azimuth, camera_elevation)
        self._renderer.update_scene(self.data, camera=cam, scene_option=opt)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class VectorIKEnv:
    """Fast synchronous vectorized 6D IK environment for any RobotSpec."""

    def __init__(
        self,
        robot: RobotSpec | str,
        num_envs: int,
        cfg: IKConfig | None = None,
        seed: int = 42,
        allow_chain_reset: bool = True,
    ) -> None:
        self.kin = RobotKinematics(robot)
        self.robot = self.kin.robot
        self.num_envs = num_envs
        self.cfg = cfg or IKConfig.for_robot(self.robot)
        self.rng = np.random.default_rng(seed)
        self.allow_chain_reset = allow_chain_reset

        self.model = self.kin.model
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.q_min = self.kin.q_min
        self.q_max = self.kin.q_max
        n = self.kin.n
        self.obs_dim = self.cfg.obs_dim(n)
        self.act_dim = n

        self.q_cur = np.zeros((num_envs, n), dtype=np.float32)
        self.p_cur = np.zeros((num_envs, 3), dtype=np.float32)
        self.R_cur = np.zeros((num_envs, 3, 3), dtype=np.float32)
        self.J_cur = np.zeros((num_envs, 6, n), dtype=np.float32)
        self.prev_action = np.zeros((num_envs, n), dtype=np.float32)
        self.prev_collided = np.zeros(num_envs, dtype=bool)
        self.p_goal = np.zeros((num_envs, 3), dtype=np.float32)
        self.R_goal = np.zeros((num_envs, 3, 3), dtype=np.float32)
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self._initialized = np.zeros(num_envs, dtype=bool)

    def _fk_jac_check_single(
        self, idx: int, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return self.kin.fk(self.datas[idx], q)

    def _reset_single(self, idx: int, force_fresh_q: bool = False) -> None:
        kin = self.kin
        p_g, R_g = None, None
        for _ in range(300):
            q_g = kin.sample_box(self.rng)
            pos, rot, _, col = self._fk_jac_check_single(idx, q_g)
            if (not col) and kin.in_workspace(pos):
                p_g, R_g = pos, rot
                break
        if p_g is None:
            p_g, R_g, _, _ = self._fk_jac_check_single(idx, kin.home_q)

        r = self.rng.random()
        if (
            self.allow_chain_reset
            and (not force_fresh_q)
            and self._initialized[idx]
            and (r < self.cfg.chain_reset_prob)
        ):
            q_0 = self.q_cur[idx].copy()
            p_0, R_0, J_0, _ = self._fk_jac_check_single(idx, q_0)
        else:
            use_home = self.rng.random() < (
                self.cfg.home_reset_prob / max(1e-6, 1.0 - self.cfg.chain_reset_prob)
            )
            q_0, p_0, R_0, J_0 = None, None, None, None
            for _ in range(300):
                q_cand = kin.sample_near_home(self.rng) if use_home else kin.sample_box(self.rng)
                pos, rot, jac, col = self._fk_jac_check_single(idx, q_cand)
                if (not col) and kin.valid_start(pos):
                    q_0, p_0, R_0, J_0 = q_cand, pos, rot, jac
                    break
            if q_0 is None:
                q_0 = kin.home_q.copy()
                p_0, R_0, J_0, _ = self._fk_jac_check_single(idx, q_0)

        self.q_cur[idx] = q_0
        self.p_cur[idx] = p_0
        self.R_cur[idx] = R_0
        self.J_cur[idx] = J_0
        self.prev_action[idx] = 0.0
        self.prev_collided[idx] = False
        self.p_goal[idx] = p_g
        self.R_goal[idx] = R_g
        self.step_counts[idx] = 0
        self._initialized[idx] = True

    def _obs(self) -> np.ndarray:
        return compute_obs_batch(
            self.q_cur,
            self.p_cur,
            self.R_cur,
            self.J_cur,
            self.prev_action,
            self.prev_collided,
            self.p_goal,
            self.R_goal,
            self.q_min,
            self.q_max,
            self.cfg.obs_features,
        )

    def reset(self) -> np.ndarray:
        for i in range(self.num_envs):
            self._reset_single(i, force_fresh_q=True)
        return self._obs()

    _STATE_ARRAYS = (
        "q_cur", "p_cur", "R_cur", "J_cur", "prev_action", "prev_collided",
        "p_goal", "R_goal", "step_counts", "_initialized",
    )

    def get_state(self) -> dict[str, Any]:
        """Episode state of all envs plus the RNG (for resuming training). MjData is recomputed by FK each step."""
        state = {k: getattr(self, k).copy() for k in self._STATE_ARRAYS}
        state["rng"] = self.rng.bit_generator.state
        return state

    def set_state(self, state: dict[str, Any]) -> np.ndarray:
        """Restore get_state() output; returns the current observation."""
        for k in self._STATE_ARRAYS:
            if state[k].shape != getattr(self, k).shape:
                raise ValueError(f"VectorIKEnv state '{k}' has shape {state[k].shape}, expected {getattr(self, k).shape}")
            getattr(self, k)[...] = state[k]
        self.rng.bit_generator.state = state["rng"]
        return self._obs()

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        acts = np.clip(actions, -1.0, 1.0).astype(np.float32)

        prev_act = self.prev_action.copy()
        cur_p_goal = self.p_goal.copy()
        cur_R_goal = self.R_goal.copy()

        dqs = action_to_delta_q(acts, self.p_cur, self.R_cur, cur_p_goal, cur_R_goal, self.cfg)
        collided = self.kin.apply_joint_step(
            self.datas, self.q_cur, self.p_cur, self.R_cur, self.J_cur, dqs, restore_on_collision=True
        )

        self.prev_action = acts.copy()
        self.prev_collided = collided.copy()
        self.step_counts += 1
        pre_reset_steps = self.step_counts.copy()

        rewards, terminated, d_pos, d_rot = compute_reward_batch(
            self.p_cur, self.R_cur, acts, prev_act, dqs, collided, cur_p_goal, cur_R_goal, self.cfg
        )
        truncated = self.step_counts >= self.cfg.max_episode_steps
        dones = terminated | truncated

        # True next observation before auto-reset (stored in the replay buffer)
        terminal_next_obs = compute_obs_batch(
            self.q_cur,
            self.p_cur,
            self.R_cur,
            self.J_cur,
            acts,
            collided,
            cur_p_goal,
            cur_R_goal,
            self.q_min,
            self.q_max,
            self.cfg.obs_features,
        )

        for idx in np.flatnonzero(dones):
            self._reset_single(int(idx))

        info = {
            "p_goal": cur_p_goal,
            "R_goal": cur_R_goal,
            "collided": collided,
            "terminal_next_obs": terminal_next_obs,
            "pos_err": d_pos,
            "rot_err": d_rot,
            "is_success": (~collided)
            & (d_pos < self.cfg.eval_pos_tol)
            & (d_rot < self.cfg.eval_rot_tol),
            "is_tight_success": (~collided)
            & (d_pos < self.cfg.term_pos_tol)
            & (d_rot < self.cfg.term_rot_tol),
            "ep_lengths": pre_reset_steps,
        }
        return self._obs(), rewards, terminated, truncated, info
