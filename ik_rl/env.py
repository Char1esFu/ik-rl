"""Robot-agnostic MuJoCo 6D inverse kinematics environments (single + vectorized).

Configuration lives in ik_rl/config.py (IKConfig), the observation in ik_rl/obs.py.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ik_rl.config import IKConfig
from ik_rl.math_utils import quat_to_rotmat, rotmat_to_quat, so3_error_vec
from ik_rl.obs import compute_obs_batch
from ik_rl.robot import RobotSpec, get_robot


def resolve_robot(robot: RobotSpec | str) -> RobotSpec:
    return get_robot(robot) if isinstance(robot, str) else robot


# Offscreen framebuffer size: upper bound for any mujoco.Renderer of the IK model (e.g. the web viewer)
MAX_RENDER_WIDTH = 2560
MAX_RENDER_HEIGHT = 1600


def build_ik_model(robot: RobotSpec) -> mujoco.MjModel:
    """Compile the robot scene with a TCP site, RGB axis markers and a mocap target marker."""
    spec = mujoco.MjSpec.from_file(str(robot.scene_xml))
    spec.visual.global_.offwidth = MAX_RENDER_WIDTH
    spec.visual.global_.offheight = MAX_RENDER_HEIGHT

    ee_body = spec.body(robot.ee_body)
    offset = np.asarray(robot.ee_offset, dtype=np.float64)
    quat = np.asarray(robot.ee_quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    ee_body.add_site(
        name="ee_site",
        pos=offset.tolist(),
        quat=quat.tolist(),
        size=[0.008, 0.008, 0.008],
        rgba=[0.0, 1.0, 1.0, 0.9],
    )
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    axis_len = 0.045
    axis_rad = 0.003
    axis_colors = ([1.0, 0.2, 0.2, 0.85], [0.2, 1.0, 0.2, 0.85], [0.2, 0.4, 1.0, 0.85])
    for i, (axis, rgba) in enumerate(zip("xyz", axis_colors)):
        tip = offset + axis_len * rot[:, i]
        ee_body.add_geom(
            name=f"ee_axis_{axis}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*offset.tolist(), *tip.tolist()],
            size=[axis_rad, 0, 0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=1,
        )

    target_body = spec.worldbody.add_body(name="target_marker", mocap=True, pos=[0.5, 0.0, 0.5])
    target_body.add_geom(
        name="target_center",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.011, 0, 0],
        rgba=[1.0, 0.85, 0.0, 0.75],
        contype=0,
        conaffinity=0,
        group=1,
    )
    t_len = 0.065
    t_rad = 0.0045
    target_colors = ([1.0, 0.0, 0.0, 0.65], [0.0, 1.0, 0.0, 0.65], [0.0, 0.3, 1.0, 0.65])
    for i, (axis, rgba) in enumerate(zip("xyz", target_colors)):
        tip = [0.0, 0.0, 0.0]
        tip[i] = t_len
        target_body.add_geom(
            name=f"target_axis_{axis}",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, *tip],
            size=[t_rad, 0, 0],
            rgba=rgba,
            contype=0,
            conaffinity=0,
            group=1,
        )

    return spec.compile()


class RobotKinematics:
    """Compiled model for a RobotSpec plus FK / Jacobian / collision queries on any MjData."""

    def __init__(self, robot: RobotSpec | str) -> None:
        self.robot = resolve_robot(robot)
        r = self.robot
        self.model = build_ik_model(r)
        m = self.model
        self.n = r.n_joints

        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in r.arm_joints]
        if min(jids) < 0:
            raise ValueError(f"{r.name}: arm joint not found in {r.scene_xml}")
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in jids], dtype=np.int64)
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in jids], dtype=np.int64)
        self.q_min = m.jnt_range[jids, 0].astype(np.float32)
        self.q_max = m.jnt_range[jids, 1].astype(np.float32)

        fixed_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j, _ in r.fixed_joints]
        self.fixed_qadr = np.array([m.jnt_qposadr[j] for j in fixed_ids], dtype=np.int64)
        self.fixed_val = np.array([v for _, v in r.fixed_joints], dtype=np.float32)

        self.home_q = np.array(r.home_q, dtype=np.float32)
        self.sample_low = np.array(r.sample_q_low, dtype=np.float32)
        self.sample_high = np.array(r.sample_q_high, dtype=np.float32)
        self.ws_low = np.array(r.workspace_low, dtype=np.float64)
        self.ws_high = np.array(r.workspace_high, dtype=np.float64)
        self.start_min = np.array(r.start_pos_min, dtype=np.float64)

        self.ee_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        target_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        self.target_mocap_id = int(m.body_mocapid[target_body_id])

        self._jacp = np.zeros((3, m.nv), dtype=np.float64)
        self._jacr = np.zeros((3, m.nv), dtype=np.float64)

    def fk(self, data: mujoco.MjData, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Set arm joints to q; return TCP position, rotation, 6xn Jacobian and collision flag."""
        data.qpos[self.arm_qadr] = q
        data.qpos[self.fixed_qadr] = self.fixed_val
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)  # required by mj_jacSite (cdof); without it J == 0
        mujoco.mj_collision(self.model, data)
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.ee_site_id)
        pos = data.site_xpos[self.ee_site_id].copy().astype(np.float32)
        rot = data.site_xmat[self.ee_site_id].reshape(3, 3).copy().astype(np.float32)
        jac = np.vstack([self._jacp[:, self.arm_dadr], self._jacr[:, self.arm_dadr]]).astype(np.float32)
        return pos, rot, jac, bool(data.ncon > 0)

    def apply_joint_step(
        self,
        data: mujoco.MjData | list[mujoco.MjData],
        q: np.ndarray,
        p: np.ndarray,
        R: np.ndarray,
        J: np.ndarray,
        dq: np.ndarray,
        restore_on_collision: bool = False,
    ) -> np.ndarray:
        """One kinematic step for a batch: q <- clip(q + dq) unless the new configuration collides.

        Shared by IKEnv, VectorIKEnv, the solver rollout and the viewer. q (B, n), p (B, 3), R (B, 3, 3) and
        J (B, 6, n) are updated in place; rows whose candidate collides stay where they are. `data` is one
        MjData (reused for every row) or one per row. With restore_on_collision, a colliding row's MjData is
        recomputed at its current q (keeps MjData consistent for rendering). Returns the collided mask (B,).
        """
        q_cands = np.clip(q + dq, self.q_min, self.q_max)
        collided = np.zeros(q.shape[0], dtype=bool)
        for i in range(q.shape[0]):
            d = data[i] if isinstance(data, (list, tuple)) else data
            pos, rot, jac, col = self.fk(d, q_cands[i])
            if not col:
                q[i], p[i], R[i], J[i] = q_cands[i], pos, rot, jac
            else:
                collided[i] = True
                if restore_on_collision:
                    self.fk(d, q[i])
        return collided

    def in_workspace(self, pos: np.ndarray) -> bool:
        return bool(np.all(self.ws_low <= pos) and np.all(pos <= self.ws_high))

    def valid_start(self, pos: np.ndarray) -> bool:
        return bool(np.all(pos >= self.start_min))

    def sample_box(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.sample_low, self.sample_high).astype(np.float32)

    def sample_near_home(self, rng: np.random.Generator) -> np.ndarray:
        noise = rng.normal(0.0, self.robot.home_noise_std, size=self.n).astype(np.float32)
        return np.clip(self.home_q + noise, self.q_min, self.q_max)


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
        for _ in range(300):
            q = self.kin.sample_box(gen)
            pos, rot, _, collided = self._fk_jac_collision(q)
            if (not collided) and self.kin.in_workspace(pos):
                return q, pos, rot
        pos, rot, _, _ = self._fk_jac_collision(self.kin.home_q)
        return self.kin.home_q.copy(), pos, rot

    def sample_initial_q(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a collision-free initial joint configuration."""
        gen = rng if rng is not None else self.np_random
        use_home = gen.random() < self.cfg.home_reset_prob
        for _ in range(300):
            q = self.kin.sample_near_home(gen) if use_home else self.kin.sample_box(gen)
            pos, _, _, collided = self._fk_jac_collision(q)
            if (not collided) and self.kin.valid_start(pos):
                return q
        return self.kin.home_q.copy()

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
