"""MuJoCo model of a robot for IK: TCP site, forward kinematics, Jacobian, collision check, samplers.

Shared by every policy (policy/sac_dls, policy/policyik), the evaluation protocols and the viewer.
"""

from __future__ import annotations

import mujoco
import numpy as np

from core.robot import RobotSpec, get_robot


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

    def sample_target(self, data: mujoco.MjData, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reachable target: (q, TCP pos, rot) of a collision-free configuration from the joint box whose TCP lies
        in the workspace (300 tries, else the home pose)."""
        for _ in range(300):
            q = self.sample_box(rng)
            pos, rot, _, collided = self.fk(data, q)
            if (not collided) and self.in_workspace(pos):
                return q, pos, rot
        pos, rot, _, _ = self.fk(data, self.home_q)
        return self.home_q.copy(), pos, rot

    def sample_start(self, data: mujoco.MjData, rng: np.random.Generator, home_prob: float) -> np.ndarray:
        """Collision-free start configuration: near the home pose with probability home_prob, else from the joint
        box; TCP above start_pos_min (300 tries, else the home pose)."""
        use_home = rng.random() < home_prob
        for _ in range(300):
            q = self.sample_near_home(rng) if use_home else self.sample_box(rng)
            pos, _, _, collided = self.fk(data, q)
            if (not collided) and self.valid_start(pos):
                return q
        return self.home_q.copy()
