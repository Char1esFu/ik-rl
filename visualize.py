"""Interactive MuJoCo 3D viewer demonstrating real-time SAC 6D IK tracking.

Supports two modes:
1. Web Viewer (auto-enabled on headless/SSH servers without $DISPLAY, or via --web):
   Hardware-accelerated EGL offscreen rendering streamed over a single port (HTTP + WebSocket)
   with interactive 3D camera orbit/pan/zoom and real-time 6D pose controls in the browser.
2. Native Desktop Viewer (--native, or default when $DISPLAY is available):
   Standard GLFW window via mujoco.viewer.launch_passive.
"""

from __future__ import annotations

import argparse
import asyncio
from http import HTTPStatus
import io
import json
import os
from pathlib import Path
import struct
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo viewer for SAC 6D IK")
    parser.add_argument(
        "--robot", type=str, default=None,
        help="Used to find the newest checkpoint if --checkpoint is not given (default: any robot)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Default: newest runs/*/checkpoints/best_model.pt"
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Parallel multi-start IK branches (1 = single-thread from q_cur, 4 = default with backup branches)",
    )
    parser.add_argument("--hold-sec", type=float, default=1.0, help="Pause at solved target before sampling next")
    parser.add_argument("--web", action="store_true", help="Force browser-based interactive 3D viewer")
    parser.add_argument("--native", action="store_true", help="Force native GLFW desktop window")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host for web viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port for web viewer (auto-forwarded in Remote SSH)")
    parser.add_argument("--width", type=int, default=960, help="Initial web render width (the page then requests its own size)")
    parser.add_argument("--height", type=int, default=640, help="Initial web render height")
    parser.add_argument(
        "--max-width", type=int, default=1920,
        help="Web render resolution cap; the page's size is scaled down to fit (higher = sharper but slower, <= 2560)",
    )
    parser.add_argument("--max-height", type=int, default=1200, help="Web render resolution cap (<= 1600)")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate for web viewer")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality (1-95) for web stream")
    return parser.parse_args()


# Web UI page (HTML/CSS/JS); __ROBOT_NAME__ and __{X,Y,Z}_{MIN,MAX}__ are filled in at startup
WEB_HTML_PATH = Path(__file__).resolve().parent / "web" / "viewer.html"


def rpy_deg_to_rotmat(rpy_deg: list[float] | Any, np_mod: Any) -> Any:
    """Convert extrinsic XYZ (roll, pitch, yaw in degrees) to 3x3 rotation matrix Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = np_mod.radians(np_mod.asarray(rpy_deg, dtype=np_mod.float64))
    cr, sr = np_mod.cos(r), np_mod.sin(r)
    cp, sp = np_mod.cos(p), np_mod.sin(p)
    cy, sy = np_mod.cos(y), np_mod.sin(y)
    rot = np_mod.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np_mod.float32,
    )
    return rot


def rotmat_to_rpy_deg(rot: Any, np_mod: Any) -> list[float]:
    """Extract (roll, pitch, yaw) in degrees from 3x3 rotation matrix."""
    R = np_mod.asarray(rot, dtype=np_mod.float64)
    sp = float(np_mod.clip(-R[2, 0], -1.0, 1.0))
    pitch = float(np_mod.arcsin(sp))
    if abs(sp) < 0.9999:
        roll = float(np_mod.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np_mod.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = float(np_mod.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return [
        round(float(np_mod.degrees(roll)), 1),
        round(float(np_mod.degrees(pitch)), 1),
        round(float(np_mod.degrees(yaw)), 1),
    ]


def figure8_pose(robot: Any, t: float, pivot_xy: Any) -> tuple[Any, Any]:
    """Target pose on the robot's figure-8 demo trajectory at time t (see RobotSpec.fig8_*).

    `pivot_xy` is the first arm joint's axis position, used for fig8_radial_yaw.
    """
    import numpy as np

    c, a = robot.fig8_center, robot.fig8_amplitude
    r0, ra = robot.fig8_rpy_deg, robot.fig8_rpy_amplitude_deg
    px = c[0] + a[0] * np.cos(0.9 * t)
    py = c[1] + a[1] * np.sin(0.9 * t)
    pz = c[2] + a[2] * np.sin(1.8 * t)
    roll = r0[0] + ra[0] * np.sin(0.9 * t)
    pitch = r0[1] + ra[1] * np.cos(0.9 * t)
    yaw = r0[2] + ra[2] * np.sin(0.45 * t)
    if robot.fig8_radial_yaw:
        yaw += np.degrees(np.arctan2(py - pivot_xy[1], px - pivot_xy[0]))
    return np.array([px, py, pz], dtype=np.float32), rpy_deg_to_rotmat([roll, pitch, yaw], np)


class MultiStartIKTracker:
    """Real-time multi-start IK coordinator with collision-free joint-space fallback.

    Seed 0 always tracks directly from the visible arm's current `env.q_cur` so 92%+ of
    targets and incremental slider tweaks move continuously without switching IK branches.
    In parallel, `num_restarts - 1` shadow branches (Home + forward-workspace seeds) run
    batched on GPU. If Seed 0 gets trapped against joint stops or self-collision, the
    arm smoothly slews in joint space to the winning backup branch and fine-parks (<2mm).
    """

    def __init__(self, env: Any, agent: Any, num_restarts: int = 4, seed: int = 42) -> None:
        import mujoco
        import numpy as np


        self.mj = mujoco
        self.np = np
        self.env = env
        self.agent = agent
        self.k_shadow = max(0, int(num_restarts) - 1)
        self.rng = np.random.default_rng(seed)

        self._kin = env.kin
        self._home_q = env.kin.home_q
        n = env.kin.n

        self._sdata = mujoco.MjData(env.model)

        self.s_q = np.zeros((self.k_shadow, n), dtype=np.float32)
        self.s_p = np.zeros((self.k_shadow, 3), dtype=np.float32)
        self.s_R = np.zeros((self.k_shadow, 3, 3), dtype=np.float32)
        self.s_J = np.zeros((self.k_shadow, 6, n), dtype=np.float32)
        self.s_prev_act = np.zeros((self.k_shadow, n), dtype=np.float32)
        self.s_prev_col = np.zeros(self.k_shadow, dtype=bool)

        self.step_in_ep = 0
        self.goal_counted = False
        self.parked = False
        self.slew_target_q: Any = None
        self.via_home = False

        self.main_best_score = float("inf")
        self.main_stagnant = 0
        self.global_best_score = float("inf")
        self.global_stagnant = 0

        self.best_shadow_q: Any = None
        self.best_shadow_score = float("inf")
        self.best_shadow_solved = False
        self.best_shadow_tight = False
        self.on_new_goal()

    def _shadow_fk(self, q: Any) -> tuple[Any, Any, Any, bool]:
        return self._kin.fk(self._sdata, q)

    def _sample_shadow_seed(self, idx: int) -> Any:
        np = self.np
        if idx == 0:
            return self._home_q.copy()
        for _ in range(80):
            cand = self._kin.sample_box(self.rng)
            p_c, _, _, col = self._shadow_fk(cand)
            if (not col) and self._kin.valid_start(p_c):
                return cand
        return self._home_q.copy()

    def on_new_goal(self) -> None:
        from ik_rl.math_utils import so3_error_vec

        np = self.np
        self.step_in_ep = 0
        self.goal_counted = False
        self.parked = False
        self.slew_target_q = None
        self.via_home = False

        d_pos = float(np.linalg.norm(self.env.p_goal - self.env.p_cur))
        _, d_rot_arr = so3_error_vec(self.env.R_cur[None], self.env.R_goal[None])
        d_rot = float(d_rot_arr[0])
        self.main_best_score = d_pos + 0.2 * d_rot
        self.main_stagnant = 0
        self.global_best_score = self.main_best_score
        self.global_stagnant = 0

        self.best_shadow_q = None
        self.best_shadow_score = float("inf")
        self.best_shadow_solved = False
        self.best_shadow_tight = False

        if self.k_shadow > 0:
            self.s_prev_act.fill(0.0)
            self.s_prev_col.fill(False)
            for i in range(self.k_shadow):
                q0 = self._sample_shadow_seed(i)
                p0, R0, J0, _ = self._shadow_fk(q0)
                self.s_q[i], self.s_p[i], self.s_R[i], self.s_J[i] = q0, p0, R0, J0
            self._record_shadow_progress()

    def _record_shadow_progress(self) -> None:
        from ik_rl.math_utils import so3_error_vec

        if self.k_shadow <= 0:
            return
        np = self.np
        p_goals = np.broadcast_to(self.env.p_goal[None], (self.k_shadow, 3))
        R_goals = np.broadcast_to(self.env.R_goal[None], (self.k_shadow, 3, 3))
        d_pos = np.linalg.norm(p_goals - self.s_p, axis=-1)
        _, d_rot = so3_error_vec(self.s_R, R_goals)
        scores = d_pos + 0.2 * d_rot
        idx = int(np.argmin(scores))
        if float(scores[idx]) < self.best_shadow_score:
            self.best_shadow_score = float(scores[idx])
            self.best_shadow_q = self.s_q[idx].copy()
            if (float(d_pos[idx]) < self.env.cfg.eval_pos_tol) and (
                float(d_rot[idx]) < self.env.cfg.eval_rot_tol
            ):
                self.best_shadow_solved = True
            if (float(d_pos[idx]) < self.env.cfg.term_pos_tol) and (
                float(d_rot[idx]) < self.env.cfg.term_rot_tol
            ):
                self.best_shadow_tight = True

    def _step_slew(self) -> tuple[Any, dict[str, Any], float]:
        from ik_rl.math_utils import so3_error_vec

        np = self.np
        env = self.env
        waypoint = self._home_q if self.via_home else self.slew_target_q
        diff = waypoint - env.q_cur
        max_diff = float(np.max(np.abs(diff)))
        slew_rate = 0.16
        dq = diff * min(1.0, slew_rate / max(max_diff, 1e-6))
        q_cand = np.clip(env.q_cur + dq, env.q_min, env.q_max)
        p_c, R_c, J_c, col = env._fk_jac_collision(q_cand)

        if col and (not self.via_home):
            self.via_home = True
            diff = self._home_q - env.q_cur
            max_diff = float(np.max(np.abs(diff)))
            dq = diff * min(1.0, slew_rate / max(max_diff, 1e-6))
            q_cand = np.clip(env.q_cur + dq, env.q_min, env.q_max)
            p_c, R_c, J_c, col = env._fk_jac_collision(q_cand)

        if not col:
            env.q_cur = q_cand
            env.p_cur, env.R_cur, env.J_cur = p_c, R_c, J_c
        else:
            env._fk_jac_collision(env.q_cur)

        env.prev_action.fill(0.0)
        env.prev_collided = False
        env.step_count = 0

        if self.via_home and float(np.max(np.abs(self._home_q - env.q_cur))) < 0.08:
            self.via_home = False
        elif (not self.via_home) and float(np.max(np.abs(self.slew_target_q - env.q_cur))) < 0.03:
            self.slew_target_q = None
            self.main_stagnant = 0

        self.step_in_ep += 1
        d_pos = float(np.linalg.norm(env.p_goal - env.p_cur))
        _, d_rot_arr = so3_error_vec(env.R_cur[None], env.R_goal[None])
        d_rot = float(d_rot_arr[0])
        info = {
            "pos_err": d_pos,
            "rot_err": d_rot,
            "is_success": (d_pos < env.cfg.eval_pos_tol) and (d_rot < env.cfg.eval_rot_tol),
            "is_tight_success": (d_pos < env.cfg.term_pos_tol) and (d_rot < env.cfg.term_rot_tol),
            "collided": False,
        }
        return env._get_obs(), info, 1.0

    def step_frame(self, smooth_alpha: float = 1.0, hold_when_tight: bool = False) -> tuple[Any, dict[str, Any], bool]:
        """Advance one control frame. Returns (obs, info, newly_solved)."""
        from ik_rl.env import action_to_delta_q
        from ik_rl.math_utils import so3_error_vec
        from ik_rl.obs import compute_obs_batch

        np = self.np
        env = self.env

        d_pos0 = float(np.linalg.norm(env.p_goal - env.p_cur))
        _, d_rot0_arr = so3_error_vec(env.R_cur[None], env.R_goal[None])
        d_rot0 = float(d_rot0_arr[0])
        already_tight = (d_pos0 < env.cfg.term_pos_tol) and (d_rot0 < env.cfg.term_rot_tol)

        if hold_when_tight and (self.parked or (already_tight and float(np.max(np.abs(env.prev_action))) < 0.06)):
            self.parked = True
            newly_solved = False
            if already_tight and (not self.goal_counted):
                self.goal_counted = True
                newly_solved = True
            info = {
                "pos_err": d_pos0,
                "rot_err": d_rot0,
                "is_success": (d_pos0 < env.cfg.eval_pos_tol) and (d_rot0 < env.cfg.eval_rot_tol),
                "is_tight_success": already_tight,
                "collided": False,
            }
            return env._get_obs(), info, newly_solved

        if self.slew_target_q is not None:
            obs, info, _ = self._step_slew()
            newly_solved = False
            if info["is_tight_success"] and (not self.goal_counted):
                self.goal_counted = True
                newly_solved = True
            return obs, info, newly_solved

        main_obs = env._get_obs()
        run_shadows = (self.k_shadow > 0) and (self.step_in_ep < 65) and (not already_tight)

        if run_shadows:
            p_goals = np.broadcast_to(env.p_goal[None], (self.k_shadow, 3))
            R_goals = np.broadcast_to(env.R_goal[None], (self.k_shadow, 3, 3))
            s_obs = compute_obs_batch(
                self.s_q,
                self.s_p,
                self.s_R,
                self.s_J,
                self.s_prev_act,
                self.s_prev_col,
                p_goals,
                R_goals,
                env.q_min,
                env.q_max,
                env.cfg.obs_features,
            )
            batch_obs = np.vstack([main_obs[None], s_obs])
            batch_act = self.agent.select_action(batch_obs, deterministic=True)
            raw_act = batch_act[0]
            s_acts = batch_act[1:]

            s_dqs = action_to_delta_q(s_acts, self.s_p, self.s_R, p_goals, R_goals, env.cfg)
            self.s_prev_col[:] = self._kin.apply_joint_step(
                self._sdata, self.s_q, self.s_p, self.s_R, self.s_J, s_dqs
            )
            self.s_prev_act[:] = s_acts
            self._record_shadow_progress()
        else:
            raw_act = self.agent.select_action(main_obs, deterministic=True)

        # Only damp action to zero if BOTH main and all shadow branches have truly stagnated
        # (e.g. manual slider dragged beyond physical arm reach), never on slow valid motion.
        damp = 1.0
        if hold_when_tight and (self.step_in_ep > 45) and (self.global_stagnant > 18):
            damp = float(np.clip(1.0 - (self.global_stagnant - 18) / 15.0, 0.0, 1.0))
            if damp <= 0.0:
                self.parked = True

        env.step_count = 0
        blended = smooth_alpha * raw_act + (1.0 - smooth_alpha) * env.prev_action
        action = damp * blended
        obs, _, _, _, info = env.step(action)
        self.step_in_ep += 1

        d_pos1 = float(info["pos_err"])
        d_rot1 = float(info["rot_err"])
        main_score = d_pos1 + 0.2 * d_rot1
        if main_score < self.main_best_score - 2e-4:
            self.main_best_score = main_score
            self.main_stagnant = 0
        else:
            self.main_stagnant += 1

        global_best_now = min(main_score, self.best_shadow_score)
        if global_best_now < self.global_best_score - 2e-4:
            self.global_best_score = global_best_now
            self.global_stagnant = 0
        else:
            self.global_stagnant += 1

        main_solved = bool(info["is_success"])
        main_tight = bool(info["is_tight_success"])
        if (self.k_shadow > 0) and (not main_tight) and (self.best_shadow_q is not None):
            need_rescue = False
            if (not main_solved) and self.best_shadow_solved and (self.step_in_ep >= 28 or self.main_stagnant >= 10):
                need_rescue = True
            elif self.best_shadow_tight and (self.step_in_ep >= 38) and (self.main_stagnant >= 10):
                need_rescue = True
            elif (
                (not main_solved)
                and (self.step_in_ep >= 38)
                and (self.main_stagnant >= 12)
                and (self.best_shadow_score < 0.55 * main_score)
                and (main_score - self.best_shadow_score > 0.015)
            ):
                need_rescue = True
            if need_rescue:
                self.slew_target_q = self.best_shadow_q.copy()
                self.via_home = False

        newly_solved = False
        settled_tight = bool(info["is_tight_success"]) and (float(np.max(np.abs(action))) < 0.08)
        if (settled_tight or (not hold_when_tight and main_solved)) and (not self.goal_counted):
            if settled_tight:
                self.goal_counted = True
                newly_solved = True

        return obs, info, newly_solved


def run_native_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer
    import numpy as np
    import torch

    from ik_rl.config import IKConfig
    from ik_rl.env import IKEnv
    from ik_rl.sac import SACAgent
    from ik_rl.solver import checkpoint_robot

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent, meta = SACAgent.load(args.checkpoint, device=device)
    cfg = IKConfig.from_dict(meta["env_config"]) if "env_config" in meta else None
    env = IKEnv(checkpoint_robot(meta), cfg=cfg)
    obs, info = env.reset(seed=0)
    tracker = MultiStartIKTracker(env, agent, num_restarts=args.restarts)

    print(f"Launching MuJoCo interactive viewer (restarts={args.restarts})... Close the window to exit.")
    print("Tip: Double-click the yellow target_marker sphere and drag with Ctrl+RightClick/LeftClick in mocap mode!")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = env.robot.camera_lookat
        viewer.cam.distance = env.robot.camera_distance
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        pause_until = 0.0
        while viewer.is_running():
            now = time.perf_counter()
            if now < pause_until:
                viewer.sync()
                time.sleep(0.02)
                continue

            obs, info, _ = tracker.step_frame(smooth_alpha=1.0, hold_when_tight=False)
            viewer.sync()
            time.sleep(0.025)

            settled = info.get("is_tight_success", False) and (
                float(np.max(np.abs(env.prev_action))) < 0.08
            )
            max_ep = 115 if tracker.k_shadow > 0 else env.cfg.max_episode_steps
            if (settled or (tracker.slew_target_q is None and tracker.step_in_ep >= env.cfg.max_episode_steps) or tracker.step_in_ep >= max_ep):
                print(
                    f"Target finished in {tracker.step_in_ep:3d} steps | "
                    f"pos_err={info['pos_err']*1000:.2f} mm | "
                    f"rot_err={np.degrees(info['rot_err']):.2f}°"
                )
                pause_until = time.perf_counter() + args.hold_sec
                cur_q = env.q_cur.copy()
                obs, info = env.reset(options={"q_init": cur_q})
                tracker.on_new_goal()


async def run_web_viewer_async(args: argparse.Namespace) -> None:
    import mujoco
    import numpy as np
    from PIL import Image
    import torch
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response

    from ik_rl.config import IKConfig
    from ik_rl.env import MAX_RENDER_HEIGHT, MAX_RENDER_WIDTH, IKEnv
    from ik_rl.math_utils import so3_error_vec
    from ik_rl.sac import SACAgent
    from ik_rl.solver import checkpoint_robot

    max_w = min(max(320, args.max_width), MAX_RENDER_WIDTH)
    max_h = min(max(240, args.max_height), MAX_RENDER_HEIGHT)

    def fit_render_size(w: float, h: float) -> tuple[int, int]:
        """Scale a requested size down (keeping its aspect ratio) to the resolution cap."""
        scale = min(1.0, max_w / max(w, 1.0), max_h / max(h, 1.0))
        return max(160, int(w * scale)), max(120, int(h * scale))

    width, height = fit_render_size(args.width, args.height)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent, meta = SACAgent.load(args.checkpoint, device=device)
    cfg = IKConfig.from_dict(meta["env_config"]) if "env_config" in meta else None
    env = IKEnv(checkpoint_robot(meta), cfg=cfg)
    obs, info = env.reset(seed=0)
    tracker = MultiStartIKTracker(env, agent, num_restarts=args.restarts)
    # Axis position of the first arm joint (base yaw), the pivot for RobotSpec.fig8_radial_yaw
    first_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.robot.arm_joints[0])
    fig8_pivot = env.data.xanchor[first_joint, :2].copy()

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[1] = 1

    def reset_camera() -> None:
        mujoco.mjv_defaultFreeCamera(env.model, cam)
        cam.lookat[:] = env.robot.camera_lookat
        cam.distance = env.robot.camera_distance
        cam.azimuth = 135.0
        cam.elevation = -22.0

    reset_camera()

    state: dict[str, Any] = {
        "mode": "auto",  # "auto" | "manual" | "figure8"
        "paused": False,
        "waiting_next": False,
        "pause_until": 0.0,
        "step_in_ep": 0,
        "solved_count": 0,
        "sync_seq": 0,
        "fig8_t0": time.perf_counter(),
        "paused_at": 0.0,
        "render_size": (width, height),  # requested by the page (fit to the cap)
        "obs": obs,
        "info": info,
    }
    clients: set[ServerConnection] = set()
    lo, hi = env.kin.ws_low, env.kin.ws_high
    html = WEB_HTML_PATH.read_text(encoding="utf-8").replace("__ROBOT_NAME__", env.robot.display_name)
    for i, axis in enumerate("XYZ"):
        html = html.replace(f"__{axis}_MIN__", f"{lo[i]:.3f}").replace(f"__{axis}_MAX__", f"{hi[i]:.3f}")
    html_bytes = html.encode("utf-8")

    def compute_errors() -> tuple[float, float]:
        d_pos = float(np.linalg.norm(env.p_goal - env.p_cur))
        _, d_rot_arr = so3_error_vec(env.R_cur[None], env.R_goal[None])
        return d_pos, float(d_rot_arr[0])

    def set_goal_pose(pos: np.ndarray, rot: np.ndarray, reset_tracker: bool = True) -> None:
        env.p_goal = pos.astype(np.float32).copy()
        env.R_goal = rot.astype(np.float32).copy()
        env.prev_action.fill(0.0)
        env.prev_collided = False
        env._sync_target_mocap()
        env.step_count = 0
        state["step_in_ep"] = 0
        state["obs"] = env._get_obs()
        if reset_tracker:
            tracker.on_new_goal()

    def handle_command(cmd: dict[str, Any]) -> None:
        ctype = cmd.get("type")
        if ctype == "camera_move":
            act_name = cmd.get("action", "rotate")
            dx = float(cmd.get("dx", 0.0))
            dy = float(cmd.get("dy", 0.0))
            if act_name == "pan":
                mj_act = mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif act_name == "zoom":
                mj_act = mujoco.mjtMouse.mjMOUSE_ZOOM
            else:
                mj_act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
            mujoco.mjv_moveCamera(env.model, mj_act, dx, dy, cam)
            cam.distance = float(np.clip(cam.distance, 0.35, 4.5))

        elif ctype == "resize":
            state["render_size"] = fit_render_size(float(cmd["width"]), float(cmd["height"]))

        elif ctype == "camera_reset":
            reset_camera()

        elif ctype == "set_mode":
            mode = cmd.get("mode", "auto")
            if mode in ("auto", "manual", "figure8"):
                state["mode"] = mode
                state["waiting_next"] = False
                state["pause_until"] = 0.0
                state["fig8_t0"] = time.perf_counter()
                state["sync_seq"] += 1
                if mode != "figure8":
                    tracker.on_new_goal()

        elif ctype == "set_target":
            state["mode"] = "manual"
            state["waiting_next"] = False
            pos = np.asarray(cmd["pos"], dtype=np.float32)
            pos = np.clip(pos, env.kin.ws_low, env.kin.ws_high).astype(np.float32)
            rot = rpy_deg_to_rotmat(cmd["rpy_deg"], np)
            set_goal_pose(pos, rot, reset_tracker=True)

        elif ctype == "next_target":
            # A manually requested target: stop automatic targets (auto / figure-8) and hold this one
            state["mode"] = "manual"
            _, p_new, R_new = env.sample_reachable_pose()
            set_goal_pose(p_new, R_new, reset_tracker=True)
            state["waiting_next"] = False
            state["pause_until"] = 0.0
            state["sync_seq"] += 1

        elif ctype == "point_down":
            state["mode"] = "manual"
            state["waiting_next"] = False
            rot = rpy_deg_to_rotmat([180.0, 0.0, 0.0], np)
            set_goal_pose(env.p_goal.copy(), rot, reset_tracker=True)
            state["sync_seq"] += 1

        elif ctype == "toggle_pause":
            now = time.perf_counter()
            state["paused"] = not state["paused"]
            if state["paused"]:
                state["paused_at"] = now
            else:
                # Freeze time while paused: the figure-8 target and the auto-mode hold timer resume
                # where they stopped instead of jumping ahead by the paused duration.
                state["fig8_t0"] += now - max(state["paused_at"], state["fig8_t0"])
                state["pause_until"] += now - state["paused_at"]

        elif ctype == "reset_arm":
            # Full reset: arm and target both at the home pose, automatic targets stopped (manual mode)
            q_home = env.kin.home_q.copy()
            p_home, R_home, _, _ = env.kin.fk(env.data, q_home)
            state["obs"], state["info"] = env.reset(
                options={"target_pos": p_home, "target_rot": R_home, "q_init": q_home}
            )
            state["mode"] = "manual"
            state["waiting_next"] = False
            state["pause_until"] = 0.0
            state["step_in_ep"] = 0
            state["sync_seq"] += 1
            tracker.on_new_goal()

    def process_request(conn: ServerConnection, request: Request) -> Response | None:
        if request.path == "/ws":
            return None
        if request.path == "/favicon.ico":
            return Response(int(HTTPStatus.NO_CONTENT), "No Content", Headers())
        headers = Headers(
            [
                ("Content-Type", "text/html; charset=utf-8"),
                ("Cache-Control", "no-store"),
            ]
        )
        return Response(int(HTTPStatus.OK), "OK", headers, html_bytes)

    async def ws_handler(ws: ServerConnection) -> None:
        clients.add(ws)
        try:
            async for message in ws:
                if isinstance(message, str):
                    try:
                        handle_command(json.loads(message))
                    except Exception:
                        pass
        finally:
            clients.discard(ws)

    async with serve(
        ws_handler,
        args.host,
        args.port,
        process_request=process_request,
        compression=None,
        max_queue=4,
    ):
        display_url = f"http://localhost:{args.port}"
        print("=" * 68)
        print(f"  MuJoCo Web 3D Viewer is running at: {display_url} (restarts={args.restarts})")
        print("  (In VS Code / Antigravity Remote SSH, Ctrl+Click the URL above)")
        print("  Press Ctrl+C to stop.")
        print("=" * 68, flush=True)

        dt_target = 1.0 / max(5, args.fps)
        jpeg_buf = io.BytesIO()

        try:
            while True:
                t_start = time.perf_counter()

                if clients and not state["paused"]:
                    mode = state["mode"]
                    if mode == "figure8":
                        t = t_start - state["fig8_t0"]
                        prev_a = env.prev_action.copy()
                        set_goal_pose(*figure8_pose(env.robot, t, fig8_pivot), reset_tracker=False)
                        env.prev_action[:] = prev_a
                        state["obs"] = env._get_obs()
                        raw_act = agent.select_action(state["obs"], deterministic=True)
                        action = 0.75 * raw_act + 0.25 * prev_a
                        state["obs"], _, _, _, state["info"] = env.step(action)
                        state["step_in_ep"] += 1

                    elif mode == "manual":
                        state["obs"], state["info"], newly_solved = tracker.step_frame(
                            smooth_alpha=0.85, hold_when_tight=True
                        )
                        state["step_in_ep"] = tracker.step_in_ep
                        if newly_solved:
                            state["solved_count"] += 1

                    else:  # "auto"
                        if state["waiting_next"]:
                            if t_start >= state["pause_until"]:
                                state["waiting_next"] = False
                                cur_q = env.q_cur.copy()
                                state["obs"], state["info"] = env.reset(options={"q_init": cur_q})
                                tracker.on_new_goal()
                                state["step_in_ep"] = 0
                                state["sync_seq"] += 1
                        else:
                            state["obs"], state["info"], _ = tracker.step_frame(
                                smooth_alpha=1.0, hold_when_tight=False
                            )
                            state["step_in_ep"] = tracker.step_in_ep
                            settled = state["info"].get("is_tight_success", False) and (
                                float(np.max(np.abs(env.prev_action))) < 0.08
                            )
                            max_ep = 115 if tracker.k_shadow > 0 else env.cfg.max_episode_steps
                            done_ep = (
                                settled
                                or (
                                    tracker.slew_target_q is None
                                    and tracker.step_in_ep >= env.cfg.max_episode_steps
                                )
                                or (tracker.step_in_ep >= max_ep)
                            )
                            if done_ep:
                                if settled or state["info"].get("is_success", False):
                                    state["solved_count"] += 1
                                state["waiting_next"] = True
                                state["pause_until"] = time.perf_counter() + args.hold_sec

                if clients:
                    env._sync_target_mocap()
                    mujoco.mj_forward(env.model, env.data)
                    if state["render_size"] != (width, height):
                        width, height = state["render_size"]
                        renderer.close()
                        renderer = mujoco.Renderer(env.model, height=height, width=width)
                    renderer.update_scene(env.data, camera=cam, scene_option=opt)
                    rgb = renderer.render()

                    jpeg_buf.seek(0)
                    jpeg_buf.truncate(0)
                    Image.fromarray(rgb).save(jpeg_buf, format="JPEG", quality=args.jpeg_quality)
                    jpeg_bytes = jpeg_buf.getvalue()

                    d_pos, d_rot = compute_errors()
                    meta = {
                        "mode": state["mode"],
                        "paused": state["paused"],
                        "pos_err_mm": d_pos * 1000.0,
                        "rot_err_deg": float(np.degrees(d_rot)),
                        "is_success": (d_pos < env.cfg.eval_pos_tol) and (d_rot < env.cfg.eval_rot_tol),
                        "is_tight": (d_pos < env.cfg.term_pos_tol) and (d_rot < env.cfg.term_rot_tol),
                        "step": state["step_in_ep"],
                        "solved_count": state["solved_count"],
                        "sync_seq": state["sync_seq"],
                        "p_cur": [round(float(x), 4) for x in env.p_cur],
                        "p_goal": [round(float(x), 4) for x in env.p_goal],
                        "rpy_goal_deg": rotmat_to_rpy_deg(env.R_goal, np),
                        "q": [round(float(x), 3) for x in env.q_cur],
                    }
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    packet = struct.pack("<I", len(meta_bytes)) + meta_bytes + jpeg_bytes

                    stale: list[ServerConnection] = []
                    for ws in list(clients):
                        try:
                            await ws.send(packet)
                        except Exception:
                            stale.append(ws)
                    for ws in stale:
                        clients.discard(ws)

                elapsed = time.perf_counter() - t_start
                await asyncio.sleep(max(0.002, dt_target - elapsed))
        finally:
            renderer.close()
            env.close()


def main() -> None:
    args = parse_args()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_web = args.web or (not args.native and not has_display)

    if use_web:
        os.environ.setdefault("MUJOCO_GL", "egl")

    # Imported after MUJOCO_GL is set (ik_rl pulls in mujoco)
    if args.checkpoint is None:
        from ik_rl.runs import find_latest_checkpoint

        args.checkpoint = str(find_latest_checkpoint(args.robot))
    print(f"Using checkpoint: {args.checkpoint}")

    if use_web:
        try:
            asyncio.run(run_web_viewer_async(args))
        except KeyboardInterrupt:
            print("\nViewer stopped.")
    else:
        run_native_viewer(args)


if __name__ == "__main__":
    main()
