"""Interactive viewers for any IK method: native GLFW window or browser (EGL rendering streamed over WebSocket).

Both drive a MultiStartTracker built from a controller factory; the visible arm is rendered from an MjData kept
in sync with the tracker. Modes (web): auto (random targets), manual (sliders), figure-8 (moving goal).
"""

from __future__ import annotations

import argparse
import asyncio
from http import HTTPStatus
import io
import json
from pathlib import Path
import struct
import time
from typing import Any, Callable

import mujoco
import numpy as np

from core.controller import Controller
from core.kinematics import MAX_RENDER_HEIGHT, MAX_RENDER_WIDTH, RobotKinematics
from core.robot import RobotSpec
from evaluation.viewer.tracker import MultiStartTracker
from utils.math_utils import rotmat_to_quat, rotmat_to_rpy_deg, rpy_deg_to_rotmat

# Web UI page (HTML/CSS/JS); __ROBOT_NAME__, __METHOD__ and __{X,Y,Z}_{MIN,MAX}__ are filled in at startup
WEB_HTML_PATH = Path(__file__).resolve().parent / "viewer.html"


def figure8_pose(robot: RobotSpec, t: float, pivot_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target pose on the robot's figure-8 demo trajectory at time t (see RobotSpec.fig8_*).

    `pivot_xy` is the first arm joint's axis position, used for fig8_radial_yaw.
    """
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
    return np.array([px, py, pz], dtype=np.float32), rpy_deg_to_rotmat([roll, pitch, yaw])


class Scene:
    """Robot kinematics, the tracker and an MjData mirroring the visible arm and the target marker."""

    def __init__(self, make_controller: Callable[[], Controller], robot: RobotSpec, restarts: int) -> None:
        self.kin = RobotKinematics(robot)
        self.robot = self.kin.robot
        self.model = self.kin.model
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(0)
        q0 = self.kin.sample_start(self.data, self.rng, home_prob=0.25)
        _, p_goal, R_goal = self.kin.sample_target(self.data, self.rng)
        self.tracker = MultiStartTracker(make_controller, self.kin, q0, p_goal, R_goal, num_restarts=restarts)
        self.sync()

    def sync(self) -> None:
        """Copy the visible arm and the target into the MjData (for rendering)."""
        t = self.tracker
        self.kin.fk(self.data, t.q)
        self.data.mocap_pos[self.kin.target_mocap_id] = t.p_goal
        self.data.mocap_quat[self.kin.target_mocap_id] = rotmat_to_quat(t.R_goal)
        mujoco.mj_forward(self.model, self.data)

    def random_target(self) -> tuple[np.ndarray, np.ndarray]:
        _, p, R = self.kin.sample_target(self.data, self.rng)
        return p, R


# ---------------------------------------------------------------------------------------------------- native


def run_native_viewer(scene: Scene, args: argparse.Namespace) -> None:
    import mujoco.viewer

    tracker = scene.tracker
    print(f"Launching MuJoCo interactive viewer (restarts={args.restarts})... Close the window to exit.")
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = scene.robot.camera_lookat
        viewer.cam.distance = scene.robot.camera_distance
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0

        pause_until = 0.0
        while viewer.is_running():
            if time.perf_counter() < pause_until:
                viewer.sync()
                time.sleep(0.02)
                continue
            info, _ = tracker.step_frame(smooth_alpha=1.0, hold_when_tight=False)
            scene.sync()
            viewer.sync()
            time.sleep(0.025)
            if tracker.episode_done():
                print(
                    f"Target finished in {tracker.step_in_ep:3d} steps | "
                    f"pos_err={info['pos_err'] * 1000:.2f} mm | rot_err={np.degrees(info['rot_err']):.2f}°"
                )
                pause_until = time.perf_counter() + args.hold_sec
                tracker.set_goal(*scene.random_target())


# ---------------------------------------------------------------------------------------------------- web


async def run_web_viewer_async(scene: Scene, args: argparse.Namespace, method_label: str) -> None:
    from PIL import Image
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response

    max_w = min(max(320, args.max_width), MAX_RENDER_WIDTH)
    max_h = min(max(240, args.max_height), MAX_RENDER_HEIGHT)

    def fit_render_size(w: float, h: float) -> tuple[int, int]:
        """Scale a requested size down (keeping its aspect ratio) to the resolution cap."""
        scale = min(1.0, max_w / max(w, 1.0), max_h / max(h, 1.0))
        return max(160, int(w * scale)), max(120, int(h * scale))

    width, height = fit_render_size(args.width, args.height)
    kin, model, data, tracker = scene.kin, scene.model, scene.data, scene.tracker
    # Axis position of the first arm joint (base yaw), the pivot for RobotSpec.fig8_radial_yaw
    kin.fk(data, kin.home_q)
    first_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, scene.robot.arm_joints[0])
    fig8_pivot = data.xanchor[first_joint, :2].copy()
    scene.sync()

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[1] = 1

    def reset_camera() -> None:
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = scene.robot.camera_lookat
        cam.distance = scene.robot.camera_distance
        cam.azimuth = 135.0
        cam.elevation = -22.0

    reset_camera()

    state: dict[str, Any] = {
        "mode": "auto",  # "auto" | "manual" | "figure8"
        "paused": False,
        "waiting_next": False,
        "pause_until": 0.0,
        "solved_count": 0,
        "sync_seq": 0,
        "fig8_t0": time.perf_counter(),
        "paused_at": 0.0,
        "render_size": (width, height),  # requested by the page (fit to the cap)
    }
    clients: set[ServerConnection] = set()
    lo, hi = kin.ws_low, kin.ws_high
    html = (WEB_HTML_PATH.read_text(encoding="utf-8").replace("__ROBOT_NAME__", scene.robot.display_name)
            .replace("__METHOD__", method_label))
    for i, axis in enumerate("XYZ"):
        html = html.replace(f"__{axis}_MIN__", f"{lo[i]:.3f}").replace(f"__{axis}_MAX__", f"{hi[i]:.3f}")
    html_bytes = html.encode("utf-8")

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
            mujoco.mjv_moveCamera(model, mj_act, dx, dy, cam)
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
            pos = np.clip(np.asarray(cmd["pos"], dtype=np.float32), kin.ws_low, kin.ws_high).astype(np.float32)
            tracker.set_goal(pos, rpy_deg_to_rotmat(cmd["rpy_deg"]))

        elif ctype == "next_target":
            # A manually requested target: stop automatic targets (auto / figure-8) and hold this one
            state["mode"] = "manual"
            tracker.set_goal(*scene.random_target())
            state["waiting_next"] = False
            state["pause_until"] = 0.0
            state["sync_seq"] += 1

        elif ctype == "point_down":
            state["mode"] = "manual"
            state["waiting_next"] = False
            tracker.set_goal(tracker.p_goal.copy(), rpy_deg_to_rotmat([180.0, 0.0, 0.0]))
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
            q_home = kin.home_q.copy()
            p_home, R_home, _, _ = kin.fk(data, q_home)
            tracker.reset_arm(q_home, p_home, R_home)
            state["mode"] = "manual"
            state["waiting_next"] = False
            state["pause_until"] = 0.0
            state["sync_seq"] += 1

    def process_request(conn: ServerConnection, request: Request) -> Response | None:
        if request.path == "/ws":
            return None
        if request.path == "/favicon.ico":
            return Response(int(HTTPStatus.NO_CONTENT), "No Content", Headers())
        headers = Headers([("Content-Type", "text/html; charset=utf-8"), ("Cache-Control", "no-store")])
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

    async with serve(ws_handler, args.host, args.port, process_request=process_request, compression=None,
                     max_queue=4):
        print("=" * 68)
        print(f"  MuJoCo Web 3D Viewer ({method_label}) is running at: http://localhost:{args.port} "
              f"(restarts={args.restarts})")
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
                        tracker.set_goal(*figure8_pose(scene.robot, t_start - state["fig8_t0"], fig8_pivot),
                                         new_goal=False)
                        tracker.step_direct(smooth=0.75)
                    elif mode == "manual":
                        _, newly_solved = tracker.step_frame(smooth_alpha=0.85, hold_when_tight=True)
                        if newly_solved:
                            state["solved_count"] += 1
                    elif state["waiting_next"]:  # auto: holding a finished target
                        if t_start >= state["pause_until"]:
                            state["waiting_next"] = False
                            tracker.set_goal(*scene.random_target())
                            state["sync_seq"] += 1
                    else:  # auto
                        info, _ = tracker.step_frame(smooth_alpha=1.0, hold_when_tight=False)
                        if tracker.episode_done():
                            if tracker.settled() or info["is_success"]:
                                state["solved_count"] += 1
                            state["waiting_next"] = True
                            state["pause_until"] = time.perf_counter() + args.hold_sec

                if clients:
                    scene.sync()
                    if state["render_size"] != (width, height):
                        width, height = state["render_size"]
                        renderer.close()
                        renderer = mujoco.Renderer(model, height=height, width=width)
                    renderer.update_scene(data, camera=cam, scene_option=opt)
                    rgb = renderer.render()

                    jpeg_buf.seek(0)
                    jpeg_buf.truncate(0)
                    Image.fromarray(rgb).save(jpeg_buf, format="JPEG", quality=args.jpeg_quality)
                    info = tracker.info()
                    meta = {
                        "mode": state["mode"],
                        "paused": state["paused"],
                        "pos_err_mm": info["pos_err"] * 1000.0,
                        "rot_err_deg": float(np.degrees(info["rot_err"])),
                        "is_success": info["is_success"],
                        "is_tight": info["is_tight_success"],
                        "step": tracker.step_in_ep,
                        "solved_count": state["solved_count"],
                        "sync_seq": state["sync_seq"],
                        "p_cur": [round(float(x), 4) for x in tracker.p],
                        "p_goal": [round(float(x), 4) for x in tracker.p_goal],
                        "rpy_goal_deg": rotmat_to_rpy_deg(tracker.R_goal),
                        "q": [round(float(x), 3) for x in tracker.q],
                    }
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    packet = struct.pack("<I", len(meta_bytes)) + meta_bytes + jpeg_buf.getvalue()

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


def run_viewer(make_controller: Callable[[], Controller], robot: RobotSpec, method_label: str,
               args: argparse.Namespace, web: bool) -> None:
    """Start the web (web=True) or native viewer for controllers from `make_controller`."""
    scene = Scene(make_controller, robot, args.restarts)
    if web:
        try:
            asyncio.run(run_web_viewer_async(scene, args, method_label))
        except KeyboardInterrupt:
            print("\nViewer stopped.")
    else:
        run_native_viewer(scene, args)
