"""Offscreen rendering of a robot configuration with its target marker (eval galleries; any policy).

Takes a core.kinematics.RobotKinematics (model with TCP site and target marker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from utils.math_utils import rotmat_to_quat

if TYPE_CHECKING:  # utils/ does not import core/ at runtime (core depends on utils)
    from core.kinematics import RobotKinematics


def make_camera(kin: RobotKinematics, azimuth: float = 135.0, elevation: float = -22.0) -> mujoco.MjvCamera:
    """Default free camera framing the robot's workspace (camera_lookat / camera_distance of its config)."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(kin.model, cam)
    cam.lookat[:] = kin.robot.camera_lookat
    cam.distance = kin.robot.camera_distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


class PoseRenderer:
    """render(q, p_goal, R_goal) -> RGB image of the arm at q, with the target marker and TCP axes."""

    def __init__(self, kin: RobotKinematics, width: int = 640, height: int = 520) -> None:
        self.kin = kin
        self.data = mujoco.MjData(self.kin.model)
        self.renderer = mujoco.Renderer(self.kin.model, height=height, width=width)
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.opt)
        self.opt.geomgroup[1] = 1  # TCP axes and target marker

    def render(self, q: np.ndarray, p_goal: np.ndarray, R_goal: np.ndarray, azimuth: float = 135.0,
               elevation: float = -22.0) -> np.ndarray:
        kin, data = self.kin, self.data
        data.qpos[kin.arm_qadr] = q
        data.qpos[kin.fixed_qadr] = kin.fixed_val
        data.mocap_pos[kin.target_mocap_id] = p_goal
        data.mocap_quat[kin.target_mocap_id] = rotmat_to_quat(np.asarray(R_goal))
        mujoco.mj_forward(kin.model, data)
        self.renderer.update_scene(data, camera=make_camera(kin, azimuth, elevation), scene_option=self.opt)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()
