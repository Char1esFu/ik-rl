"""Domain core shared by every method and the evaluation: robots, MuJoCo kinematics (CPU / MuJoCo Warp),
DLS refinement and solver building blocks. Depends only on utils/; never imports policy/ or evaluation/."""

from core.kinematics import RobotKinematics
from core.robot import RobotSpec, available_robots, get_robot
from core.solution import IKSolution

__all__ = ["RobotSpec", "available_robots", "get_robot", "RobotKinematics", "IKSolution"]
