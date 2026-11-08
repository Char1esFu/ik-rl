"""This repository's benchmark protocol, shared by every policy, train.py, eval.py and the comparison scripts.

- EvalTargets: fixed evaluation problems (reachable target poses + start configurations), drawn with the
  shared samplers of RobotKinematics, so every method sees the same targets for a given seed.
- evaluate_solver: fixed targets with start configurations, solved with a policy's solver (stop once within
  2 mm / 0.02 rad, at most `max_steps` steps, `num_restarts` branches). Training curves (1 start) and eval.py
  (1 and 4 starts) use it, so all reported numbers are comparable.
PolicyIK's hold protocol (reach and hold for 1 s) is in evaluation/hold.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from core.kinematics import RobotKinematics
from core.robot import RobotSpec

PRECISE_POS_TOL = 1e-4  # 0.1 mm
PRECISE_ROT_TOL = float(np.radians(1e-3))  # 0.001 deg

EVAL_HOME_PROB = 0.25  # share of start configurations near the home pose

if TYPE_CHECKING:
    from core.solution import IKSolution


@dataclass
class EvalTargets:
    """Fixed evaluation problems: reachable target poses and start configurations."""

    pos: np.ndarray  # (N, 3)
    rot: np.ndarray  # (N, 3, 3)
    q_init: np.ndarray  # (N, n)

    @classmethod
    def sample(
        cls, robot: RobotSpec | str | RobotKinematics, num: int, rng: np.random.Generator,
        home_prob: float = EVAL_HOME_PROB,
    ) -> EvalTargets:
        """Per problem: a reachable target (RobotKinematics.sample_target), then a start (sample_start)."""
        kin = robot if isinstance(robot, RobotKinematics) else RobotKinematics(robot)
        data = mujoco.MjData(kin.model)
        pos, rot, q_init = [], [], []
        for _ in range(num):
            _, p, R = kin.sample_target(data, rng)
            pos.append(p)
            rot.append(R)
            q_init.append(kin.sample_start(data, rng, home_prob))
        return cls(np.array(pos), np.array(rot), np.array(q_init))

    def __len__(self) -> int:
        return len(self.pos)


def solution_metrics(solutions: list[IKSolution], pos_tol: float, rot_tol: float,
                     tight_pos_tol: float, tight_rot_tol: float) -> dict[str, float]:
    """Success rates (%) at the loose / tight tolerances, error statistics and mean steps."""
    pe = np.array([s.pos_err_m for s in solutions])
    re = np.array([s.rot_err_rad for s in solutions])
    pe_mm, re_deg = pe * 1000.0, np.degrees(re)
    return {
        "success_rate_5mm": float(np.mean((pe < pos_tol) & (re < rot_tol)) * 100.0),
        "success_rate_2mm": float(np.mean((pe < tight_pos_tol) & (re < tight_rot_tol)) * 100.0),
        # high-precision level (0.1 mm / 0.001 deg, as in DLS-based IK papers); needs refine=True
        "success_rate_0.1mm": float(np.mean((pe < PRECISE_POS_TOL) & (re < PRECISE_ROT_TOL)) * 100.0),
        "mean_pos_err_mm": float(np.mean(pe_mm)),
        "median_pos_err_mm": float(np.median(pe_mm)),
        "p95_pos_err_mm": float(np.percentile(pe_mm, 95)),
        "max_pos_err_mm": float(np.max(pe_mm)),
        "mean_rot_err_deg": float(np.mean(re_deg)),
        "median_rot_err_deg": float(np.median(re_deg)),
        "p95_rot_err_deg": float(np.percentile(re_deg, 95)),
        "max_rot_err_deg": float(np.max(re_deg)),
        "mean_steps": float(np.mean([s.steps for s in solutions])),
        "mean_refine_iters": float(np.mean([s.refine_iters for s in solutions])),
    }


def evaluate_solver(
    solver,  # SACIKSolver / PolicyIKSolver (anything with solve_batch and cfg.term_* / eval_* tolerances)
    targets: EvalTargets,
    num_restarts: int = 1,
    max_steps: int = 120,
    refine: bool = False,
) -> tuple[dict[str, float], list[IKSolution]]:
    """Benchmark protocol: solve all targets (one batch) to the tight tolerance; returns metrics and solutions.

    refine=True adds DLS refinement towards 0.1 mm / 0.001 deg after the policy.
    """
    cfg = solver.cfg
    solutions = solver.solve_batch(
        targets.pos,
        target_rot=targets.rot,
        q_init=targets.q_init,
        max_steps=max_steps,
        pos_tol=cfg.term_pos_tol,
        rot_tol=cfg.term_rot_tol,
        num_restarts=num_restarts,
        refine=refine,
        refine_pos_tol=PRECISE_POS_TOL,
        refine_rot_tol=PRECISE_ROT_TOL,
    )
    metrics = solution_metrics(solutions, cfg.eval_pos_tol, cfg.eval_rot_tol, cfg.term_pos_tol, cfg.term_rot_tol)
    return metrics, solutions
