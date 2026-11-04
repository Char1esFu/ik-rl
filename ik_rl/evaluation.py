"""Evaluation of trained IK policies, shared by train.py, eval.py and the comparison scripts.

- evaluate_solver: the benchmark protocol. Fixed targets with start configurations, solved with
  SACIKSolver (stop once within 2 mm / 0.02 rad, at most `max_steps` steps, `num_restarts` branches).
  train.py uses it with one start for the training curves, best-model selection and early stopping,
  eval.py with 1 and 4 starts, so all reported numbers are comparable.
- evaluate_parked: fixed-horizon rollouts in the training environment (max_episode_steps, final error,
  no early exit): checks that the policy also holds still at the target. Logged as an extra metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ik_rl.env import IKEnv, VectorIKEnv

if TYPE_CHECKING:
    from ik_rl.sac import SACAgent
    from ik_rl.solver import IKSolution, SACIKSolver


@dataclass
class EvalTargets:
    """Fixed evaluation problems: reachable target poses and start configurations."""

    pos: np.ndarray  # (N, 3)
    rot: np.ndarray  # (N, 3, 3)
    q_init: np.ndarray  # (N, n)

    @classmethod
    def sample(cls, env: IKEnv, num: int, rng: np.random.Generator) -> EvalTargets:
        """Per problem: a reachable target (sample_reachable_pose), then a start (sample_initial_q)."""
        pos, rot, q_init = [], [], []
        for _ in range(num):
            _, p, R = env.sample_reachable_pose(rng)
            pos.append(p)
            rot.append(R)
            q_init.append(env.sample_initial_q(rng))
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
        "mean_pos_err_mm": float(np.mean(pe_mm)),
        "median_pos_err_mm": float(np.median(pe_mm)),
        "p95_pos_err_mm": float(np.percentile(pe_mm, 95)),
        "max_pos_err_mm": float(np.max(pe_mm)),
        "mean_rot_err_deg": float(np.mean(re_deg)),
        "median_rot_err_deg": float(np.median(re_deg)),
        "p95_rot_err_deg": float(np.percentile(re_deg, 95)),
        "max_rot_err_deg": float(np.max(re_deg)),
        "mean_steps": float(np.mean([s.steps for s in solutions])),
    }


def evaluate_solver(
    solver: SACIKSolver,
    targets: EvalTargets,
    num_restarts: int = 1,
    max_steps: int = 120,
) -> tuple[dict[str, float], list[IKSolution]]:
    """Benchmark protocol: solve all targets (one batch) to the tight tolerance; returns metrics and solutions."""
    cfg = solver.cfg
    solutions = solver.solve_batch(
        targets.pos,
        target_rot=targets.rot,
        q_init=targets.q_init,
        max_steps=max_steps,
        pos_tol=cfg.term_pos_tol,
        rot_tol=cfg.term_rot_tol,
        num_restarts=num_restarts,
    )
    metrics = solution_metrics(solutions, cfg.eval_pos_tol, cfg.eval_rot_tol, cfg.term_pos_tol, cfg.term_rot_tol)
    return metrics, solutions


def evaluate_parked(agent: SACAgent, eval_env: VectorIKEnv, num_episodes: int = 128) -> dict[str, float]:
    """Fixed-horizon deterministic rollouts on fresh random targets; success / error at the final step."""
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    horizon = eval_env.cfg.max_episode_steps
    successes_5mm: list[bool] = []
    successes_2mm: list[bool] = []
    pos_errs: list[float] = []
    rot_errs: list[float] = []

    episodes_target = max(num_episodes, n_envs)
    while len(pos_errs) < episodes_target:
        actions = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(actions)
        for i in np.flatnonzero(terminated | truncated):
            pos_errs.append(float(info["pos_err"][i]) * 1000.0)
            rot_errs.append(float(np.degrees(info["rot_err"][i])))
            successes_5mm.append(bool(info["is_success"][i]))
            successes_2mm.append(bool(info["is_tight_success"][i]))
            if len(pos_errs) >= episodes_target:
                break

    return {
        "success_rate_5mm": float(np.mean(successes_5mm) * 100.0),
        "success_rate_2mm": float(np.mean(successes_2mm) * 100.0),
        "median_pos_err_mm": float(np.median(pos_errs)),
        "median_rot_err_deg": float(np.median(rot_errs)),
        "horizon": float(horizon),
    }
