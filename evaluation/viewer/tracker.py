"""Real-time multi-start IK tracking of a moving / changing goal for the interactive viewer (any method).

The visible arm is controller row 0 of a 1-row controller (`main`); `num_restarts - 1` shadow branches run in a
second controller from the home pose and random collision-free seeds. If the visible arm gets trapped (joint
stops, self-collision, a far IK branch), it slews in joint space to the best shadow configuration and fine-parks.
Controllers come from a factory (policy.make_controller), so this works for every method.
"""

from __future__ import annotations

from typing import Any, Callable

import mujoco
import numpy as np

from core.controller import Controller
from core.kinematics import RobotKinematics

SHADOW_STEPS = 65  # shadows run for the first steps of a target only
SLEW_RATE = 0.16  # rad per frame (joint-space rescue)


class MultiStartTracker:
    def __init__(
        self,
        make_controller: Callable[[], Controller],
        kin: RobotKinematics,
        q0: np.ndarray,
        p_goal: np.ndarray,
        R_goal: np.ndarray,
        num_restarts: int = 4,
        seed: int = 42,
    ) -> None:
        self.kin = kin
        self.k_shadow = max(0, int(num_restarts) - 1)
        self.rng = np.random.default_rng(seed)
        self._data = mujoco.MjData(kin.model)
        self.main = make_controller()
        self.shadow = make_controller() if self.k_shadow > 0 else None
        self.cfg = self.main.cfg  # eval_* / term_* tolerances
        self.horizon = self.main.horizon
        self.max_steps = self.horizon + 50 if self.k_shadow > 0 else self.horizon
        self.p_goal = np.asarray(p_goal, dtype=np.float32).copy()
        self.R_goal = np.asarray(R_goal, dtype=np.float32).copy()
        self.main.reset(np.asarray(q0, dtype=np.float32)[None], self.p_goal[None], self.R_goal[None])
        self.on_new_goal()

    # ---------------------------------------------------------------- state of the visible arm
    @property
    def q(self) -> np.ndarray:
        return self.main.q[0]

    @property
    def p(self) -> np.ndarray:
        return self.main.p[0]

    @property
    def R(self) -> np.ndarray:
        return self.main.R[0]

    def errors(self) -> tuple[float, float]:
        e_p, e_r = self.main.errors()
        return float(e_p[0]), float(e_r[0])

    def info(self) -> dict[str, Any]:
        d_pos, d_rot = self.errors()
        c = self.cfg
        return {
            "pos_err": d_pos,
            "rot_err": d_rot,
            "is_success": (d_pos < c.eval_pos_tol) and (d_rot < c.eval_rot_tol),
            "is_tight_success": (d_pos < c.term_pos_tol) and (d_rot < c.term_rot_tol),
        }

    def settled(self) -> bool:
        """Within the tight tolerance and (almost) no longer moving."""
        return bool(self.info()["is_tight_success"]) and float(self.main.activity()[0]) < 0.08

    # ---------------------------------------------------------------- goals
    def set_goal(self, p_goal: np.ndarray, R_goal: np.ndarray, new_goal: bool = True) -> None:
        """new_goal: a new target (controller memory and shadows restart); False: a moving goal (figure-8)."""
        self.p_goal = np.asarray(p_goal, dtype=np.float32).copy()
        self.R_goal = np.asarray(R_goal, dtype=np.float32).copy()
        if new_goal:
            self.main.reset(self.q[None].copy(), self.p_goal[None], self.R_goal[None])
            self.on_new_goal()
        else:
            self.main.set_goal(np.array([0]), self.p_goal[None], self.R_goal[None])

    def reset_arm(self, q: np.ndarray, p_goal: np.ndarray, R_goal: np.ndarray) -> None:
        self.p_goal = np.asarray(p_goal, dtype=np.float32).copy()
        self.R_goal = np.asarray(R_goal, dtype=np.float32).copy()
        self.main.reset(np.asarray(q, dtype=np.float32)[None], self.p_goal[None], self.R_goal[None])
        self.on_new_goal()

    def on_new_goal(self) -> None:
        self.step_in_ep = 0
        self.goal_counted = False
        self.parked = False
        self.slew_target_q: np.ndarray | None = None
        self.via_home = False
        d_pos, d_rot = self.errors()
        self.main_best_score = d_pos + 0.2 * d_rot
        self.main_stagnant = 0
        self.global_best_score = self.main_best_score
        self.global_stagnant = 0
        self.best_shadow_q: np.ndarray | None = None
        self.best_shadow_score = float("inf")
        self.best_shadow_solved = False
        self.best_shadow_tight = False
        if self.shadow is not None:
            seeds = np.stack([self._shadow_seed(i) for i in range(self.k_shadow)])
            self.shadow.reset(seeds, np.repeat(self.p_goal[None], self.k_shadow, 0),
                              np.repeat(self.R_goal[None], self.k_shadow, 0))
            self._record_shadow_progress()

    def _shadow_seed(self, idx: int) -> np.ndarray:
        if idx == 0:
            return self.kin.home_q.copy()
        for _ in range(80):
            cand = self.kin.sample_box(self.rng)
            p_c, _, _, col = self.kin.fk(self._data, cand)
            if (not col) and self.kin.valid_start(p_c):
                return cand
        return self.kin.home_q.copy()

    def _record_shadow_progress(self) -> None:
        e_p, e_r = self.shadow.errors()
        scores = e_p + 0.2 * e_r
        idx = int(np.argmin(scores))
        if float(scores[idx]) < self.best_shadow_score:
            self.best_shadow_score = float(scores[idx])
            self.best_shadow_q = self.shadow.q[idx].copy()
            c = self.cfg
            if e_p[idx] < c.eval_pos_tol and e_r[idx] < c.eval_rot_tol:
                self.best_shadow_solved = True
            if e_p[idx] < c.term_pos_tol and e_r[idx] < c.term_rot_tol:
                self.best_shadow_tight = True

    # ---------------------------------------------------------------- stepping
    def _step_slew(self) -> None:
        """Move the visible arm towards the rescue configuration (via home if the direct path collides)."""
        q = self.q
        waypoint = self.kin.home_q if self.via_home else self.slew_target_q

        def candidate(target: np.ndarray) -> tuple[np.ndarray, bool]:
            diff = target - q
            dq = diff * min(1.0, SLEW_RATE / max(float(np.max(np.abs(diff))), 1e-6))
            q_c = np.clip(q + dq, self.kin.q_min, self.kin.q_max)
            return q_c, self.kin.fk(self._data, q_c)[3]

        q_c, col = candidate(waypoint)
        if col and not self.via_home:
            self.via_home = True
            q_c, col = candidate(self.kin.home_q)
        if not col:
            self.main.reset(q_c[None].astype(np.float32), self.p_goal[None], self.R_goal[None])
        q = self.q
        if self.via_home and float(np.max(np.abs(self.kin.home_q - q))) < 0.08:
            self.via_home = False
        elif (not self.via_home) and float(np.max(np.abs(self.slew_target_q - q))) < 0.03:
            self.slew_target_q = None
            self.main_stagnant = 0
        self.step_in_ep += 1

    def step_direct(self, smooth: float = 1.0) -> dict[str, Any]:
        """One policy step of the visible arm only (moving goals: no shadows / rescue)."""
        self.main.step(smooth=smooth)
        self.step_in_ep += 1
        return self.info()

    def step_frame(self, smooth_alpha: float = 1.0, hold_when_tight: bool = False) -> tuple[dict[str, Any], bool]:
        """Advance one control frame. Returns (info, newly_solved)."""
        info0 = self.info()
        already_tight = info0["is_tight_success"]

        if hold_when_tight and (self.parked or (already_tight and float(self.main.activity()[0]) < 0.06)):
            self.parked = True
            newly_solved = already_tight and not self.goal_counted
            self.goal_counted |= newly_solved
            return info0, newly_solved

        if self.slew_target_q is not None:
            self._step_slew()
            info = self.info()
            newly_solved = info["is_tight_success"] and not self.goal_counted
            self.goal_counted |= newly_solved
            return info, newly_solved

        if self.shadow is not None and self.step_in_ep < SHADOW_STEPS and not already_tight:
            self.shadow.step()
            self._record_shadow_progress()

        # Damp the action to zero only if both the visible arm and all shadow branches have stagnated
        # (e.g. a manual slider dragged beyond the arm's reach), never on slow valid motion.
        damp = 1.0
        if hold_when_tight and self.step_in_ep > 45 and self.global_stagnant > 18:
            damp = float(np.clip(1.0 - (self.global_stagnant - 18) / 15.0, 0.0, 1.0))
            if damp <= 0.0:
                self.parked = True
        self.main.step(smooth=smooth_alpha, scale=damp)
        self.step_in_ep += 1

        info = self.info()
        main_score = info["pos_err"] + 0.2 * info["rot_err"]
        if main_score < self.main_best_score - 2e-4:
            self.main_best_score, self.main_stagnant = main_score, 0
        else:
            self.main_stagnant += 1
        global_best_now = min(main_score, self.best_shadow_score)
        if global_best_now < self.global_best_score - 2e-4:
            self.global_best_score, self.global_stagnant = global_best_now, 0
        else:
            self.global_stagnant += 1

        main_solved, main_tight = info["is_success"], info["is_tight_success"]
        if self.shadow is not None and not main_tight and self.best_shadow_q is not None:
            rescue = (
                ((not main_solved) and self.best_shadow_solved and (self.step_in_ep >= 28 or self.main_stagnant >= 10))
                or (self.best_shadow_tight and self.step_in_ep >= 38 and self.main_stagnant >= 10)
                or (
                    (not main_solved)
                    and self.step_in_ep >= 38
                    and self.main_stagnant >= 12
                    and self.best_shadow_score < 0.55 * main_score
                    and main_score - self.best_shadow_score > 0.015
                )
            )
            if rescue:
                self.slew_target_q = self.best_shadow_q.copy()
                self.via_home = False

        settled_tight = main_tight and float(self.main.activity()[0]) < 0.08
        newly_solved = settled_tight and not self.goal_counted
        self.goal_counted |= newly_solved
        return info, newly_solved

    def episode_done(self) -> bool:
        """Auto mode: the current target is finished (settled, or the step budget is used up)."""
        return (
            self.settled()
            or (self.slew_target_q is None and self.step_in_ep >= self.horizon)
            or self.step_in_ep >= self.max_steps
        )
