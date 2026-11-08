"""PolicyIK's hold protocol for any controller (port of simulation/policyik_sim/{evaluate,metrics,
corrected_goals}.py of github.com/Tinaliu0123/PolicyIK to this repository's kinematic setting).

Single attempt: a fixed horizon of `steps` (160 = 5.3 s at 30 Hz) from the task's start configuration; a task
succeeds when the MEAN position / orientation errors over the final `tail` steps (30 = 1 s) are below
1 mm and 0.2 deg (strict) or 1 cm and 2 deg (coarse). The controller must reach the goal and hold it.

Deployment protocol (cumulative, every attempt judged against the original goal):
  1. single attempt;
  2. if it failed but ended within 1 cm: one attempt towards an error-corrected goal that cancels the
     final residual (goal shifted by -e_p and rotated by the inverse final rotation error);
  3. otherwise / still failing: up to 8 attempts that first command a transient goal (0.30 m along a random
     direction, rotated 90 deg about a random axis) for 40 steps, then the original goal for 120 steps.
E[N] = mean number of executed attempts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.controller import Controller
from evaluation.benchmark import EvalTargets
from evaluation.rollout import HorizonLog, run_horizon
from utils.math_utils import so3_error_vec

STRICT = (0.001, np.radians(0.2))
COARSE = (0.01, np.radians(2.0))
GATE = 0.01  # error-corrected goal only for failures ending within 1 cm
TAIL = 30
HZ = 30.0
N_TRANSIENT = 8
TRANSIENT_STEPS, TRANSIENT_OFFSET_M, TRANSIENT_ROT_DEG = 40, 0.30, 90.0


@dataclass
class HoldTasks:
    q_start: np.ndarray  # (N, n)
    p_goal: np.ndarray  # (N, 3)
    R_goal: np.ndarray  # (N, 3, 3)

    def __len__(self) -> int:
        return len(self.q_start)

    @classmethod
    def sample(cls, robot, num: int, seed: int) -> HoldTasks:
        """Fixed tasks drawn like eval.py / compare_models.py targets (EvalTargets with the given seed)."""
        t = EvalTargets.sample(robot, num, np.random.default_rng(seed))
        return cls(t.q_init.astype(np.float32), t.pos.astype(np.float32), t.rot.astype(np.float32))


def _rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(th, 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def tail_means(log: HorizonLog) -> tuple[np.ndarray, np.ndarray]:
    return log.pos_t[:, -TAIL:].mean(1), log.rot_t[:, -TAIL:].mean(1)


def single_attempt_metrics(log: HorizonLog) -> dict[str, float]:
    p, o = tail_means(log)
    strict = (p < STRICT[0]) & (o < STRICT[1])
    inst = (log.pos_t < STRICT[0]) & (log.rot_t < STRICT[1])
    entered, contained = inst.any(1), inst[:, -TAIL:].all(1)
    first = np.argmax(inst, 1)
    drift_p = np.linalg.norm(log.p_hist[:, -1] - log.p_hist[:, -TAIL - 1], axis=1)
    _, drift_o = so3_error_vec(log.R_hist[:, -1], log.R_hist[:, -TAIL - 1], out_dtype=np.float64)
    med = lambda x: float(np.median(x)) if len(x) else float("nan")  # noqa: E731
    return {
        "strict": 100 * float(strict.mean()),
        "coarse": 100 * float(((p < COARSE[0]) & (o < COARSE[1])).mean()),
        "entry": 100 * float(entered.mean()),
        "containment": 100 * float(contained.mean()),
        "uncontained_after_entry": 100 * float((entered & ~contained).sum() / max(entered.sum(), 1)),
        "entry_time_s": med((first[entered] + 1) / HZ),
        "drift_um": 1e6 * med(drift_p[strict]),
        "drift_deg": float(np.degrees(med(drift_o[strict]))),
        "pos_mean_mm": 1e3 * float(p.mean()),
        "pos_median_mm": 1e3 * float(np.median(p)),
        "ori_mean_deg": float(np.degrees(o.mean())),
        "ori_median_deg": float(np.degrees(np.median(o))),
    }


def corrected_goals(log: HorizonLog, tasks: HoldTasks) -> tuple[np.ndarray, np.ndarray]:
    """Goal shifted by the negative final position error and rotated by the inverse final rotation error."""
    p_fin, R_fin = log.p_hist[:, -1].astype(np.float64), log.R_hist[:, -1].astype(np.float64)
    p_c = tasks.p_goal - (p_fin - tasks.p_goal)
    Rg = tasks.R_goal.astype(np.float64)
    v, _ = so3_error_vec(Rg, R_fin, out_dtype=np.float64)  # world-frame: R_fin = Exp(v) R_goal
    R_c = _rotvec_to_mat(-v) @ Rg
    return p_c.astype(np.float32), R_c.astype(np.float32)


def transient_goals(tasks: HoldTasks, seed: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    N = len(tasks)
    dirs = g.normal(size=(N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    axes = g.normal(size=(N, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    p = tasks.p_goal + TRANSIENT_OFFSET_M * dirs
    R = _rotvec_to_mat(np.radians(TRANSIENT_ROT_DEG) * axes) @ tasks.R_goal.astype(np.float64)
    return p.astype(np.float32), R.astype(np.float32)


def _run(ctrl: Controller, tasks: HoldTasks, rows: np.ndarray, steps: int, goal=None, transient=None,
         chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Tail-mean errors (vs the original goal) for a subset of tasks; goal / transient override the command."""
    pe = np.full(len(tasks), np.inf)
    oe = np.full(len(tasks), np.inf)
    for s in range(0, len(rows), chunk):
        r = rows[s : s + chunk]
        if goal is not None:  # command a different goal, measure against the original one
            log = run_horizon(ctrl, tasks.q_start[r], tasks.p_goal[r], tasks.R_goal[r], steps,
                              transient=(goal[0][r], goal[1][r], steps))
        elif transient is not None:
            log = run_horizon(ctrl, tasks.q_start[r], tasks.p_goal[r], tasks.R_goal[r], steps,
                              transient=(transient[0][r], transient[1][r], TRANSIENT_STEPS))
        else:
            log = run_horizon(ctrl, tasks.q_start[r], tasks.p_goal[r], tasks.R_goal[r], steps)
        pe[r], oe[r] = tail_means(log)
    return pe, oe


def evaluate_hold(ctrl: Controller, tasks: HoldTasks, steps: int = 160, deploy: bool = True,
                  chunk: int = 4096) -> tuple[dict, HorizonLog]:
    """Single-attempt metrics and (optionally) the deployment protocol; returns (metrics, single-attempt log)."""
    N = len(tasks)
    logs = []
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        logs.append(run_horizon(ctrl, tasks.q_start[sl], tasks.p_goal[sl], tasks.R_goal[sl], steps))
    log = HorizonLog(*(np.concatenate([getattr(lg, f) for lg in logs]) for f in HorizonLog.__dataclass_fields__))
    out: dict = {"single": single_attempt_metrics(log)}
    if not deploy:
        return out, log

    def ok(pe, oe, tol):
        return (pe < tol[0]) & (oe < tol[1])

    pb, ob = tail_means(log)
    covered = ok(pb, ob, STRICT)
    coarse_any = ok(pb, ob, COARSE)
    attempts = np.ones(N)
    # delivered attempt = first strict success, else the last executed attempt
    pd, od = pb.copy(), ob.copy()
    steps_rows = [("single attempt", 100 * covered.mean(), 100 * coarse_any.mean(), attempts.mean())]

    call = np.flatnonzero(~covered & (pb <= GATE))
    if len(call):
        p_c, R_c = corrected_goals(log, tasks)
        pe, oe = _run(ctrl, tasks, call, steps, goal=(p_c, R_c), chunk=chunk)
        attempts[call] += 1
        s = ok(pe, oe, STRICT)
        coarse_any[call] |= ok(pe[call], oe[call], COARSE)
        pd[call], od[call] = pe[call], oe[call]
        covered |= s
    out["corrected_calls"] = int(len(call))
    out["corrected_rescued"] = int(covered[call].sum()) if len(call) else 0
    steps_rows.append(("+ error-corrected goal", 100 * covered.mean(), 100 * coarse_any.mean(), attempts.mean()))
    for k in range(N_TRANSIENT):
        rows = np.flatnonzero(~covered)
        if len(rows):
            tg = transient_goals(tasks, seed=k)
            pe, oe = _run(ctrl, tasks, rows, steps, transient=tg, chunk=chunk)  # 40 transient + 120 original
            attempts[rows] += 1
            covered |= ok(pe, oe, STRICT)
            coarse_any[rows] |= ok(pe[rows], oe[rows], COARSE)
            pd[rows], od[rows] = pe[rows], oe[rows]
        steps_rows.append((f"+ transient goal {k}", 100 * covered.mean(), 100 * coarse_any.mean(), attempts.mean()))
    out["deployment"] = {
        "steps": [{"label": a, "strict": round(b, 2), "coarse": round(c, 2), "EN": round(d, 3)}
                  for a, b, c, d in steps_rows],
        "strict": 100 * float(covered.mean()),
        "coarse_delivered": 100 * float(ok(pd, od, COARSE).mean()),
        "pos_median_mm": 1e3 * float(np.median(pd)),
        "ori_median_deg": float(np.degrees(np.median(od))),
        "EN": float(attempts.mean()),
    }
    return out, log


def refine_final(kin, q_final: np.ndarray, tasks: HoldTasks) -> dict[str, float]:
    """DLS refinement (core.refine.dls_refine) of each final configuration: success at strict / 0.1 mm."""
    import mujoco

    from core.refine import dls_refine
    from evaluation.benchmark import PRECISE_POS_TOL, PRECISE_ROT_TOL

    data = mujoco.MjData(kin.model)
    pe, oe = np.zeros(len(tasks)), np.zeros(len(tasks))
    for i in range(len(tasks)):
        r = dls_refine(kin, data, q_final[i], tasks.p_goal[i], tasks.R_goal[i])
        if r is None:
            pe[i] = oe[i] = np.inf
        else:
            pe[i], oe[i] = r.pos_err, r.rot_err
    return {
        "refined_strict": 100 * float(((pe < STRICT[0]) & (oe < STRICT[1])).mean()),
        "refined_0.1mm": 100 * float(((pe < PRECISE_POS_TOL) & (oe < PRECISE_ROT_TOL)).mean()),
    }
