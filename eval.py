"""Evaluate trained 6D IK policies (any method; method and robot are read from each checkpoint).

  --protocol benchmark (default): this repository's benchmark (evaluation/benchmark.py). Random reachable targets
      with random starts, the policy's solver stops at 2 mm / 0.02 rad (at most --max-steps), --restarts starts,
      optional --refine (DLS to 0.1 mm / 0.001 deg). Writes eval_summary_r<k>[_refine].json and a rendered
      gallery to the run's outputs/ (or --out-dir; <out-dir>/<method>_<robot>/ for several checkpoints).
  --protocol hold: PolicyIK's hold protocol (evaluation/hold.py). 160 steps from each start; strict success =
      mean errors over the final 30 steps < 1 mm and < 0.2 deg (coarse: 1 cm / 2 deg); then the deployment
      protocol (error-corrected goal + up to 8 transient-goal attempts) and, with --refine, DLS refinement of the
      final configuration. Several checkpoints (and the --dls baseline) run side by side on identical tasks.

Checkpoints: --checkpoint PATH [NAME=PATH ...]; without it, the newest runs/<method>/<robot>/*/checkpoints/
best_model.pt (benchmark: one checkpoint, any method unless --method; hold: the newest of every method).

  .venv/bin/python eval.py --method policyik --robot franka --restarts 4 --refine
  .venv/bin/python eval.py --protocol hold --robot franka --dls --refine --out runs/policyik/franka/<run>/outputs/hold_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

# Default to EGL headless GPU rendering when DISPLAY is unavailable or for fast offscreen rendering
os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from core.batch_kinematics import BatchKinematics
from core.kinematics import RobotKinematics
from core.robot import available_robots, get_robot
from evaluation.baselines import DLSController
from evaluation.benchmark import EvalTargets, evaluate_solver
from evaluation.hold import HoldTasks, evaluate_hold, refine_final
from policy import METHODS, checkpoint_method, load_solver, make_controller
from policy.policyik.solver import PolicyIKSolver
from utils.render import PoseRenderer
from utils.runs import find_latest_checkpoint, run_outputs_dir

DEFAULTS = {"benchmark": {"seed": 2026, "num_tests": 500}, "hold": {"seed": 7, "num_tests": 2000}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=list(DEFAULTS), default="benchmark")
    ap.add_argument("--robot", default=None, help=f"{', '.join(available_robots())}: selects the newest runs")
    ap.add_argument("--method", default=None, choices=list(METHODS), help="selects the newest runs (default: any / all)")
    ap.add_argument("--checkpoint", nargs="+", default=None, metavar="[NAME=]PATH",
                    help="checkpoint(s) to evaluate (default: newest run, see above)")
    ap.add_argument("--num-tests", type=int, default=None, help="targets / tasks (default: benchmark 500, hold 2000)")
    ap.add_argument("--seed", type=int, default=None, help="target seed (default: benchmark 2026, hold 7)")
    ap.add_argument("--refine", action="store_true",
                    help="DLS refinement (benchmark: after the policy; hold: of the single attempt's final configuration)")
    b = ap.add_argument_group("benchmark protocol")
    b.add_argument("--restarts", type=int, default=1, help="start configurations per solve (1 = single rollout)")
    b.add_argument("--max-steps", type=int, default=120, help="max policy steps per solve")
    b.add_argument("--timing-tests", type=int, default=100, help="targets timed with single solve() calls")
    b.add_argument("--out-dir", default=None, help="default: <run>/outputs of the checkpoint")
    h = ap.add_argument_group("hold protocol")
    h.add_argument("--steps", type=int, default=160, help="steps per attempt")
    h.add_argument("--no-deploy", action="store_true", help="single attempt only")
    h.add_argument("--dls", action="store_true", help="add the damped-least-squares baseline controller")
    h.add_argument("--dls-damping", type=float, default=0.05)
    h.add_argument("--dls-max-step", type=float, default=0.0625, help="rad per step (PolicyIK's action scale)")
    h.add_argument("--backend", default=None, choices=["warp", "cpu"], help="batched kinematics (default: warp on CUDA)")
    h.add_argument("--workers", type=int, default=8, help="kinematics worker processes (cpu backend)")
    h.add_argument("--out", default=None, help="JSON output path")
    return ap


def parse_checkpoints(items: list[str] | None) -> list[tuple[str | None, str]]:
    out = []
    for item in items or []:
        name, sep, path = item.partition("=")
        out.append((name, path) if sep else (None, item))
    return out


# ---------------------------------------------------------------------------------------------------- benchmark


def run_benchmark(args: argparse.Namespace, checkpoint: str, subdir: bool = False) -> None:
    """subdir: write to <--out-dir>/<method>_<robot>/ (several checkpoints sharing one --out-dir)."""
    solver = load_solver(checkpoint, device="cuda")
    out_dir = Path(args.out_dir) if args.out_dir else run_outputs_dir(checkpoint)
    if args.out_dir and subdir:
        meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
        out_dir = out_dir / f"{checkpoint_method(meta)}_{solver.robot.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    method = "PolicyIK" if isinstance(solver, PolicyIKSolver) else "SAC+DLS"
    env_cfg = solver.cfg  # term_* / eval_* tolerances of the benchmark
    kin = RobotKinematics(solver.robot)
    rng = np.random.default_rng(args.seed)

    print(
        f"[Eval] Benchmarking {checkpoint} on {args.num_tests} random 6D poses "
        f"(restarts={args.restarts}, max_steps={args.max_steps})..."
    )

    # Warmup GPU inference (also keeps the target set identical to earlier versions of this script)
    _, p_warm, R_warm = kin.sample_target(mujoco.MjData(kin.model), rng)
    solver.solve(p_warm, target_rot=R_warm, max_steps=10)

    targets = EvalTargets.sample(kin, args.num_tests, rng)
    suffix = "_refine" if args.refine else ""
    metrics, solutions = evaluate_solver(
        solver, targets, num_restarts=args.restarts, max_steps=args.max_steps, refine=args.refine
    )

    # Latency of single solve() calls (the batched evaluation above solves all targets at once)
    solve_times_ms: list[float] = []
    for i in range(min(args.timing_tests, len(targets))):
        t0 = time.perf_counter()
        solver.solve(
            targets.pos[i], target_rot=targets.rot[i], q_init=targets.q_init[i], max_steps=args.max_steps,
            pos_tol=env_cfg.term_pos_tol, rot_tol=env_cfg.term_rot_tol, num_restarts=args.restarts,
            refine=args.refine,
        )
        solve_times_ms.append((time.perf_counter() - t0) * 1000.0)

    gallery_samples: list[tuple[np.ndarray, float, float, int]] = []
    renderer = None
    for idx, sol in enumerate(solutions[:6]):
        try:
            if renderer is None:
                renderer = PoseRenderer(kin, width=640, height=520)
            img = renderer.render(sol.qpos, targets.pos[idx], targets.rot[idx],
                                  azimuth=130.0 + (idx % 3 - 1) * 20.0, elevation=-22.0)
            gallery_samples.append((img, sol.pos_err_m * 1000.0, float(np.degrees(sol.rot_err_rad)), sol.steps))
        except Exception as render_exc:
            print(f"[Warn] Offscreen render skipped: {render_exc}")
            break

    r = lambda x: round(float(x), 3)  # noqa: E731
    summary = {
        "checkpoint": checkpoint,
        "robot": solver.robot.name,
        "method": method,
        "num_tests": args.num_tests,
        "num_restarts": args.restarts,
        "max_steps": args.max_steps,
        "refine": args.refine,
        "success_rate_5mm_2.86deg_pct": round(metrics["success_rate_5mm"], 2),
        "success_rate_2mm_1.15deg_pct": round(metrics["success_rate_2mm"], 2),
        "success_rate_0.1mm_0.001deg_pct": round(metrics["success_rate_0.1mm"], 2),
        "pos_err_mm": {
            "mean": r(metrics["mean_pos_err_mm"]),
            "median": r(metrics["median_pos_err_mm"]),
            "p95": r(metrics["p95_pos_err_mm"]),
            "max": r(metrics["max_pos_err_mm"]),
        },
        "rot_err_deg": {
            "mean": r(metrics["mean_rot_err_deg"]),
            "median": r(metrics["median_rot_err_deg"]),
            "p95": r(metrics["p95_rot_err_deg"]),
            "max": r(metrics["max_rot_err_deg"]),
        },
        "mean_steps": round(metrics["mean_steps"], 2),
        "mean_refine_iters": round(metrics["mean_refine_iters"], 2),
        "mean_solve_time_ms": round(float(np.mean(solve_times_ms)), 2) if solve_times_ms else None,
        "solve_time_note": f"single solve() calls on the first {len(solve_times_ms)} targets",
    }

    print(f"\n================ {method} 6D IK Benchmark Summary ================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out_dir / f"eval_summary_r{args.restarts}{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if gallery_samples:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=140)
        fig.suptitle(
            f"{solver.robot.display_name} 6D IK Solved Poses ({method} Policy{' + DLS' if args.refine else ''}, restarts={args.restarts})\n"
            f"SR(<5mm, <2.86°): {summary['success_rate_5mm_2.86deg_pct']}% | "
            f"Median Error: {summary['pos_err_mm']['median']:.2f} mm, {summary['rot_err_deg']['median']:.2f}°",
            fontsize=13,
            fontweight="bold",
        )
        for ax, (img, pe, re, st) in zip(axes.flat, gallery_samples, strict=False):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"Δpos={pe:.2f} mm | Δrot={re:.2f}° ({st} steps)", fontsize=11)
        fig.tight_layout()
        gallery_path = out_dir / f"ik_eval_gallery_r{args.restarts}{suffix}.png"
        fig.savefig(gallery_path)
        plt.close(fig)
        print(f"[Eval] Saved rendered pose gallery to {gallery_path}")

    if renderer is not None:
        renderer.close()


# ---------------------------------------------------------------------------------------------------- hold


def run_hold(args: argparse.Namespace, pairs: list[tuple[str | None, str]]) -> None:
    if not pairs:
        if args.robot is None:
            raise SystemExit("--protocol hold needs --robot or --checkpoint")
        for method in [args.method] if args.method else METHODS:
            try:
                pairs.append((method, str(find_latest_checkpoint(args.robot, method=method))))
            except FileNotFoundError:
                pass
    named = []
    for name, path in pairs:
        meta = torch.load(path, map_location="cpu", weights_only=True)
        named.append((name or checkpoint_method(meta), path))
    solvers = [(name, path, load_solver(path)) for name, path in named]
    if args.robot:
        robot = get_robot(args.robot)
    elif solvers:
        robot = solvers[0][2].robot
    else:
        raise SystemExit("no checkpoint found; train first or pass --checkpoint")
    tasks = HoldTasks.sample(robot, args.num_tests, args.seed)
    bk = BatchKinematics(robot, backend=args.backend, num_workers=args.workers, max_batch=max(4096, args.num_tests))

    controllers = []
    for name, path, solver in solvers:
        if solver.robot.name != robot.name:
            raise SystemExit(f"{path} is a {solver.robot.name} checkpoint, not {robot.name}")
        controllers.append((name, path, make_controller(solver, bk)))
    if args.dls:
        controllers.append(("dls", f"damping {args.dls_damping}, max step {args.dls_max_step} rad",
                            DLSController(bk, args.dls_damping, args.dls_max_step)))

    results = {"robot": robot.name, "num_tasks": args.num_tests, "seed": args.seed, "steps": args.steps,
               "methods": {}}
    for name, path, ctrl in controllers:
        t0 = time.time()
        m, log = evaluate_hold(ctrl, tasks, steps=args.steps, deploy=not args.no_deploy)
        if args.refine:
            m["single"].update(refine_final(bk.kin, log.q_final, tasks))
        m["source"] = path
        m["eval_time_s"] = round(time.time() - t0, 1)
        results["methods"][name] = m
        s = m["single"]
        line = (f"{name:10s} strict {s['strict']:6.2f}%  coarse {s['coarse']:6.2f}%  containment "
                f"{s['containment']:6.2f}%  pos med {s['pos_median_mm']:.3f} mm  ori med {s['ori_median_deg']:.3f} deg")
        if "deployment" in m:
            line += f"  | deploy strict {m['deployment']['strict']:.2f}% E[N] {m['deployment']['EN']:.3f}"
        if args.refine:
            line += f"  | +DLS refine strict {s['refined_strict']:.2f}% 0.1mm {s['refined_0.1mm']:.2f}%"
        print(line, flush=True)
    bk.close()

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, float):
            return round(o, 4)
        return o

    print("\n" + json.dumps(clean(results), indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(clean(results), indent=1))


def main() -> None:
    args = build_parser().parse_args()
    for key, value in DEFAULTS[args.protocol].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    pairs = parse_checkpoints(args.checkpoint)
    if args.protocol == "hold":
        run_hold(args, pairs)
        return
    paths = [p for _, p in pairs] or [str(find_latest_checkpoint(args.robot, method=args.method))]
    for path in paths:
        run_benchmark(args, path, subdir=len(paths) > 1)


if __name__ == "__main__":
    main()
