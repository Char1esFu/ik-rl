"""Evaluate trained SAC 6D IK Solver on unseen poses & render MuJoCo gallery."""

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
import numpy as np

from ik_rl.env import IKEnv
from ik_rl.evaluation import EvalTargets, evaluate_solver
from ik_rl.robot import available_robots
from ik_rl.runs import find_latest_checkpoint, run_outputs_dir
from ik_rl.solver import SACIKSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC 6D IK solver")
    parser.add_argument(
        "--robot", type=str, default=None,
        help=f"Robot ({', '.join(available_robots())}) whose newest checkpoint to use if --checkpoint is not given (default: any)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Default: newest runs/<robot>/*/checkpoints/best_model.pt (robot is read from the checkpoint)",
    )
    parser.add_argument("--num-tests", type=int, default=500, help="Number of random reachable 6D targets")
    parser.add_argument("--restarts", type=int, default=1, help="Number of initial seeds per solve (1=single rollout)")
    parser.add_argument("--max-steps", type=int, default=120, help="Max kinematic steps per solve")
    parser.add_argument("--out-dir", type=str, default=None, help="Default: <run>/outputs of the checkpoint")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--refine", action="store_true", help="DLS refinement after the policy (towards 0.1 mm / 0.001 deg)"
    )
    parser.add_argument("--timing-tests", type=int, default=100, help="Targets timed with single solve() calls")
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(find_latest_checkpoint(args.robot))
    out_dir = Path(args.out_dir) if args.out_dir else run_outputs_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = SACIKSolver(args.checkpoint, device="cuda")
    env_cfg = solver.cfg
    env = IKEnv(solver.robot, cfg=env_cfg, render_mode="rgb_array", render_width=640, render_height=520)
    rng = np.random.default_rng(args.seed)

    print(
        f"[Eval] Benchmarking {args.checkpoint} on {args.num_tests} random 6D poses "
        f"(restarts={args.restarts}, max_steps={args.max_steps})..."
    )

    # Warmup GPU inference (also keeps the target set identical to earlier versions of this script)
    _, p_warm, R_warm = env.sample_reachable_pose(rng)
    solver.solve(p_warm, target_rot=R_warm, max_steps=10)

    targets = EvalTargets.sample(env, args.num_tests, rng)
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
    for idx, sol in enumerate(solutions[:6]):
        try:
            env.reset(options={"target_pos": targets.pos[idx], "target_rot": targets.rot[idx], "q_init": sol.qpos})
            img = env.render(camera_azimuth=130.0 + (idx % 3 - 1) * 20.0, camera_elevation=-22.0)
            gallery_samples.append((img, sol.pos_err_m * 1000.0, float(np.degrees(sol.rot_err_rad)), sol.steps))
        except Exception as render_exc:
            print(f"[Warn] Offscreen render skipped: {render_exc}")
            break

    r = lambda x: round(float(x), 3)  # noqa: E731
    summary = {
        "checkpoint": args.checkpoint,
        "robot": solver.robot.name,
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

    print("\n================ SAC 6D IK Benchmark Summary ================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out_dir / f"eval_summary_r{args.restarts}{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render gallery figure
    if gallery_samples:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=140)
        fig.suptitle(
            f"{solver.robot.display_name} 6D IK Solved Poses (SAC Policy{' + DLS' if args.refine else ''}, restarts={args.restarts})\n"
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

    env.close()


if __name__ == "__main__":
    main()
