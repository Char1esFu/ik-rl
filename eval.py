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
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(find_latest_checkpoint(args.robot))
    out_dir = Path(args.out_dir) if args.out_dir else run_outputs_dir(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = SACIKSolver(args.checkpoint, device="cuda")
    env_cfg = solver.cfg
    env = IKEnv(solver.robot, cfg=env_cfg, render_mode="rgb_array", render_width=640, render_height=520)
    rng = np.random.default_rng(args.seed)

    pos_errs_mm: list[float] = []
    rot_errs_deg: list[float] = []
    solve_steps: list[int] = []
    solve_times_ms: list[float] = []
    succ_5mm: list[bool] = []
    succ_2mm: list[bool] = []

    gallery_samples: list[tuple[np.ndarray, float, float, int]] = []

    print(
        f"[Eval] Benchmarking {args.checkpoint} on {args.num_tests} random 6D poses "
        f"(restarts={args.restarts}, max_steps={args.max_steps})..."
    )

    # Warmup GPU inference
    _, p_warm, R_warm = env.sample_reachable_pose(rng)
    solver.solve(p_warm, target_rot=R_warm, max_steps=10)

    for idx in range(args.num_tests):
        _, target_pos, target_rot = env.sample_reachable_pose(rng)
        q_init = env.sample_initial_q(rng)

        t0 = time.perf_counter()
        sol = solver.solve(
            target_pos=target_pos,
            target_rot=target_rot,
            q_init=q_init,
            max_steps=args.max_steps,
            pos_tol=env_cfg.term_pos_tol,
            rot_tol=env_cfg.term_rot_tol,
            num_restarts=args.restarts,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        pe_mm = sol.pos_err_m * 1000.0
        re_deg = float(np.degrees(sol.rot_err_rad))

        pos_errs_mm.append(pe_mm)
        rot_errs_deg.append(re_deg)
        solve_steps.append(sol.steps)
        solve_times_ms.append(dt_ms)
        succ_5mm.append((sol.pos_err_m < 0.005) and (sol.rot_err_rad < 0.05))
        succ_2mm.append((sol.pos_err_m < 0.002) and (sol.rot_err_rad < 0.02))

        if idx < 6:
            try:
                env.reset(options={"target_pos": target_pos, "target_rot": target_rot, "q_init": sol.qpos})
                img = env.render(camera_azimuth=130.0 + (idx % 3 - 1) * 20.0, camera_elevation=-22.0)
                gallery_samples.append((img, pe_mm, re_deg, sol.steps))
            except Exception as render_exc:
                if idx == 0:
                    print(f"[Warn] Offscreen render skipped: {render_exc}")

    pos_arr = np.array(pos_errs_mm)
    rot_arr = np.array(rot_errs_deg)
    steps_arr = np.array(solve_steps)
    time_arr = np.array(solve_times_ms)

    summary = {
        "checkpoint": args.checkpoint,
        "robot": solver.robot.name,
        "num_tests": args.num_tests,
        "num_restarts": args.restarts,
        "success_rate_5mm_2.86deg_pct": round(float(np.mean(succ_5mm) * 100.0), 2),
        "success_rate_2mm_1.15deg_pct": round(float(np.mean(succ_2mm) * 100.0), 2),
        "pos_err_mm": {
            "mean": round(float(np.mean(pos_arr)), 3),
            "median": round(float(np.median(pos_arr)), 3),
            "p95": round(float(np.percentile(pos_arr, 95)), 3),
            "max": round(float(np.max(pos_arr)), 3),
        },
        "rot_err_deg": {
            "mean": round(float(np.mean(rot_arr)), 3),
            "median": round(float(np.median(rot_arr)), 3),
            "p95": round(float(np.percentile(rot_arr, 95)), 3),
            "max": round(float(np.max(rot_arr)), 3),
        },
        "mean_steps": round(float(np.mean(steps_arr)), 2),
        "mean_solve_time_ms": round(float(np.mean(time_arr)), 2),
    }

    print("\n================ SAC 6D IK Benchmark Summary ================")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(out_dir / f"eval_summary_r{args.restarts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render gallery figure
    if gallery_samples:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=140)
        fig.suptitle(
            f"{solver.robot.display_name} 6D IK Solved Poses (SAC Policy, restarts={args.restarts})\n"
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
        gallery_path = out_dir / f"ik_eval_gallery_r{args.restarts}.png"
        fig.savefig(gallery_path)
        plt.close(fig)
        print(f"[Eval] Saved rendered pose gallery to {gallery_path}")

    env.close()


if __name__ == "__main__":
    main()
