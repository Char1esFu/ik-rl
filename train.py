"""Train Soft Actor-Critic (SAC) for 6D inverse kinematics of a registered robot."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ik_rl.config import IKConfig, SACConfig, coerce_value, load_experiment, merge_overrides, parse_set_args
from ik_rl.env import VectorIKEnv
from ik_rl.obs import OBS_FEATURES
from ik_rl.replay_buffer import GPUReplayBuffer
from ik_rl.robot import DEFAULT_ROBOT, RobotSpec, available_robots, get_robot
from ik_rl.sac import SACAgent


def evaluate_policy(
    agent: SACAgent,
    eval_env: VectorIKEnv,
    num_episodes: int = 128,
) -> dict[str, float]:
    """Run deterministic policy rollouts on fresh random 6D targets and measure final parked accuracy."""
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    first_solved_step = np.full(n_envs, eval_env.cfg.max_episode_steps, dtype=np.int32)
    cur_step = np.zeros(n_envs, dtype=np.int32)

    pos_errs: list[float] = []
    rot_errs_deg: list[float] = []
    successes_5mm: list[bool] = []
    successes_2mm: list[bool] = []
    ep_steps: list[int] = []

    episodes_target = max(num_episodes, n_envs)
    while len(pos_errs) < episodes_target:
        actions = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(actions)
        cur_step += 1
        newly_solved = info["is_success"] & (first_solved_step == eval_env.cfg.max_episode_steps)
        first_solved_step = np.where(newly_solved, cur_step, first_solved_step)

        dones = terminated | truncated
        for i in np.flatnonzero(dones):
            pe = float(info["pos_err"][i])
            re_deg = float(np.degrees(info["rot_err"][i]))
            pos_errs.append(pe * 1000.0)  # final parked error (mm)
            rot_errs_deg.append(re_deg)   # final parked error (deg)
            successes_5mm.append(bool(info["is_success"][i]))
            successes_2mm.append(bool(info["is_tight_success"][i]))
            ep_steps.append(int(first_solved_step[i]))
            first_solved_step[i] = eval_env.cfg.max_episode_steps
            cur_step[i] = 0
            if len(pos_errs) >= episodes_target:
                break

    pos_arr = np.array(pos_errs, dtype=np.float64)
    rot_arr = np.array(rot_errs_deg, dtype=np.float64)
    return {
        "success_rate_5mm": float(np.mean(successes_5mm) * 100.0),
        "success_rate_2mm": float(np.mean(successes_2mm) * 100.0),
        "mean_pos_err_mm": float(np.mean(pos_arr)),
        "median_pos_err_mm": float(np.median(pos_arr)),
        "mean_rot_err_deg": float(np.mean(rot_arr)),
        "median_rot_err_deg": float(np.median(rot_arr)),
        "mean_steps": float(np.mean(ep_steps)),
    }


def save_training_curves(history: list[dict[str, float]], out_path: Path, title: str = "6D IK") -> None:
    if not history:
        return
    steps = [h["env_steps"] / 1000.0 for h in history]
    sr5 = [h["success_rate_5mm"] for h in history]
    sr2 = [h["success_rate_2mm"] for h in history]
    pos_mm = [h["mean_pos_err_mm"] for h in history]
    pos_med = [h["median_pos_err_mm"] for h in history]
    rot_deg = [h["mean_rot_err_deg"] for h in history]
    rot_med = [h["median_rot_err_deg"] for h in history]
    alphas = [h["alpha"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    fig.suptitle(f"{title} — SAC Training Curves", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(steps, sr5, "o-", lw=2.2, label="Success (< 5 mm & < 2.86°)")
    ax.plot(steps, sr2, "s--", lw=2.0, label="Tight Success (< 2 mm & < 1.15°)")
    ax.set_xlabel("Environment Steps (×10³)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-2, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("6D IK Solve Success Rate")

    ax = axes[0, 1]
    ax.plot(steps, pos_mm, "o-", lw=2.2, color="#d62728", label="Mean Position Error")
    ax.plot(steps, pos_med, "s--", lw=2.0, color="#ff7f0e", label="Median Position Error")
    ax.axhline(5.0, color="gray", linestyle=":", label="5 mm threshold")
    ax.axhline(2.0, color="green", linestyle=":", label="2 mm threshold")
    ax.set_xlabel("Environment Steps (×10³)")
    ax.set_ylabel("Position Error (mm)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("End-Effector Position Error (Log Scale)")

    ax = axes[1, 0]
    ax.plot(steps, rot_deg, "o-", lw=2.2, color="#1f77b4", label="Mean Orientation Error")
    ax.plot(steps, rot_med, "s--", lw=2.0, color="#17becf", label="Median Orientation Error")
    ax.axhline(2.86, color="gray", linestyle=":", label="2.86° (0.05 rad)")
    ax.axhline(1.15, color="green", linestyle=":", label="1.15° (0.02 rad)")
    ax.set_xlabel("Environment Steps (×10³)")
    ax.set_ylabel("Orientation Error (deg)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("End-Effector Orientation Error (Log Scale)")

    ax = axes[1, 1]
    ax2 = ax.twinx()
    mean_st = [h["mean_steps"] for h in history]
    ax.plot(steps, mean_st, "o-", lw=2.2, color="#2ca02c", label="Mean Solve Steps")
    ax2.plot(steps, alphas, "^--", lw=1.8, color="#9467bd", label="SAC Temperature α")
    ax.set_xlabel("Environment Steps (×10³)")
    ax.set_ylabel("Steps to Converge", color="#2ca02c")
    ax2.set_ylabel("Temperature α", color="#9467bd")
    ax.grid(True, alpha=0.3)
    ax.set_title("Convergence Speed & Entropy Temperature")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def set_cosine_lr(agent: SACAgent, base_lr: float, min_lr: float, progress: float) -> float:
    """Cosine learning rate schedule from base_lr down to min_lr."""
    p = min(max(progress, 0.0), 1.0)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * p))
    for group in agent.actor_opt.param_groups:
        group["lr"] = lr
    for group in agent.critic_opt.param_groups:
        group["lr"] = lr
    return lr


def apply_train_defaults(parser: argparse.ArgumentParser, values: dict) -> None:
    """Use the `train:` section of an experiment YAML as defaults for train.py options (CLI flags still win)."""
    actions = {a.dest: a for a in parser._actions if a.dest not in ("help", "exp", "overrides")}
    unknown = set(values) - set(actions)
    if unknown:
        parser.error(f"unknown train options in --exp: {sorted(unknown)}; available: {sorted(actions)}")
    defaults = {}
    for key, value in values.items():
        action = actions[key]
        if action.default is not None:
            value = coerce_value(action.default, value, f"train.{key}")
        elif action.type is not None and value is not None and not isinstance(value, list):
            value = action.type(value)
        defaults[key] = value
    parser.set_defaults(**defaults)


def build_configs(
    args: argparse.Namespace, robot: RobotSpec, overrides: dict[str, dict]
) -> tuple[IKConfig, SACConfig]:
    """Defaults < robot env_overrides < --exp < --set / --obs-features."""
    env_overrides = dict(overrides.get("env", {}))
    if args.obs_features:
        env_overrides["obs_features"] = tuple(args.obs_features)
    env_cfg = IKConfig.for_robot(robot, **env_overrides)

    sac_overrides = overrides.get("sac", {})
    derived = sorted({"obs_dim", "act_dim", "batch_size"} & set(sac_overrides))
    if derived:
        raise ValueError(f"sac.{derived[0]} is set by train.py (from the robot / observation, or --batch-size)")
    n = robot.n_joints
    sac_cfg = SACConfig.from_overrides(
        {
            "target_entropy": -float(n),
            **sac_overrides,
            "obs_dim": env_cfg.obs_dim(n),
            "act_dim": n,
            "batch_size": args.batch_size,
        },
        section="sac",
    )
    return env_cfg, sac_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC for 6D IK in MuJoCo")
    parser.add_argument(
        "--robot", type=str, default=DEFAULT_ROBOT,
        help=f"Robot name ({', '.join(available_robots())}) or path to a robot YAML config",
    )
    parser.add_argument(
        "--obs-features", nargs="+", default=None, metavar="NAME",
        help=f"Observation feature groups to use (default: all): {' '.join(OBS_FEATURES)}",
    )
    parser.add_argument(
        "--exp", type=str, default=None, metavar="YAML",
        help="Experiment YAML with optional env: / sac: / train: sections (see configs/exp/example.yaml)",
    )
    parser.add_argument(
        "--set", nargs="+", default=[], metavar="SECTION.KEY=VALUE", dest="overrides",
        help="Override IKConfig / SACConfig fields, e.g. env.reward_scale=0.3 sac.gamma=0.97 (wins over --exp)",
    )
    parser.add_argument("--total-steps", type=int, default=1_600_000, help="Total environment steps")
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel MuJoCo environments")
    parser.add_argument("--updates-per-rollout", type=int, default=6, help="SAC updates per rollout step")
    parser.add_argument("--batch-size", type=int, default=1024, help="SAC minibatch size")
    parser.add_argument("--buffer-size", type=int, default=2_000_000, help="GPU replay buffer capacity")
    parser.add_argument("--warmup-steps", type=int, default=16_000, help="Initial random exploration steps")
    parser.add_argument("--eval-interval", type=int, default=80_000, help="Evaluation interval in env steps")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Deterministic eval episodes")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on SAC losses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Parent directory for run folders")
    parser.add_argument(
        "--run-name", type=str, default=None, help="Run folder name (default: start timestamp YYYYmmdd_HHMMSS)"
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Override outputs dir (default: <run>/outputs)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoint dir (default: <run>/checkpoints)")
    pre_args, _ = parser.parse_known_args()
    try:
        exp = load_experiment(pre_args.exp) if pre_args.exp else {}
        apply_train_defaults(parser, exp.pop("train", {}))
        args = parser.parse_args()
        cli = parse_set_args(args.overrides)
        if "train" in cli:
            parser.error("--set train.* is not supported; use the command-line flag (e.g. --total-steps)")
        overrides = merge_overrides(exp, cli)
        robot = get_robot(args.robot)
        env_cfg, sac_cfg = build_configs(args, robot, overrides)
    except (ValueError, KeyError) as e:
        parser.error(str(e))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.runs_dir) / robot.name / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "outputs"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    obs_size, act_size = sac_cfg.obs_dim, sac_cfg.act_dim
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "robot": robot.name,
                "robot_spec": robot.to_dict(),
                "args": vars(args),
                "overrides": overrides,
                "env_config": env_cfg.to_dict(),
                "sac_config": sac_cfg.to_dict(),
            },
            f,
            indent=2,
        )
    print(f"[Init] Run directory: {run_dir}")

    print(f"[Init] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"[Init] Creating {args.num_envs} vectorized {robot.display_name} MuJoCo envs...")
    vec_env = VectorIKEnv(robot, num_envs=args.num_envs, cfg=env_cfg, seed=args.seed, allow_chain_reset=True)
    eval_env = VectorIKEnv(robot, num_envs=32, cfg=env_cfg, seed=args.seed + 1000, allow_chain_reset=False)

    agent = SACAgent(cfg=sac_cfg, device=device, use_compile=args.compile)
    buffer = GPUReplayBuffer(
        obs_dim=obs_size, act_dim=act_size, capacity=args.buffer_size, device=device
    )

    obs = vec_env.reset()
    env_steps = 0
    grad_updates = 0
    next_eval_step = args.eval_interval
    best_score = float("inf")
    history: list[dict[str, float]] = []

    t_start = time.perf_counter()
    last_metrics: dict[str, float] = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "alpha": sac_cfg.init_alpha,
        "q_mean": 0.0,
        "entropy": 0.0,
    }

    print(
        f"[Train] Starting SAC | total_steps={args.total_steps:,} | "
        f"batch_size={args.batch_size} | updates_per_rollout={args.updates_per_rollout}"
    )

    while env_steps < args.total_steps:
        if env_steps < args.warmup_steps:
            actions = np.random.uniform(-1.0, 1.0, size=(args.num_envs, act_size)).astype(np.float32)
        else:
            actions = agent.select_action(obs, deterministic=False)

        next_obs, rewards, terminated, truncated, info = vec_env.step(actions)
        # Store the true pre-reset next observation; `terminated` (not truncated) gates bootstrapping
        buffer.add_batch(obs, actions, rewards, info["terminal_next_obs"], terminated)
        obs = next_obs
        env_steps += args.num_envs

        if env_steps >= args.warmup_steps and buffer.size >= args.batch_size:
            progress = (env_steps - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
            cur_lr = set_cosine_lr(agent, base_lr=sac_cfg.actor_lr, min_lr=4e-5, progress=progress)

            for _ in range(args.updates_per_rollout):
                b_obs, b_act, b_rew, b_nobs, b_done = buffer.sample(args.batch_size)
                last_metrics = agent.update(b_obs, b_act, b_rew, b_nobs, b_done)
                grad_updates += 1
        else:
            cur_lr = sac_cfg.actor_lr

        if env_steps >= next_eval_step or env_steps >= args.total_steps:
            eval_stats = evaluate_policy(agent, eval_env, num_episodes=args.eval_episodes)
            elapsed = time.perf_counter() - t_start
            sps = env_steps / max(elapsed, 1e-6)

            record = {
                "env_steps": env_steps,
                "grad_updates": grad_updates,
                "wall_time_sec": round(elapsed, 2),
                "lr": cur_lr,
                **last_metrics,
                **eval_stats,
            }
            history.append(record)

            print(
                f"[Step {env_steps:7,d}/{args.total_steps:,d}] "
                f"SR(5mm): {eval_stats['success_rate_5mm']:5.1f}% | "
                f"SR(2mm): {eval_stats['success_rate_2mm']:5.1f}% | "
                f"PosErr: {eval_stats['mean_pos_err_mm']:5.2f}mm (med {eval_stats['median_pos_err_mm']:4.2f}mm) | "
                f"RotErr: {eval_stats['mean_rot_err_deg']:4.2f}° (med {eval_stats['median_rot_err_deg']:4.2f}°) | "
                f"Steps: {eval_stats['mean_steps']:4.1f} | "
                f"α: {last_metrics['alpha']:.4f} | "
                f"SPS: {sps:,.0f} | t={elapsed:.1f}s",
                flush=True,
            )

            # Prioritize highest 5mm & 2mm success rates, breaking ties by median error
            score = (
                -10.0 * eval_stats["success_rate_5mm"]
                - 5.0 * eval_stats["success_rate_2mm"]
                + eval_stats["median_pos_err_mm"]
                + eval_stats["median_rot_err_deg"]
            )
            meta = {
                "robot": robot.name,
                "robot_spec": robot.to_dict(),
                "env_config": env_cfg.to_dict(),
                "eval_stats": eval_stats,
                "env_steps": env_steps,
            }
            agent.save(ckpt_dir / "latest_model.pt", extra_meta=meta)
            if score < best_score:
                best_score = score
                agent.save(ckpt_dir / "best_model.pt", extra_meta=meta)

            with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            save_training_curves(history, out_dir / "training_curves.png", title=f"{robot.display_name} 6D IK")

            next_eval_step += args.eval_interval

    total_time = time.perf_counter() - t_start
    print(f"\n[Done] Training completed in {total_time:.1f}s! Best checkpoint saved to {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()

