"""SAC+DLS training: Soft Actor-Critic for 6D inverse kinematics of a registered robot (train.py --method sac+dls).

Runs are written to runs/sac+dls/<robot>/<YYYYmmdd_HHMMSS>/ (config.json, checkpoints/, outputs/).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.robot import DEFAULT_ROBOT, RobotSpec, available_robots, get_robot
from evaluation.benchmark import EvalTargets, evaluate_solver
from policy.sac_dls.config import EXP_SECTIONS, IKConfig, SACConfig
from policy.sac_dls.env import VectorIKEnv
from policy.sac_dls.evaluation import evaluate_parked
from policy.sac_dls.obs import OBS_FEATURES
from policy.sac_dls.replay_buffer import GPUReplayBuffer
from policy.sac_dls.sac import SACAgent
from policy.sac_dls.solver import SACIKSolver
from utils.config import coerce_value, load_experiment, merge_overrides, parse_set_args
from utils.runs import method_runs_dir

METHOD = "sac+dls"


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
    if all("parked_success_rate_2mm" in h for h in history):
        ax.plot(steps, [h["parked_success_rate_2mm"] for h in history], ":", lw=1.6, color="gray",
                label="Tight Success, fixed horizon (parked)")
    ax.set_xlabel("Environment Steps (×10³)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(-2, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("6D IK Solve Success Rate (solver, 1 start)")

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


RESUME_FILE = "resume_state.pt"
# Options a resumed run may change; everything else comes from the run's config.json
RESUME_OVERRIDABLE = (
    "total_steps", "early_stop_sr", "early_stop_window", "plateau_patience", "resume_interval", "compile",
)


def eval_score(stats: dict[str, float]) -> float:
    """best_model.pt selection score (lower is better): 5 mm / 2 mm success first, median errors break ties."""
    return (
        -10.0 * stats["success_rate_5mm"]
        - 5.0 * stats["success_rate_2mm"]
        + stats["median_pos_err_mm"]
        + stats["median_rot_err_deg"]
    )


def early_stop_reason(history: list[dict[str, float]], args: argparse.Namespace) -> str | None:
    """Early-stop check on the eval history (stateless, so it works unchanged after --resume).

    A (good enough): mean 2 mm success of the last `early_stop_window` evals >= `early_stop_sr`.
    C (plateau): the best_model score has not improved for `plateau_patience` evals.
    """
    if args.early_stop_window > 0 and len(history) >= args.early_stop_window:
        recent = [h["success_rate_2mm"] for h in history[-args.early_stop_window:]]
        mean = sum(recent) / len(recent)
        if mean >= args.early_stop_sr:
            return f"mean SR(2mm) of the last {len(recent)} evals = {mean:.2f}% >= {args.early_stop_sr}%"
    if args.plateau_patience > 0:
        scores = [eval_score(h) for h in history]
        since_best = len(scores) - 1 - scores.index(min(scores))
        if since_best >= args.plateau_patience:
            return f"no new best model for {since_best} evals (plateau)"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the SAC+DLS policy for 6D IK (train.py --method sac+dls)")
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
    parser.add_argument(
        "--resume", type=str, default=None, metavar="RUN_DIR",
        help="Continue a run from <RUN_DIR>/checkpoints/resume_state.pt. Robot and configs come from the run's "
        "config.json; only --total-steps, --early-stop-*, --plateau-patience, --resume-interval and --compile "
        "may change. "
        "Raising --total-steps stretches the cosine learning-rate schedule (the LR rises again)",
    )
    parser.add_argument("--total-steps", type=int, default=5_000_000, help="Total environment steps (upper bound)")
    parser.add_argument(
        "--early-stop-sr", type=float, default=98.5, metavar="PCT",
        help="Early stop (good enough): mean 2 mm / 1.15 deg success rate (%%) of the last --early-stop-window evals",
    )
    parser.add_argument(
        "--early-stop-window", type=int, default=5, metavar="N",
        help="Number of recent evals averaged for --early-stop-sr (0 disables this criterion)",
    )
    parser.add_argument(
        "--plateau-patience", type=int, default=15, metavar="N",
        help="Early stop (plateau): stop when the best-model score has not improved for N evals (0 disables)",
    )
    parser.add_argument(
        "--resume-interval", type=int, default=1_000_000,
        help="Write the resume state every N env steps (also at the end, on early stop and on Ctrl+C; 0 = only then)",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Parallel MuJoCo environments")
    parser.add_argument("--updates-per-rollout", type=int, default=6, help="SAC updates per rollout step")
    parser.add_argument("--batch-size", type=int, default=1024, help="SAC minibatch size")
    parser.add_argument("--buffer-size", type=int, default=2_000_000, help="GPU replay buffer capacity")
    parser.add_argument("--warmup-steps", type=int, default=16_000, help="Initial random exploration steps")
    parser.add_argument("--eval-interval", type=int, default=80_000, help="Evaluation interval in env steps")
    parser.add_argument(
        "--eval-episodes", type=int, default=500,
        help="Evaluation size: fixed solver targets (curves, best model, early stop) and parked episodes",
    )
    parser.add_argument("--eval-max-steps", type=int, default=120, help="Solver step limit in evaluations")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on SAC losses")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--runs-dir", type=str, default=str(method_runs_dir(METHOD)),
        help="Parent directory for <robot>/<run> folders",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Run folder name (default: start timestamp YYYYmmdd_HHMMSS)"
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Override outputs dir (default: <run>/outputs)")
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoint dir (default: <run>/checkpoints)")
    return parser


def setup_new_run(
    parser: argparse.ArgumentParser, argv: list[str] | None
) -> tuple[argparse.Namespace, RobotSpec, IKConfig, SACConfig, dict, Path]:
    pre_args, _ = parser.parse_known_args(argv)
    try:
        exp = load_experiment(pre_args.exp, EXP_SECTIONS) if pre_args.exp else {}
        apply_train_defaults(parser, exp.pop("train", {}))
        args = parser.parse_args(argv)
        cli = parse_set_args(args.overrides, EXP_SECTIONS)
        if "train" in cli:
            parser.error("--set train.* is not supported; use the command-line flag (e.g. --total-steps)")
        overrides = merge_overrides(exp, cli)
        robot = get_robot(args.robot)
        env_cfg, sac_cfg = build_configs(args, robot, overrides)
    except (ValueError, KeyError) as e:
        parser.error(str(e))
    run_dir = Path(args.runs_dir) / robot.name / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    return args, robot, env_cfg, sac_cfg, overrides, run_dir


def setup_resumed_run(
    parser: argparse.ArgumentParser, argv: list[str] | None
) -> tuple[argparse.Namespace, RobotSpec, IKConfig, SACConfig, dict, Path]:
    cli = parser.parse_args(argv)
    if cli.exp or cli.overrides or cli.obs_features:
        parser.error("--exp / --set / --obs-features cannot be combined with --resume (the run's config is reused)")
    run_dir = Path(cli.resume)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        parser.error(f"--resume: {cfg_path} not found")
    with open(cfg_path, encoding="utf-8") as f:
        saved = json.load(f)
    args = argparse.Namespace(**{**vars(cli), **saved["args"]})  # options added later fall back to the CLI defaults
    for key in RESUME_OVERRIDABLE:
        if getattr(cli, key) != parser.get_default(key):
            setattr(args, key, getattr(cli, key))
    args.resume = cli.resume
    robot = RobotSpec.from_dict(saved["robot_spec"])
    env_cfg = IKConfig.from_dict(saved["env_config"])
    sac_cfg = SACConfig.from_dict(saved["sac_config"])
    return args, robot, env_cfg, sac_cfg, saved.get("overrides", {}), run_dir


def save_resume_state(path: Path, payload: dict) -> None:
    """Atomically write the full training state (written to a temp file, then renamed)."""
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    resuming = parser.parse_known_args(argv)[0].resume is not None
    args, robot, env_cfg, sac_cfg, overrides, run_dir = (setup_resumed_run if resuming else setup_new_run)(parser, argv)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "outputs"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_path = ckpt_dir / RESUME_FILE

    obs_size, act_size = sac_cfg.obs_dim, sac_cfg.act_dim
    run_config = {
        "method": METHOD,
        "robot": robot.name,
        "robot_spec": robot.to_dict(),
        "args": vars(args),
        "overrides": overrides,
        "env_config": env_cfg.to_dict(),
        "sac_config": sac_cfg.to_dict(),
    }

    def write_run_config(**extra: object) -> None:
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump({**run_config, **extra}, f, indent=2)

    if resuming:
        with open(run_dir / "config.json", encoding="utf-8") as f:
            previous = json.load(f)
        run_config["resumes"] = previous.get("resumes", [])
    write_run_config()
    print(f"[Init] Run directory: {run_dir}{' (resuming)' if resuming else ''}")

    print(f"[Init] Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"[Init] Creating {args.num_envs} vectorized {robot.display_name} MuJoCo envs...")
    vec_env = VectorIKEnv(robot, num_envs=args.num_envs, cfg=env_cfg, seed=args.seed, allow_chain_reset=True)
    eval_env = VectorIKEnv(robot, num_envs=32, cfg=env_cfg, seed=args.seed + 1000, allow_chain_reset=False)

    agent = SACAgent(cfg=sac_cfg, device=device, use_compile=args.compile)
    # Evaluation = eval.py protocol (single start): fixed targets, solver on the live agent
    eval_solver = SACIKSolver.from_agent(agent, robot, env_cfg)
    eval_targets = EvalTargets.sample(
        robot, args.eval_episodes, np.random.default_rng(args.seed + 2000), home_prob=env_cfg.home_reset_prob
    )
    buffer = GPUReplayBuffer(
        obs_dim=obs_size, act_dim=act_size, capacity=args.buffer_size, device=device
    )

    env_steps = 0
    grad_updates = 0
    next_eval_step = args.eval_interval
    next_resume_save = args.resume_interval
    best_score = float("inf")
    elapsed_before = 0.0
    history: list[dict[str, float]] = []
    last_metrics: dict[str, float] = {
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "alpha": sac_cfg.init_alpha,
        "q_mean": 0.0,
        "entropy": 0.0,
    }

    if resuming:
        if not resume_path.exists():
            parser.error(f"--resume: {resume_path} not found")
        print(f"[Init] Loading resume state {resume_path} ...")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        agent.load_training_state(state["agent"])
        buffer.load_state_dict(state["buffer"])
        obs = vec_env.set_state(state["vec_env"])
        eval_env.set_state(state["eval_env"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if device == "cuda" and state["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        c = state["counters"]
        env_steps, grad_updates = c["env_steps"], c["grad_updates"]
        next_eval_step, best_score = c["next_eval_step"], c["best_score"]
        elapsed_before = c["elapsed_sec"]
        next_resume_save = (env_steps // args.resume_interval + 1) * args.resume_interval if args.resume_interval > 0 else 0
        history, last_metrics = state["history"], state["last_metrics"]
        del state
        run_config["resumes"].append({"from_env_steps": env_steps, "total_steps": args.total_steps,
                                      "at": datetime.now().isoformat(timespec="seconds")})
        write_run_config()
        print(f"[Init] Resumed at env step {env_steps:,} (buffer {buffer.size:,}, {len(history)} evals)")
    else:
        obs = vec_env.reset()

    def resume_payload() -> dict:
        return {
            "version": 1,
            "agent": agent.training_state(),
            "buffer": buffer.state_dict(),
            "vec_env": vec_env.get_state(),
            "eval_env": eval_env.get_state(),
            "rng": {
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            },
            "counters": {
                "env_steps": env_steps,
                "grad_updates": grad_updates,
                "next_eval_step": next_eval_step,
                "best_score": best_score,
                "elapsed_sec": elapsed_before + time.perf_counter() - t_start,
            },
            "history": history,
            "last_metrics": last_metrics,
        }

    # Ctrl+C / SIGTERM: finish the current step, save the resume state and exit (press twice to abort hard)
    stop_signal = {"requested": False}

    def request_stop(signum: int, frame: object) -> None:
        if stop_signal["requested"]:
            raise KeyboardInterrupt
        stop_signal["requested"] = True
        print("\n[Signal] Stopping after the current step and saving the resume state (press again to abort)", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    t_start = time.perf_counter()
    stop_reason = "completed"
    print(
        f"[Train] Starting SAC | total_steps={args.total_steps:,} | "
        f"batch_size={args.batch_size} | updates_per_rollout={args.updates_per_rollout} | early stop: "
        + (f"mean SR(2mm) of last {args.early_stop_window} evals >= {args.early_stop_sr}%"
           if args.early_stop_window > 0 else "SR off")
        + " / "
        + (f"plateau {args.plateau_patience} evals" if args.plateau_patience > 0 else "plateau off")
    )

    while env_steps < args.total_steps:
        if stop_signal["requested"]:
            stop_reason = "interrupted"
            break
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
            eval_stats, _ = evaluate_solver(eval_solver, eval_targets, num_restarts=1, max_steps=args.eval_max_steps)
            parked = evaluate_parked(agent, eval_env, num_episodes=args.eval_episodes)
            eval_stats.update({f"parked_{k}": v for k, v in parked.items() if k != "horizon"})
            elapsed = elapsed_before + time.perf_counter() - t_start
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
                f"parked SR(2mm): {eval_stats['parked_success_rate_2mm']:5.1f}% | "
                f"α: {last_metrics['alpha']:.4f} | "
                f"SPS: {sps:,.0f} | t={elapsed:.1f}s",
                flush=True,
            )

            score = eval_score(eval_stats)
            meta = {
                "method": METHOD,
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

            reason = early_stop_reason(history, args)
            if reason:
                stop_reason = "early_stop"
                print(f"[EarlyStop] {reason} at env step {env_steps:,}", flush=True)
                break

        if args.resume_interval > 0 and env_steps >= next_resume_save:
            save_resume_state(resume_path, resume_payload())
            next_resume_save += args.resume_interval

    save_resume_state(resume_path, resume_payload())
    write_run_config(result={"stop_reason": stop_reason, "env_steps": env_steps, "grad_updates": grad_updates})
    total_time = elapsed_before + time.perf_counter() - t_start
    print(
        f"\n[Done] Training {stop_reason} at env step {env_steps:,} in {total_time:.1f}s. "
        f"Best checkpoint: {ckpt_dir / 'best_model.pt'} | resume state: {resume_path}"
    )


if __name__ == "__main__":
    main()
