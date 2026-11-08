"""PolicyIK training: PPO for 6D IK of a registered robot (train.py --method policyik; see README.md here).

Runs are written to runs/policyik/<robot>/<YYYYmmdd_HHMMSS>/ (config.json, train.log, checkpoints/, outputs/).
The deployed policy is checkpoints/best_model.pt: the exponentially weighted average of the last 20 periodic
checkpoints (as in PolicyIK); checkpoints/final_model.pt is the last iterate.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
from pathlib import Path
import signal
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.batch_kinematics import BatchKinematics
from core.robot import available_robots, get_robot
from evaluation.hold import HoldTasks, evaluate_hold
from policy.policyik.config import PolicyIKConfig, PPOConfig
from policy.policyik.controller import PolicyIKController
from policy.policyik.env import PolicyIKVecEnv
from policy.policyik.ppo import PPOAgent, average_state_dicts
from utils.config import parse_set_args
from utils.runs import method_runs_dir

METHOD = "policyik"
EXP_SECTIONS = ("env", "ppo")  # --set sections: PolicyIKConfig / PPOConfig
RESUME_FILE = "resume_state.pt"


def parse_set(items: list[str]) -> dict[str, dict]:
    return {"env": {}, "ppo": {}, **parse_set_args(items, EXP_SECTIONS)}


def plot_curves(history: list[dict], path: Path, title: str) -> None:
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=120)
    fig.suptitle(f"{title} — PolicyIK (PPO) training", fontweight="bold")
    ax = axes[0]
    for k, lab in [("strict", "strict (1 mm, 0.2°)"), ("coarse", "coarse (1 cm, 2°)"), ("entry", "entry")]:
        ax.plot(ep, [h[k] for h in history], "o-", ms=3, label=lab)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("epoch")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("hold protocol, single attempt (160 steps)")
    ax = axes[1]
    ax.semilogy(ep, [h["pos_median_mm"] for h in history], "o-", ms=3, label="median pos (mm)")
    ax.semilogy(ep, [h["ori_median_deg"] for h in history], "s-", ms=3, label="median ori (deg)")
    ax.axhline(1.0, color="gray", ls=":")
    ax.axhline(0.2, color="gray", ls="--")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("final-second mean errors")
    ax = axes[2]
    ax.plot(ep, [h["train_reward"] for h in history], "-", label="mean step reward")
    ax2 = ax.twinx()
    ax2.plot(ep, [h["train_in_tol"] for h in history], "-", color="C1", label="train in-tolerance (%)")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("training")
    fig.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="franka", help=f"{', '.join(available_robots())} or a YAML path")
    ap.add_argument("--epochs", type=int, default=None, help="PPO epochs, num_envs x horizon transitions each (default 13000)")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--backend", default=None, choices=["warp", "cpu"], help="kinematics (default: warp on CUDA)")
    ap.add_argument("--workers", type=int, default=8, help="kinematics worker processes (cpu backend)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=250, help="epochs between hold-protocol evaluations")
    ap.add_argument("--eval-tasks", type=int, default=500)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], help="env.KEY=VALUE / ppo.KEY=VALUE")
    ap.add_argument("--runs-dir", default=str(method_runs_dir(METHOD)), help="parent of <robot>/<run> folders")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="run directory to continue (from checkpoints/resume_state.pt)")
    ap.add_argument("--resume-every", type=int, default=500, help="epochs between resume-state saves")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.resume)
        saved = json.loads((run_dir / "config.json").read_text())
        for k in ("robot", "num_envs", "seed", "eval_tasks", "eval_every"):
            setattr(args, k, saved["args"][k])
        args.epochs = args.epochs or saved["args"]["epochs"]
        overrides = saved["overrides"]
    else:
        overrides = parse_set(args.overrides)
        args.epochs = args.epochs or 13000
    robot = get_robot(args.robot)
    if not args.resume:
        run_dir = Path(args.runs_dir) / robot.name / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    cfg = PolicyIKConfig.from_overrides(overrides["env"], section="env")
    ppo = PPOConfig.from_overrides(overrides["ppo"], section="ppo")
    ckpt_dir, out_dir = run_dir / "checkpoints", run_dir / "outputs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bk = BatchKinematics(robot, backend=args.backend, num_workers=args.workers,
                         max_batch=max(args.num_envs, args.eval_tasks))
    args.backend = bk.backend
    t0 = time.time()
    env = PolicyIKVecEnv(bk, cfg, args.num_envs, seed=args.seed)
    n = bk.n
    agent = PPOAgent(env.obs_dim, n, ppo, device=args.device)
    dev = agent.device
    tasks = HoldTasks.sample(robot, args.eval_tasks, seed=123)
    print(f"[Init] {robot.display_name}: obs {env.obs_dim}, act {n}, {args.num_envs} envs, {bk.backend} kinematics, "
          f"pools built in {time.time() - t0:.1f}s; run dir {run_dir}", flush=True)

    meta = {"robot": robot.name, "robot_spec": robot.to_dict(), "env_config": cfg.to_dict()}
    history: list[dict] = []
    snapshots: collections.deque = collections.deque(maxlen=ppo.average_last)  # (epoch, state_dict)
    start_epoch = 0
    if args.resume:
        st = torch.load(run_dir / "checkpoints" / RESUME_FILE, map_location="cpu", weights_only=False)
        agent.net.load_state_dict(st["model"])
        agent.opt.load_state_dict(st["opt"])
        agent.lr = st["lr"]
        history, start_epoch = st["history"], st["epoch"]
        snapshots.extend(st["snapshots"])
        env.common_step = st["common_step"]
        print(f"[Resume] from epoch {start_epoch}", flush=True)

    def write_config(**extra) -> None:
        (run_dir / "config.json").write_text(json.dumps(
            {"method": METHOD, "args": vars(args), "overrides": overrides, "robot": robot.name,
             "env_config": cfg.to_dict(), "ppo_config": ppo.to_dict(), **extra}, indent=2, default=str))

    write_config()
    stop = {"flag": False}

    def on_signal(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        print("[Signal] stopping after this epoch", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def save_resume(epoch: int) -> None:
        tmp = ckpt_dir / (RESUME_FILE + ".tmp")
        torch.save({"model": agent.net.state_dict(), "opt": agent.opt.state_dict(), "lr": agent.lr,
                    "history": history, "epoch": epoch, "snapshots": list(snapshots),
                    "common_step": env.common_step}, tmp)
        tmp.replace(ckpt_dir / RESUME_FILE)

    def evaluate(net_state: dict | None = None) -> dict:
        ev = agent
        if net_state is not None:
            ev = PPOAgent(env.obs_dim, n, ppo, device=args.device)
            ev.net.load_state_dict(net_state)
        ev.net.eval()
        m, _ = evaluate_hold(PolicyIKController(ev, bk, cfg), tasks, deploy=False)
        agent.net.train()
        return m["single"]

    obs = env.reset()
    H, N = ppo.horizon, args.num_envs
    buf = {
        "obs": torch.zeros(H, N, env.obs_dim, device=dev),
        "act": torch.zeros(H, N, n, device=dev),
        "logp": torch.zeros(H, N, device=dev),
        "mu": torch.zeros(H, N, n, device=dev),
        "value": torch.zeros(H, N, device=dev),
        "reward": torch.zeros(H, N, device=dev),
    }
    roll_rew = roll_tol = 0.0
    t_last = time.time()
    epoch = start_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        agent.net.train()
        in_tol = torch.zeros((), device=dev)
        rew_sum = torch.zeros((), device=dev)
        for t in range(H):
            o = obs.to(dev)
            a, logp, mu, v = agent.net.act(o)
            buf["obs"][t], buf["act"][t], buf["logp"][t], buf["mu"][t], buf["value"][t] = o, a, logp, mu, v
            obs, rew, time_out, final_obs, info = env.step(a)
            r = rew.to(dev)
            with torch.no_grad():  # bootstrap time-outs with V(s_T) (rows without time-out add 0)
                _, v_T = agent.net(final_obs.to(dev))
                v_T = agent.net.value_rms.denormalize(v_T.unsqueeze(-1)).squeeze(-1)
            r = r + ppo.gamma * v_T * time_out.to(dev).float()
            buf["reward"][t] = r
            in_tol += info["in_tol"].float().mean().to(dev)
            rew_sum += rew.mean().to(dev)
        with torch.no_grad():
            _, last_v = agent.net(obs.to(dev))
        buf["last_value"] = last_v
        buf["std"] = agent.net.log_std.exp().detach()
        stats = agent.update(buf)
        roll_rew, roll_tol = float(rew_sum) / H, 100 * float(in_tol) / H

        if epoch % 50 == 0:
            dt = time.time() - t_last
            t_last = time.time()
            print(f"[ep {epoch:5d}] rew {roll_rew:+.4f} in_tol {roll_tol:5.1f}% kl {stats['kl']:.4f} "
                  f"lr {stats['lr']:.1e} std {stats['std']:.3f} c_loss {stats['c_loss']:.3f} "
                  f"| {50 * H * N / dt / 1e3:.0f}k steps/s", flush=True)
        if epoch % ppo.save_every == 0:
            snapshots.append((epoch, {k: v.detach().cpu().clone() for k, v in agent.net.state_dict().items()}))
            agent.save(ckpt_dir / "final_model.pt", {**meta, "epoch": epoch})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = evaluate()
            history.append({"epoch": epoch, "transitions": epoch * H * N, "train_reward": roll_rew,
                            "train_in_tol": roll_tol, **m})
            print(f"[eval {epoch}] strict {m['strict']:.1f}% coarse {m['coarse']:.1f}% entry {m['entry']:.1f}% "
                  f"pos med {m['pos_median_mm']:.3f} mm ori med {m['ori_median_deg']:.3f} deg", flush=True)
            (out_dir / "training_metrics.json").write_text(json.dumps(history, indent=1))
            plot_curves(history, out_dir / "training_curves.png", robot.display_name)
        if epoch % args.resume_every == 0:
            save_resume(epoch)
        if stop["flag"]:
            break

    stop_reason = "interrupted" if stop["flag"] else "completed"
    save_resume(epoch)
    agent.save(ckpt_dir / "final_model.pt", {**meta, "epoch": epoch})
    result = {"stop_reason": stop_reason, "epochs": epoch, "transitions": epoch * H * N}
    if snapshots:
        last = snapshots[-1][0]
        weights = [0.5 ** ((last - e) / ppo.average_half_life) for e, _ in snapshots]
        avg = average_state_dicts([s for _, s in snapshots], weights)
        agent.save(ckpt_dir / "best_model.pt", {**meta, "epoch": epoch, "averaged_epochs": [e for e, _ in snapshots]},
                   state_dict=avg)
        result["final_eval"] = evaluate()
        result["averaged_eval"] = evaluate(avg)
        print(f"[Done] final strict {result['final_eval']['strict']:.2f}% | "
              f"averaged ({len(snapshots)} ckpts) strict {result['averaged_eval']['strict']:.2f}%", flush=True)
    write_config(result=result)
    bk.close()


if __name__ == "__main__":
    main()
