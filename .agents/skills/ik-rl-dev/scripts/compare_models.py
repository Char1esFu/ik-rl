"""Head-to-head evaluation of IK checkpoints on identical targets/initial configs.

Metrics (ik_rl.evaluation, same protocol as eval.py and the training curves):
  - solver: SACIKSolver, stop at 2 mm / 0.02 rad, max 120 steps, restarts 1 and 4
  - parked: fixed-horizon rollout in the training env, final error after max_episode_steps (65)
Usage (from the repo root):
  PYTHONPATH=. .venv/bin/python .agents/skills/ik-rl-dev/scripts/compare_models.py <robot> name=path/to/best_model.pt [name2=...]
The target set depends only on <robot> and SEED, so results are comparable across runs and code versions
as long as target sampling is unchanged.
"""
import json
import sys

import numpy as np

from ik_rl import SACIKSolver
from ik_rl.env import IKEnv, VectorIKEnv
from ik_rl.evaluation import EvalTargets, evaluate_parked, evaluate_solver

N_TESTS = 2000
SEED = 7

ROBOT = sys.argv[1]
pairs = [a.split("=", 1) for a in sys.argv[2:]]
targets = EvalTargets.sample(IKEnv(ROBOT), N_TESTS, np.random.default_rng(SEED))

results = {}
for name, path in pairs:
    solver = SACIKSolver(path, device="cuda")
    res = {}
    for k in (1, 4):
        m, sols = evaluate_solver(solver, targets, num_restarts=k, max_steps=120)
        pe = np.array([s.pos_err_m for s in sols])
        res[f"r{k}"] = {
            "sr_5mm": round(m["success_rate_5mm"], 2),
            "sr_2mm": round(m["success_rate_2mm"], 2),
            "fail_cnt_2mm": int(round(N_TESTS * (100 - m["success_rate_2mm"]) / 100)),
            "pos_mean_mm": round(m["mean_pos_err_mm"], 3),
            "pos_p99_mm": round(float(np.percentile(pe, 99) * 1000), 3),
            "rot_mean_deg": round(m["mean_rot_err_deg"], 3),
            "mean_steps": round(m["mean_steps"], 2),
        }
    ev = VectorIKEnv(solver.robot, num_envs=32, cfg=solver.cfg, seed=SEED + 1000, allow_chain_reset=False)
    res["parked"] = {k: round(v, 3) for k, v in evaluate_parked(solver.agent, ev, num_episodes=1024).items()}
    results[name] = res
    print(name, json.dumps(res, indent=1), flush=True)

print("\nSUMMARY_JSON " + json.dumps(results))
