"""Head-to-head evaluation of IK checkpoints on identical targets/initial configs.

Metrics:
  - solver (SACIKSolver, early-stop at 2mm/0.02rad, max 120 steps), restarts 1 and 4
  - parked fixed-horizon rollout (train.evaluate_policy): final error after 65 policy steps
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
from train import evaluate_policy

N_TESTS = 2000
SEED = 7

ROBOT = sys.argv[1]
pairs = [a.split("=", 1) for a in sys.argv[2:]]

# Shared target set
env = IKEnv(ROBOT)
rng = np.random.default_rng(SEED)
targets = []
for _ in range(N_TESTS):
    _, p, R = env.sample_reachable_pose(rng)
    q0 = env.sample_initial_q(rng)
    targets.append((p, R, q0))

results = {}
for name, path in pairs:
    solver = SACIKSolver(path, device="cuda")
    res = {}
    for k in (1, 4):
        pe, re, st = [], [], []
        for p, R, q0 in targets:
            s = solver.solve(p, target_rot=R, q_init=q0, max_steps=120,
                             pos_tol=0.002, rot_tol=0.02, num_restarts=k)
            pe.append(s.pos_err_m); re.append(s.rot_err_rad); st.append(s.steps)
        pe, re = np.array(pe), np.array(re)
        res[f"r{k}"] = {
            "sr_5mm": round(float(np.mean((pe < 0.005) & (re < 0.05)) * 100), 2),
            "sr_2mm": round(float(np.mean((pe < 0.002) & (re < 0.02)) * 100), 2),
            "fail_cnt_2mm": int(np.sum(~((pe < 0.002) & (re < 0.02)))),
            "pos_mean_mm": round(float(pe.mean() * 1000), 3),
            "pos_p99_mm": round(float(np.percentile(pe, 99) * 1000), 3),
            "rot_mean_deg": round(float(np.degrees(re.mean())), 3),
            "mean_steps": round(float(np.mean(st)), 2),
        }
    ev = VectorIKEnv(solver.robot, num_envs=32, cfg=solver.cfg, seed=SEED + 1000, allow_chain_reset=False)
    parked = evaluate_policy(solver.agent, ev, num_episodes=1024)
    res["parked65"] = {k: round(v, 3) for k, v in parked.items()}
    results[name] = res
    print(name, json.dumps(res, indent=1), flush=True)

print("\nSUMMARY_JSON " + json.dumps(results))
