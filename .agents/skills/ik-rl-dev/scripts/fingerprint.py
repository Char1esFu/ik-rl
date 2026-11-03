"""Env rollout fingerprint for a robot (used to verify refactors are behavior-preserving).

Usage (from the repo root):
  PYTHONPATH=. .venv/bin/python .agents/skills/ik-rl-dev/scripts/fingerprint.py franka|ur5e
Hashes obs / reward / info of seeded random rollouts (vectorized + single env, default IKConfig).
Record the value BEFORE a change and compare AFTER; any difference means behavior changed.
"""
import hashlib
import sys

import numpy as np

from ik_rl.config import IKConfig
from ik_rl.env import IKEnv, VectorIKEnv

robot = sys.argv[1] if len(sys.argv) > 1 else "franka"
h = hashlib.sha256()
venv = VectorIKEnv(robot, num_envs=16, cfg=IKConfig(), seed=3)
n = venv.act_dim
obs = venv.reset()
rng = np.random.default_rng(0)
for _ in range(200):
    a = rng.uniform(-1, 1, size=(16, n)).astype(np.float32)
    obs, r, te, tr, info = venv.step(a)
    for x in (obs, r, te, tr, info["terminal_next_obs"], info["pos_err"], info["rot_err"],
              info["is_success"], info["is_tight_success"], info["collided"]):
        h.update(np.ascontiguousarray(x).tobytes())

env = IKEnv(robot)
o, _ = env.reset(seed=5)
for _ in range(100):
    o, r, te, tr, info = env.step(rng.uniform(-1, 1, size=n).astype(np.float32))
    h.update(o.tobytes()); h.update(np.float32(r).tobytes())
print(f"{robot} env fingerprint:", h.hexdigest()[:16])
