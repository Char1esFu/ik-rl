"""SAC+DLS-specific evaluation: fixed-horizon ("parked") rollouts in the training environment.

evaluate_parked runs max_episode_steps deterministic steps (no early exit) and reports the final error, i.e.
checks that the policy also holds still at the target. Logged as an extra metric during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from policy.sac_dls.env import VectorIKEnv

if TYPE_CHECKING:
    from policy.sac_dls.sac import SACAgent


def evaluate_parked(agent: SACAgent, eval_env: VectorIKEnv, num_episodes: int = 128) -> dict[str, float]:
    """Fixed-horizon deterministic rollouts on fresh random targets; success / error at the final step."""
    obs = eval_env.reset()
    n_envs = eval_env.num_envs
    horizon = eval_env.cfg.max_episode_steps
    successes_5mm: list[bool] = []
    successes_2mm: list[bool] = []
    pos_errs: list[float] = []
    rot_errs: list[float] = []

    episodes_target = max(num_episodes, n_envs)
    while len(pos_errs) < episodes_target:
        actions = agent.select_action(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(actions)
        for i in np.flatnonzero(terminated | truncated):
            pos_errs.append(float(info["pos_err"][i]) * 1000.0)
            rot_errs.append(float(np.degrees(info["rot_err"][i])))
            successes_5mm.append(bool(info["is_success"][i]))
            successes_2mm.append(bool(info["is_tight_success"][i]))
            if len(pos_errs) >= episodes_target:
                break

    return {
        "success_rate_5mm": float(np.mean(successes_5mm) * 100.0),
        "success_rate_2mm": float(np.mean(successes_2mm) * 100.0),
        "median_pos_err_mm": float(np.median(pos_errs)),
        "median_rot_err_deg": float(np.median(rot_errs)),
        "horizon": float(horizon),
    }
