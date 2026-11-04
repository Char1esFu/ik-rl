# `ik_rl`: Learned 6D IK with SAC

Technical details of the shared IK pipeline (environment, observation, action, reward, SAC, solver) and per-robot results. Everything robot-specific is a YAML file in [`configs/`](../configs) loaded into a `RobotSpec` ([`robot.py`](robot.py)). For installation and usage, see the [root README](../README.md).

Sizes below are given for a robot with $n$ arm joints ($n = 7$ for the Franka Panda, $6$ for the UR5e, $5$ for the SO-ARM101).

## Results

500 random reachable target poses all around each robot, with random start configurations (`eval.py`, solver stops at 2 mm / 0.02 rad, at most 120 steps; latest run in `runs/<robot>/`). Solve time: single `solve()` calls on an RTX 4090.

| Robot | Restarts | Success (< 5 mm & < 2.86°) | Success (< 2 mm & < 1.15°) | Mean steps | Mean solve time |
|---|---|---|---|---|---|
| Franka Panda | 1 | 94.8% | 93.8% | 22.8 | 8.2 ms |
| Franka Panda | 4 | 99.4% | 99.2% | 15.0 | 5.8 ms |
| UR5e | 1 | 88.2% | 85.6% | 29.3 | 13.3 ms |
| UR5e | 4 | 99.4% | 99.2% | 20.5 | 8.2 ms |
| SO-ARM101 | 1 | 99.4% | 99.0% | 15.7 | 4.7 ms |
| SO-ARM101 | 4 | 99.8% | 99.8% | 11.2 | 4.5 ms |

Single-start failures are mostly local minima far from the target (not step-limit or near misses); multi-start solving (one GPU batch) recovers almost all of them. The UR5e is non-redundant (up to 8 discrete IK solutions per pose) and its joints turn +-2 pi; targets and starts are sampled within one turn, because "wound-up" start configurations near +-2 pi got pinned at joint limits (single-start 77% with a +-2 pi sampling box). Its remaining single-start failures are partly collision-blocked paths.

The SO-ARM101 has only 5 joints, so not every 6D pose is reachable: the gripper's approach direction must lie in the vertical plane through the shoulder-pan axis and the TCP. Evaluation targets are forward kinematics of sampled joint configurations and are therefore reachable; for arbitrary user-specified poses (viewer sliders, *Point Down*) the policy reaches the closest pose it can. The same reward and hyperparameters as for the other robots are used (no `env_overrides`).

## Package Layout

| File | Contents |
|---|---|
| `robot.py` | `RobotSpec`, loaded from `configs/<robot>.yaml` |
| `config.py` | `IKConfig` (environment / action / reward / observation choice), `SACConfig`, experiment overrides |
| `obs.py` | `OBS_FEATURES` and the observation builder `compute_obs_batch` |
| `env.py` | `RobotKinematics` (FK / Jacobian / collision, `apply_joint_step`), `IKEnv`, `VectorIKEnv`, action scaling and reward |
| `sac.py` | SAC agent (actor, twin critics, entropy tuning) |
| `replay_buffer.py` | GPU replay buffer |
| `rollout.py` | batched policy rollout (`rollout_groups`): N targets x k start branches in one GPU batch |
| `solver.py` | `SACIKSolver` inference API (`solve`, `solve_batch`), built on `rollout.py` |
| `evaluation.py` | evaluation protocol shared by training, `eval.py` and comparison scripts |
| `math_utils.py` | rotation utilities |
| `runs.py` | locate checkpoints in `runs/<robot>/` |

## Task Setup

Robot-specific values come from `configs/<robot>.yaml`; see the comments there.

- **Model and TCP**: the menagerie MJCF scene with a TCP site added to `ee_body` at `ee_offset` / `ee_quat` (Franka: `hand` + 0.1034 m, fingers fixed open; UR5e: tool flange on `wrist_3_link`).
- **Targets**: joint configurations are sampled uniformly from the `sample_q_low` / `sample_q_high` box; the resulting TCP pose is accepted as a target if the configuration is collision-free and the TCP lies in the `workspace_low` / `workspace_high` box. Every target is therefore reachable. For all robots the box is the physical joint range minus a 0.25 rad margin ("soft limits", as on a real controller), so targets surround the robot instead of covering only the region in front of it. Sampling right up to the hard limits was tested on the Franka and lowered single-start success, mostly for targets with a joint near its limit. No special handling of the base-joint seam (e.g. Franka's ±166°) is needed: the policy learns which way to turn, and targets behind the robot are solved as reliably as those in front.
- **Start configurations** (training): 25% home pose + $\mathcal{N}(0, 0.25^2)$ joint noise, 30% continue from the previous episode's final configuration, otherwise uniform from the joint box; all collision-free.
- **Collisions**: each candidate configuration is checked with `mj_collision` (self and ground). Colliding moves are rejected and the arm stays in place.
- **Episodes**: fixed horizon of 65 steps without termination; the critic bootstraps at truncation, so the policy learns to reach the target and then hold still.
- **Success**: $d_{pos} < 5$ mm and $d_{rot} < 0.05$ rad; tight success: $d_{pos} < 2$ mm and $d_{rot} < 0.02$ rad.

## Orientation Error

The orientation error is the world-frame axis-angle vector

$$
\omega_{err} = \mathrm{Log}(R_{goal} R_{cur}^\top), \qquad d_{rot} = \|\omega_{err}\| \in [0, \pi],
$$

computed with $\theta = \mathrm{atan2}(\|\mathrm{vee}(R_{err} - R_{err}^\top)\|, \mathrm{tr}(R_{err}) - 1)$ for full precision near $0$, with a dedicated branch near $\pi$.

## Observation

The observation is a concatenation of feature groups defined in `OBS_FEATURES` ([`obs.py`](obs.py)); its size is derived from that table rather than hard-coded. With all groups (default) it is $11n + 35$: 112-D for the Franka Panda, 101-D for the UR5e.

| Group | Dim | Contents |
|---|---|---|
| `q_norm` | $n$ | joint angles normalized to the joint limits, $2 (q - q_{min}) / (q_{max} - q_{min}) - 1$ |
| `sin_cos_q` | $2n$ | $\sin q$, $\cos q$ (periodic; $q$ and $q \pm 2\pi$ map to the same values) |
| `position` | 14 | $p_{cur}$, $p_{goal}$, $\Delta p$, $\Delta p / \|\Delta p\|$, $\|\Delta p\|$, $\log_{10}\|\Delta p\| / 4$ |
| `orientation` | 20 | $R_{cur}$, $R_{goal}$ in 6D representation (first two rotation-matrix columns, Zhou et al.), $\omega_{err}$, $\omega_{err} / d_{rot}$, $d_{rot} / \pi$, $\log_{10} d_{rot} / 4$ |
| `jacobian` | $6n$ | $0.5 \, J$, the 6×n TCP Jacobian |
| `jt_direction` | $n$ | $\tanh(J^\top [\hat{u}_{pos}; \hat{u}_{rot}])$, the Jacobian-transpose (gradient) direction toward the target |
| `prev_action` | $n$ | previous action |
| `prev_collided` | 1 | whether the previous step collided |

The direction and log-scale features keep the input informative from meter-level to sub-millimeter errors. Groups can be selected with `train.py --obs-features ...` (stored in the checkpoint, so evaluation and the viewer rebuild the same observation). Ablation on the Franka Panda (2 seeds each, 2000 targets): removing `jacobian` + `jt_direction` changes single-start success from 98.2% to 97.6% and increases the mean number of steps by ~1 (16.8 → 17.9).

## Action

The action $a \in [-1, 1]^n$ is a gain-scheduled joint increment:

$$
\Delta q = \Delta q_{max} \cdot s \cdot a, \qquad s = \mathrm{clip}\!\left(\max\!\left(\frac{d_{pos}}{0.12}, \frac{d_{rot}}{0.40}\right), 0.012, 1\right),
$$

with $\Delta q_{max} = 0.18$ rad. Far from the target the arm takes full steps; near it the step size shrinks to ~2 mrad, giving fine resolution for millimeter-level convergence.

## Reward

Dense reward computed on the next state:

$$
r = 0.25 \left[ K_{pos} + K_{rot} + 1.0 \cdot \mathbb{1}_{succ} - 0.015 \left\| \frac{\Delta q}{\Delta q_{max}} \right\|^2 - 0.04 \, \|a_t - a_{t-1}\|^2 - 1.0 \cdot \mathbb{1}_{col} \right]
$$

Position and orientation share one kernel after normalization, so both exert the same precision pressure at their tolerances:

$$
K(e_c, e_f) = -0.25 \, e_c + 1.0 \, e^{-2 e_c} + 1.5 \, e^{-0.4 e_f}
$$

| | Coarse $e_c$ | Fine $e_f$ |
|---|---|---|
| Position | $d_{pos} / 0.12$ m | $d_{pos} / 2$ mm |
| Orientation | $d_{rot} / 0.40$ rad | $d_{rot} / 0.02$ rad |

- **Linear term**: unbounded global guidance far from the target.
- **Coarse kernel**: length scale ~6 cm / 0.2 rad.
- **Fine kernel**: length scale 2.5× the tight tolerance (5 mm / 0.05 rad).
- $\mathbb{1}_{succ}$: tight success (both tolerances, no collision), paid every step it holds.

All weights are fields of `IKConfig` ([`config.py`](config.py)). Override them per experiment with `train.py --exp <yaml>` / `--set env.<field>=<value>` (see the root README); a robot can also override defaults via `RobotSpec.env_overrides`.

## SAC

| Hyperparameter | Value |
|---|---|
| Actor / critics | MLP 512-512-256, LayerNorm + SiLU; twin Q critics with target networks |
| Policy | tanh-squashed Gaussian, $\log\sigma \in [-5.5, 2]$ |
| Learning rate | $3 \times 10^{-4}$, cosine decay to $4 \times 10^{-5}$ (actor and critic) |
| Discount $\gamma$ / Polyak $\tau$ | 0.96 / 0.005 |
| Entropy | automatic tuning, $\alpha_0 = 0.05$, $\alpha \ge 10^{-4}$, target entropy $-n$ |
| Batch size / replay buffer | 1024 / $2 \times 10^6$ (stored on GPU) |
| Gradient clipping | critic 10, actor 5 |

Training uses 64 vectorized environments for at most $5 \times 10^6$ environment steps (16k random warm-up steps, 6 gradient updates per vectorized step; the cosine learning-rate schedule spans the full budget). Every 80k steps the policy is evaluated with the same protocol as `eval.py` (`ik_rl/evaluation.py`): 500 fixed targets with fixed start configurations, solved by `SACIKSolver` with one start (stop at 2 mm / 0.02 rad, at most 120 steps), so the training curves are directly comparable to the results above. `best_model.pt` is selected by the score $-10 \cdot SR_{5mm} - 5 \cdot SR_{2mm} + \text{median } d_{pos} [\text{mm}] + \text{median } d_{rot} [°]$ (lower is better). The fixed-horizon (65-step, "parked") success rate in the training environment is logged as well (`parked_*`). Training stops early when either the mean 2 mm success rate of the last 5 evaluations is $\ge 98.5\%$ (`--early-stop-sr`, `--early-stop-window`) or the best-model score has not improved for 15 evaluations (`--plateau-patience`).

The full training state (networks, optimizers, replay buffer, environment and RNG states) is written to `checkpoints/resume_state.pt` every $10^6$ steps, at the end and on Ctrl+C; `train.py --resume <run_dir>` continues a run exactly (an interrupted and resumed run reproduces an uninterrupted one bit for bit). `resume_state.pt` is large (up to ~2 GB) and not tracked by git.

## Solver

`SACIKSolver.solve()` rolls out the deterministic policy until the best branch is within tolerance (default 5 mm / 0.05 rad, at most 80 steps). With `num_restarts = k`, $k$ branches run as one GPU batch: branch 0 starts from `q_init` if given (plus one branch from the home pose), otherwise from the home pose; the remaining branches start from random collision-free configurations. The branch with the lowest $d_{pos} + 0.2 \, d_{rot}$ is returned.
