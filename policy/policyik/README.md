# PolicyIK (PPO): training and deviations from the paper

A learned 6D IK policy after **PolicyIK**: Y. Liu, X. Liu, Z. Dong, K. W. Ross, S. Wan, *PolicyIK:
High-Precision Inverse Kinematics for Robot Arms with Reinforcement Learning*,
[arXiv:2611.05372](https://arxiv.org/abs/2611.05372), code [Tinaliu0123/PolicyIK](https://github.com/Tinaliu0123/PolicyIK).
It is adapted to this repository's kinematic MuJoCo setting and runs side by side with [SAC+DLS](../sac_dls/README.md).
Robots, kinematics, collision model, target / start samplers, the evaluation protocols, DLS refinement and the
solver API are shared (`core/`, `utils/`, `evaluation/`).

```bash
.venv/bin/python train.py --method policyik --robot franka            # runs/policyik/franka/<YYYYmmdd_HHMMSS>/
.venv/bin/python train.py --resume runs/policyik/franka/<run>         # continue (method / robot read from the run)
.venv/bin/python eval.py --method policyik --robot franka --restarts 4 [--refine]   # this repo's benchmark
.venv/bin/python eval.py --protocol hold --robot franka --dls --refine   # PolicyIK's hold protocol: SAC+DLS vs PolicyIK vs DLS
```

```python
from policy import load_solver                       # SAC+DLS or PolicyIK, by checkpoint
solver = load_solver("runs/policyik/franka/<run>/checkpoints/best_model.pt")
sol = solver.solve(target_pos, target_quat=q_wxyz, num_restarts=4, refine=True)   # same API as SACIKSolver
```

## The method (as in the paper)

PolicyIK treats IK as **goal-conditioned closed-loop control**: at 30 Hz a policy reads the pose error and
outputs an increment of a *joint-position reference*. Success means the arm **reaches and holds** the pose:
mean errors over the final second < 1 mm and < 0.2°.

1. **Accumulated reference with an error-dependent gain.**
   - $z_t = \mathrm{clip}(z_{t-1} + \eta\,\beta(e_p, e_R)\,a_t)$, with $\eta = 0.0625$ rad.
   - $\beta = \max(G(e_p), G(e_R))$ ramps linearly from $\beta_{min}$ near the goal to 1 far away, so the
     step size shrinks near the goal at any time in the episode.
2. **Observation** (`core.py`, $3n + 26$ dims; 47 for the Franka):
   - $q - q_{home}$ and $\dot q$;
   - the goal position;
   - three keypoint offsets (goal − current points 0.3 m along the TCP axes);
   - **two-scale clipped body-frame pose errors**: coarse 1 cm / 20° and fine 0.5 mm / 0.25°;
   - the error history $h_t$;
   - the previous action.
   - There is no TCP orientation and no Jacobian.
3. **Tolerance-normalized progress rewards.** With $\rho_t = \max(e_p / 1\,\mathrm{mm}, e_R / 0.2°)$ and the
   running best $b_t = \min(b_{t-1}, \rho_t)$, reset with every new goal:
   - improvement $26.8\,[U(\rho_t) - U(b_{t-1})]_+$, with $U(\rho) = \tfrac12(1 + \tanh(2(1-\rho)))$; it pays
     only for new bests, so each accuracy level is paid once;
   - potential shaping $\gamma\Phi_t - \Phi_{t-1}$, with $\Phi = -\min(50, \rho)$;
   - backslide cost $-0.2 \min(3, [\rho_t - b_{t-1}]_+)$;
   - **dwell** $\mathbb{1}[\rho_t \le 1]$ for every step inside the tolerance;
   - $h_t = \mathrm{clip}([b_{t-1}, \rho_t - b_{t-1}] / 20, 0, 1)$ is observed, so the reward stays Markov.

   Dense terms:
   - −0.2 e_p and 0.1(1 − tanh(e_p/0.1));
   - keypoint kernels 1 − tanh(ε/0.2) and 1 − tanh(ε/0.05), with ε the mean distance of three centered
     keypoints at 0.45 m;
   - a joint-margin cost, only while ρ > 10;
   - motion costs on action rate and joint velocity, raised after 4500 env steps.

   Every term is multiplied by dt = 1/30.
4. **PPO at scale.**
   - 4096 envs × 24 steps per epoch, 13,000 epochs (1.28·10⁹ transitions), 5 mini-epochs of 24,576 samples;
   - shared 256-256 ELU actor-critic trunk with a state-independent log-std;
   - input and value normalization;
   - adaptive LR, starting at 10⁻³ with KL target 0.01;
   - γ = 0.99, λ = 0.95, clip 0.2, entropy 0.01, value coefficient 2;
   - 12 s episodes with a new goal every 4 s, without resetting the arm.
5. **Checkpoint averaging.** The deployed policy (`best_model.pt`) is the exponentially weighted average
   (half-life 400 epochs) of the last 20 checkpoints, saved every 100 epochs.
6. **Evaluation** (`evaluation/hold.py`, `eval.py --protocol hold`):
   - a single attempt is 160 steps; strict / coarse success is judged on the final-30-step mean errors;
   - the deployment protocol re-commands an error-corrected goal if the attempt ends within 1 cm, then
     runs up to 8 attempts that each start with a transient goal (0.30 m, 90° away, for 40 steps).

Paper results (simulated OpenArm 7-DoF, 4120 tasks):
- 93.5% strict in a single attempt, 98.9% with the deployment protocol;
- DLS / QP / MPC: 50–56%.

## Deviations from the paper

All deviations are deliberate. The values are defaults in `config.py`; the paper's values are noted next to
them.

**1. Kinematic transition instead of PD-tracked dynamics.**
- *Paper:* Isaac Sim, 60 Hz physics; a joint PD controller (stiffness 80, damping 4, no gravity
  feed-forward) tracks $z_t$.
- *Here:* $q_{t+1} = z_t$, clipped to the joint limits. A move that would collide is rejected: the arm stays
  and $z$ is re-anchored at $q$, as in the SAC+DLS environment.
- *Why:* this repository solves kinematic IK. It is a solver that outputs a configuration, not a
  joint-level controller.
- *Effect:* the dynamics-specific benefits of the paper cannot appear: compensating steady-state offsets
  from gravity / PD gains, and robustness to tracking lag.

**2. Command-state feedback $d_t = z_{t-1} - q_t$ omitted.**
- It is identically zero under the kinematic transition. The observation is therefore the paper's
  "w/o command state" variant (47-D for 7 joints).
- In the paper this channel is worth 9.2 points strict, its largest ablation effect, entirely due to
  dynamics.

**3. Gain schedule widened.** This is the one change that training needs.

| | β_min | β = 1 above (position / rotation) | lower ramp end |
|---|---|---|---|
| paper | 0.1 | 9.31 mm / 2.328° | 0.5 mm / 0.415° |
| here | **0.012** | **60 mm / 0.3 rad** | 0.5 mm / 0.415° |

With the paper's schedule, Franka and UR5e training stalled:
- Franka: 0.0% strict at epoch 1250, median error 7.2 mm.
- UR5e: 0.0% strict at epoch 750, median error 10 mm.
- Training rollouts never entered the tolerance: in_tol stayed at 0.0%.
- The 3× smaller SO-101 still learned: 88.6% strict at epoch 750.
- With the widened schedule, at the same epochs: Franka 88.8%, UR5e 62.6%, SO-101 95.0%.

The cause is exploration noise:
- PPO's Gaussian stays at σ ≈ 0.3 to the end (the released PolicyIK checkpoints have σ = 0.28–0.31).
- The TCP jitter per step is about σ·η·β·L·√n ≈ 0.024·β m on the Franka. With β = 1 above 9.3 mm, the
  jitter is about 2.6× the error itself, so noisy rollouts cannot contract into 1 mm and the fine regime
  never yields a learning signal.
- In Isaac Sim, the PD-tracked arm low-pass filters this noise; a kinematic arm executes it directly.
- A noisy oracle confirms it: the DLS direction plus σ = 0.3 action noise is never in tolerance with the
  paper's schedule (0.0%), but 47% of the time with the wider one, the same as without noise.

The new ramp keeps the jitter below 0.5× the error, with a floor of ~0.3 mm / 0.03°. It is the same idea as
SAC+DLS's gain schedule (12 cm / 0.4 rad, floor 0.012). The runs for this comparison are not kept; to
reproduce, use `--set env.gain_min=0.1 env.gain_pos_hi=0.00931 env.gain_rot_hi=0.04063`.

**4. Joint velocity** in the observation and motion cost is the executed step / dt, not a measured velocity.

**5. Collision cost** −1·dt per rejected (colliding) step. This term is new; in the paper, physics handles
contacts.

**6. Workspace, samplers and pools.**
- *Paper:* goals in a cylinder in front of the arm (r ≤ 0.49 m, z 0.15–0.50 m, base keep-out); a fixed
  200k-configuration source set with disjoint evaluation rows.
- *Here:* this repository's all-around workspace and acceptance rules (`RobotKinematics.sample_target` /
  `sample_start`, 25% of starts near the home pose). Training samples 200k start and 200k goal
  configurations at start-up; evaluation tasks are fresh draws (`EvalTargets`, fixed seeds).

**7. One continuous run instead of three segments.**
- The paper trains in three resumed segments (epochs 3500 / 8500 / 13000), and the motion-cost curriculum
  counter restarts in each.
- Here the curriculum switches once, after 4500 env steps.

**8. PPO reimplemented in torch** (`ppo.py`) instead of rl_games, with the same hyperparameters. Small
differences:
- Time-outs bootstrap with $\gamma V(s_T)$ of the true final state. rl_games uses the value of the state
  before the step.
- The observation normalizer is updated once per epoch on the whole rollout, after the update. rl_games
  updates it in the training forward passes.
- When a goal changes, the running best / potential restart from the current error, and the next step's
  progress reward counts normally. Isaac Lab zeroes the progress terms on the reset step.
- Episode clocks are staggered at the first reset.

**9. Simulator.** MuJoCo 4.0 kinematics on the GPU through MuJoCo Warp (`core/batch_kinematics.py`), with a
CPU fallback. The torch core keeps everything on the GPU.

Unchanged from the paper:
- action scale η;
- observation scales;
- all reward weights and kernels;
- tolerance 1 mm / 0.2°;
- PPO hyperparameters and network;
- episode / goal timing;
- training observation noise (±0.01);
- checkpoint averaging;
- hold and deployment evaluation protocols.

## Training

- **Budget:** `train.py --method policyik --robot <robot>` uses the paper's budget, 13,000 epochs = 1.28·10⁹
  transitions. Options: `--epochs`, `--num-envs`, `--seed`, `--eval-every` / `--eval-tasks` (hold-protocol
  evaluations during training), and `--set env.<field>=v ppo.<field>=v` to override `PolicyIKConfig` /
  `PPOConfig`.
- **Speed:** with the Warp backend, one run reaches ~600k steps/s alone (~40 min) or ~340k steps/s with
  three runs sharing a 4090 (~1 h each). The CPU backend (`--backend cpu --workers 16`) reaches ~90k
  steps/s.
- **GPU memory:** ~1.7 GB per run. The Warp contact buffers grow with the number of worlds (~0.8 GB for
  4096 Franka worlds), so batch-size buckets are reused.
- **Outputs** in `runs/policyik/<robot>/<run>/`:
  - `config.json`, `train.log`;
  - `outputs/training_curves.png` and `training_metrics.json` (hold-protocol single attempt on 500 fixed
    tasks every 250 epochs);
  - `checkpoints/best_model.pt` (the averaged, deployed policy) and `final_model.pt` (the last iterate).
- **Resume:** `checkpoints/resume_state.pt` (git-ignored) is written every 500 epochs and on Ctrl+C;
  `train.py --resume <run>` continues from it. The environment state is resampled, so a resumed run is not
  bit-identical.
- **`solve()` path:** a single `solve()` call (≤ 32 rows) runs fully on the CPU with one torch thread.
  Tiny tensors with many threads are ~5× slower, and GPU launches add latency.

## Results

One seed per robot (seed 1), full budget. Runs: `runs/policyik/<robot>/20261107_230000/`; SAC+DLS runs:
`runs/sac+dls/<robot>/`.

**PolicyIK's hold protocol** (`eval.py --protocol hold --dls --refine`; 2000 tasks, seed 7; saved as
`outputs/hold_compare.json`):
- **strict**: the mean errors over the final second (30 of 160 steps) are < 1 mm and < 0.2°.
- **coarse**: the same test at < 1 cm and < 2°.
- **deployment**: plus the error-corrected goal and up to 8 transient-goal attempts; E[N] is the mean number
  of attempts.
- **+ DLS refine**: `dls_refine` of the single attempt's final configuration, success at 0.1 mm / 0.001°.

| Robot | Method | Strict | Coarse | Median pos / ori | Deployment strict (E[N]) | + DLS refine, 0.1 mm |
|---|---|---|---|---|---|---|
| Franka | PolicyIK | **95.2%** | **97.0%** | 0.146 mm / 0.031° | 98.65% (1.19) | **96.15%** |
| | SAC+DLS | 86.8% | 94.4% | 0.396 mm / 0.079° | **98.85%** (1.34) | 94.35% |
| | DLS baseline | 51.4% | 53.7% | 0.158 mm / 0.006° | 79.25% (3.58) | 55.05% |
| UR5e | PolicyIK | **87.4%** | **89.8%** | 0.159 mm / 0.036° | 93.40% (1.70) | **91.85%** |
| | SAC+DLS | 79.6% | 87.6% | 0.489 mm / 0.083° | **97.25%** (1.59) | 90.05% |
| | DLS baseline | 62.5% | 64.7% | 0.000 mm / 0.000° | 87.50% (2.72) | 71.15% |
| SO-ARM101 | PolicyIK | **98.9%** | **99.2%** | 0.048 mm / 0.011° | 99.85% (1.04) | **96.00%** |
| | SAC+DLS | 97.6% | 98.7% | 0.086 mm / 0.030° | 99.85% (1.05) | 94.75% |
| | DLS baseline | 67.0% | 67.9% | 0.000 mm / 0.000° | 87.80% (2.66) | 66.45% |

The DLS baseline is a reference controller, not a trained method:
- kinematic damped least squares (λ = 0.05, at most 0.0625 rad per step) under the same step rules;
- its 51% on the Franka matches the paper's simulated DLS (50.4%);
- when it converges it is exact (median 0 mm), but about half the time it stalls in local minima or at
  joint limits.

**This repository's benchmark** (`eval.py`: 500 targets, stop at 2 mm / 0.02 rad, at most 120 steps; SAC+DLS
numbers from its [README](../sac_dls/README.md)):
- Solve times are for single `solve()` calls on a 4090 shared with other jobs.
- SAC+DLS under the same load: Franka 17.5 / 10.6 ms, UR5e 27.3 / 14.8 ms, SO-101 9.5 / 8.3 ms for
  1 / 4 starts. Its README values were measured on an idle GPU.

| Robot | Starts | Refine | PolicyIK 2 mm | SAC+DLS 2 mm | PolicyIK 0.1 mm | SAC+DLS 0.1 mm | PolicyIK steps | PolicyIK time |
|---|---|---|---|---|---|---|---|---|
| Franka | 1 | no | **97.6%** | 93.8% | | | 21.1 | 11.7 ms |
| Franka | 1 | yes | **98.4%** | 96.4% | **96.2%** | 92.0% | 21.1 + 3.4 | 12.6 ms |
| Franka | 4 | no | **99.6%** | 99.2% | | | 16.1 | 11.6 ms |
| Franka | 4 | yes | 99.8% | 99.4% | 98.8% | 98.8% | 16.1 + 3.5 | 11.5 ms |
| UR5e | 1 | no | **87.8%** | 85.6% | | | 23.3 | 17.0 ms |
| UR5e | 1 | yes | **93.0%** | 90.6% | **92.2%** | 90.4% | 23.3 + 5.1 | 16.5 ms |
| UR5e | 4 | no | 96.6% | **99.2%** | | | 18.9 | 11.8 ms |
| UR5e | 4 | yes | 97.4% | **99.6%** | 97.0% | **99.6%** | 18.9 + 4.9 | 11.9 ms |
| SO-ARM101 | 1 | no | 98.8% | 99.0% | | | 12.0 | 6.7 ms |
| SO-ARM101 | 1 | yes | 99.6% | 99.0% | 95.6% | 96.0% | 12.0 + 1.9 | 7.2 ms |
| SO-ARM101 | 4 | no | 99.4% | 99.8% | | | 9.7 | 6.5 ms |
| SO-ARM101 | 4 | yes | 99.6% | 99.8% | 97.0% | **98.2%** | 9.7 + 2.3 | 6.9 ms |

**Takeaways**
- **Single attempt:** PolicyIK is more precise and holds better: +8.4 (Franka), +7.8 (UR5e) and +1.3
  (SO-101) points strict, at a 1.8–3× lower median error. Its tolerance-normalized progress / dwell rewards
  target exactly this criterion, while SAC+DLS's reward targets 2 mm. It also wins this repo's single-start
  benchmark on the Franka and UR5e.
- **With retries** (4 starts, or the deployment protocol), SAC+DLS catches up, and is better on the UR5e
  (99.2% vs 96.6%).
- **UR5e failures depend on the target, not the start:** 8 random restarts rescue 76% of PolicyIK's
  single-attempt failures, vs 96% for SAC+DLS.
  - The ~3% unsolved targets lie mostly at the edge of the reach: radius > 0.75 m for 20% of them vs 8% of
    all targets, and lower heights.
  - They are not at joint limits: 2% have a joint within 0.05 rad of a limit.
- **Over SAC+DLS vs over DLS:** PolicyIK's single-attempt gain over SAC+DLS (+8 points) is far smaller than
  over DLS (+44 points on the Franka); in the kinematic setting SAC+DLS is already a strong baseline. The
  paper's largest effect (deviation 2) cannot show up without dynamics.

**Checks**
- *CPU vs Warp backend:* the same checkpoint gives identical hold metrics on both backends (one borderline
  task of 500 differs on the SO-101). Warp FK matches MuJoCo CPU to 4e-7 m, with 0 collision disagreements
  on 5000 random configurations per robot. Training curves agree on both backends (Franka epoch 1000: 87.6%
  CPU vs 87.2% Warp).
- *Torch core vs the first numpy implementation:* rewards agree to 3.5e-5. Observations agree except on 7 of
  120k samples at a rotation error of ~180°, where the axis-angle direction is ill-conditioned in float32.

## Files

| File | Contents |
|---|---|
| `config.py` | `PolicyIKConfig` (env / observation / reward; paper values noted), `PPOConfig` |
| `core.py` | `PolicyIKBatch` (torch): action, observation, ρ / running-best state and reward; used by training and inference |
| `env.py` | `PolicyIKVecEnv`: pools, 12 s episodes, goal resampling every 4 s, time-outs |
| `ppo.py` | actor-critic, running normalizers, PPO update, checkpoint averaging |
| `controller.py` | `PolicyIKController`: the policy as a batched closed-loop controller |
| `solver.py` | `PolicyIKSolver` (`solve`, `solve_batch`, `num_restarts`, `refine`) |
| `train.py` | training loop (`train.py --method policyik`) |

Shared code used here:
- `core/batch_kinematics.py`: Warp / CPU batched FK and collision checks, pool sampling;
- `core/controller.py`: controller interface, `run_best` (solver rollout);
- `core/solution.py`: `IKSolution`, start configurations, refinement of branches;
- `core/refine.py`: `dls_refine`;
- `evaluation/hold.py`, `evaluation/rollout.py`: hold and deployment protocol (in-training evaluation, `eval.py --protocol hold`);
- `utils/config.py`: `--set` overrides.
