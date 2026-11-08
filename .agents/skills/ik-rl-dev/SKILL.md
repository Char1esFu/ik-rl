---
name: ik-rl-dev
description: >-
  Architecture, conventions and verification workflow for the ik-rl repository (learned 6D IK for
  robot arms in MuJoCo; methods SAC+DLS and PolicyIK side by side in policy/, shared code in core/,
  utils/ and evaluation/).
  Use this skill when modifying ik-rl: adding a robot or a method, changing the observation / reward /
  environment, refactoring shared code, training or comparing models, or checking that a change is
  behavior-preserving.
---

# ik-rl development

## Architecture

```
configs/<robot>.yaml   # everything robot-specific (RobotSpec); no Python per robot
configs/exp/*.yaml     # SAC+DLS experiment overrides (env: / sac: / train: sections), train.py --exp
utils/                 # generic helpers, no robot / method knowledge; imports nothing else at runtime
  config.py            # _ConfigMixin + --exp / --set override helpers (sections given by each method)
  runs.py render.py math_utils.py
core/                  # domain core; imports core + utils only
  robot.py             # RobotSpec + get_robot(name | yaml path), available_robots()
  kinematics.py        # build_ik_model, RobotKinematics (fk, apply_joint_step, sample_target / sample_start)
  batch_kinematics.py  # BatchKinematics: batched FK + collision on MuJoCo Warp (GPU) or CPU workers
  refine.py            # dls_refine (float64 DLS; the "DLS" of SAC+DLS, refine=True of every solver)
  solution.py          # IKSolution, checkpoint_robot, start_configs, parse_target_rot, refine_branches
  controller.py        # Controller / KinematicController interface, run_best (solver rollout for any controller)
evaluation/            # method-independent protocols; imports core + utils (never policy)
  benchmark.py         # EvalTargets, evaluate_solver: this repo's benchmark (stop at 2 mm / 0.02 rad)
  hold.py              # HoldTasks, evaluate_hold, refine_final: PolicyIK's hold protocol + deployment protocol
  rollout.py baselines.py   # run_horizon (fixed horizon); DLSController (baseline, not a trained method)
  viewer/              # tracker.py (MultiStartTracker on any Controller), app.py (native / web viewer),
                       #   viewer.html (__ROBOT_NAME__ / __METHOD__ / __{X,Y,Z}_{MIN,MAX}__ filled at startup)
policy/                # learning methods side by side; METHODS / load_solver / checkpoint_method in __init__
  sac_dls/             # method "sac+dls": config (IKConfig, SACConfig), obs, env (IKEnv, VectorIKEnv),
                       #   sac, replay_buffer, rollout (rollout_groups), solver (SACIKSolver), evaluation
                       #   (evaluate_parked), controller (SACController), train, README.md (method + results)
  policyik/            # method "policyik": config, core (PolicyIKBatch, torch), env, ppo, controller, solver,
                       #   train, README.md (training + deviations from the paper + results)
train.py               # train.py --method {sac+dls,policyik} --robot <name|yaml> [method options]
eval.py                # eval.py --protocol {benchmark,hold}: any checkpoints (method read from them)
visualize.py           # visualize.py --method / --robot / --checkpoint: viewer for any method
runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/   # method dir names: "sac+dls", "policyik"
```

- Layering (keep it acyclic): `utils` <- `core` <- `evaluation` <- `policy` <- scripts. `evaluation` takes a
  solver or a controller and never imports `policy`; `policy` uses `evaluation` for in-training evaluation;
  only scripts (`train.py`, `eval.py`, `visualize.py`) combine `policy.load_solver` with the protocols.
- The hold protocol and the viewer step every method through `core.controller.Controller` (`policy.make_controller`;
  `cpu=True` for the viewer's few rows). A method's controller implements `reset` / `set_goal` / `step(smooth,
  scale)` (a = scale * (smooth * a_policy + (1 - smooth) * a_prev)) / `activity()` and sets `horizon` (viewer step
  budget: SAC+DLS 65 = its episode length, PolicyIK 120). `activity()` decides "settled" (< 0.08) and parking
  (< 0.06): SAC+DLS max |a|; PolicyIK the executed step / eta (its raw action stays large near the goal because the
  gain is ~0.012 there). The viewer was checked headless (tracker on 40 targets per method / robot, web viewer
  over WebSocket); the native GLFW window needs a display.

- Python package `policy/sac_dls` vs method name / runs dir `sac+dls` (`+` is not valid in a module name).
  Register a new method in `policy.METHODS`; its package needs `train.py` with `main(argv)`, a solver with
  `solve` / `solve_batch` / `cfg.term_* / eval_*` tolerances, and checkpoints that store `"method"`
  (`checkpoint_method` treats checkpoints without it as `sac+dls`). Checkpoints hold only tensors and plain
  types (`torch.load(..., weights_only=True)` works), so moving modules never breaks them.
- Robot-specific values live **only** in YAML (scene, arm joints, fixed joints, TCP body/offset/quat,
  home, joint sampling box, workspace, camera, viewer figure-8 `fig8_*`). Paths are relative to the repo root. Never hard-code joint
  counts, `qpos[:7]`, finger joints or workspace numbers in shared code; use `RobotKinematics`
  (`arm_qadr`, `arm_dadr`, `q_min/q_max`, `home_q`, `sample_box`, `in_workspace`, `valid_start`, `fk`).
- Targets and starts of every method come from `RobotKinematics.sample_target` / `sample_start` (`IKEnv`
  delegates to them with the same RNG consumption), so `EvalTargets.sample(robot, n, rng)` gives identical
  targets for every method and for results recorded before the refactor.
- **Method choices belong in `policy/<method>/config.py`, not robot YAML.** SAC+DLS: `IKConfig` /
  `SACConfig` (e.g. reward weights, `obs_features`); experiments override them via `train.py --method sac+dls
  --exp configs/exp/<name>.yaml` and/or `--set env.<field>=v sac.<field>=v` (precedence: defaults < robot
  `env_overrides` < `--exp` < `--set` / CLI flags). PolicyIK: `PolicyIKConfig` / `PPOConfig` via `--set env.* ppo.*`.
  Changing a default changes the baseline (and the fingerprints below); prefer overrides for trials.
  `sac.obs_dim/act_dim/batch_size` are derived by the SAC trainer and cannot be overridden.
- SAC+DLS observation: feature groups in `OBS_FEATURES` (`policy/sac_dls/obs.py`, `name: (dims per joint,
  fixed dims)`); a code schema, not a config value. `obs_dim()` is derived from it, never hand-written. To add
  a feature: add a row to `OBS_FEATURES` and a branch in `compute_obs_batch`. Pass `cfg.obs_features` at every
  `compute_obs_batch` call (env x3, rollout, SACController).
- One kinematic step (q += dq, clip, FK + collision, colliding rows stay put) exists once per backend:
  `RobotKinematics.apply_joint_step` (SAC envs, rollout) and `KinematicController._move` /
  `PolicyIKBatch.step` on `BatchKinematics` (same semantics). The SAC policy loop exists once:
  `policy/sac_dls/rollout.rollout_groups`; `core.controller.run_best` is its method-independent twin.
- **Benchmark protocol** (`evaluation/benchmark.py`): fixed targets + starts, a policy's solver, stop at
  2 mm / 0.02 rad, max 120 steps. SAC training (1 start, 500 targets, curves / best model / early stop),
  eval.py (1 and 4 starts) and compare_models.py use it. Batched solving matches one-at-a-time `solve()` in
  success (TF32 matmuls make actions differ ~1e-3 with batch size, so failing trajectories are not
  bit-identical).
- High precision comes from **DLS refinement** (`core/refine.dls_refine` via `core/solution.refine_branches`,
  `solve(..., refine=True)`, `eval.py --refine`): float64 kinematics, step halving, joint limits, final
  collision check; refines the best pose, then the other branches' final configurations if needed (SO-101
  policy sometimes lands on the mirrored pan/wrist + pi configuration, ~0.3 mm off).
- Checkpoint meta stores `method`, `robot`, `robot_spec` and `env_config`. `checkpoint_robot(meta)` /
  `<Config>.from_dict(meta["env_config"])` rebuild the exact setup, so solver / eval / viewer must load the
  robot and config **from the checkpoint**, not from defaults.

## PolicyIK (`policy/policyik/`)

- PPO after PolicyIK (arXiv:2611.05372); `policy/policyik/README.md` lists every deviation from the paper.
  `PolicyIKBatch` (`core.py`, torch) is the single implementation of the step / obs / reward (training env,
  controller, solver).
- Kinematics backends (`core/batch_kinematics.BatchKinematics`): `warp` (default on CUDA; MuJoCo Warp, CUDA
  graph per world-count bucket, torch tensors stay on the GPU; <= 32 rows go to the CPU) and `cpu` (worker
  processes + shared memory; MuJoCo holds the GIL, so threads do not scale; scripts need an
  `if __name__ == "__main__"` guard because of spawn). Warp matches `RobotKinematics.fk` to float32 precision
  with identical collision flags; its Jacobian is built from `xanchor` / `xaxis` (hinge joints). GPU memory
  grows with the world count (~0.8 GB per 4096 Franka worlds, ~3 GB for 16384), so buckets are reused; a
  training process needs ~1.7 GB. Check free VRAM before launching (other jobs share the GPU).
- Evaluation metrics stay numpy float64. CPU and Warp backends give the same hold-protocol results for the
  same checkpoint (one borderline task of 500 may flip).
- Training: 13,000 epochs x 4096 envs x 24 steps (1.28e9 transitions): ~600k steps/s with Warp (~40 min per
  robot alone, ~1 h with three in parallel) vs ~90k with 8-16 CPU workers.
- Gain schedule: PolicyIK's (beta_min 0.1, ramp to 1 at 9.3 mm / 2.3 deg) relies on PD filtering of the
  exploration noise; kinematically, Franka / UR5e never enter the tolerance during training (stall at 7-10 mm,
  0% strict). Defaults use 0.012 / 60 mm / 0.3 rad (config.py; README deviation 3).
- Single `solve()` of PolicyIK runs fully on the CPU with one torch thread (multi-threaded tiny ops: 5x slower).
- `eval.py --protocol hold`: hold protocol for the newest checkpoint of each method + `--dls` baseline on `HoldTasks.sample`
  (EvalTargets, seed 7, 2000 tasks, same targets as compare_models.py).

## Adding a robot

1. `git -C mujoco_menagerie sparse-checkout add <robot_dir>` (submodule is sparse; pinned commit unchanged).
2. Copy `configs/ur5e.yaml` to `configs/<name>.yaml`; fill in joints, TCP, home (preferably facing +x
   like the existing robots so the viewer sliders / camera work), sampling box, workspace, and for a
   robot of a different size `camera_*` and `fig8_*` (the defaults are Franka-sized). Make the TCP z-axis
   the approach direction (`ee_quat`; e.g. SO-101's menagerie site uses x), which Point Down / figure-8 assume.
   Fewer than 6 joints (SO-101, 5-DoF): only poses whose approach lies in the arm's vertical plane are
   reachable; training targets are FK samples so that is fine, but the figure-8 needs
   `fig8_radial_yaw: true`, no roll amplitude, and the pan axis is not necessarily at the origin.
   Check a candidate figure-8 with a numerical (damped least squares) IK before training.
3. Sampling box = physical joint limits minus 0.25 rad ("soft limits") for every joint; workspace box =
   whole reach above the floor; `start_pos_min: [-.inf, -.inf, z]`. Analyse the reachable workspace first
   (target share by azimuth around the base-yaw axis, x < 0 reach, height / radius behind the robot) and
   document it in the YAML; e.g. SO-101 reaches behind only by tipping the upper arm over the top.
   Sampling up to the hard limits hurt accuracy (Franka test: targets with a joint < 0.1 rad from a limit
   73% vs 96% single-start); the base seam itself (Franka +-166 deg) needs no special handling.
   Sanity-check (one-off script): home pose collision-free and inside the workspace; TCP equals any
   reference site; Jacobian matches finite differences; acceptance rate (franka ~89%, so101 ~65%,
   ur5e ~40% because the +-2pi shoulder lift often hits the floor; rejected samples only cost time).
4. Train each method (long-running, see below), then `eval.py --method <m> --robot <name> --restarts 1` and
   `--restarts 4` and `eval.py --protocol hold --robot <name> --dls --refine`; put results in `policy/<method>/README.md`,
   add the robot to the root README table and sparse-checkout line.

## Verifying a change

- **Behavior-preserving refactor**: record [`scripts/fingerprint.py`](scripts/fingerprint.py) (SAC+DLS env) for
  every robot before the change and compare after (must be identical). Re-run `eval.py --method <m> --robot <r>
  --restarts 1|4 [--refine] --out-dir <scratch>` for both methods and diff against the run's
  `outputs/eval_summary_*.json` (all but timings), and `eval.py --protocol hold --robot <r> --dls --refine --out <scratch>`
  against `runs/policyik/<r>/<run>/outputs/hold_compare.json`. To compare with the code before a change,
  `git worktree add <scratch> HEAD` and symlink `mujoco_menagerie/*` into it.
- **Change that alters behavior** (obs / reward / env): existing checkpoints become stale; retrain.
  Compare models with [`scripts/compare_models.py`](scripts/compare_models.py) (2000 fixed targets,
  single/multi-start + 65-step parked accuracy). Use >= 2 seeds per variant: single-run differences of
  1-2 points are within seed noise.
- Jacobian sanity: `RobotKinematics.fk` must call `mj_comPos` before `mj_jacSite`; without it the
  Jacobian is silently all zeros. Check against finite differences or an `mj_forward` path after
  touching FK code.

## Training and runs

- SAC+DLS: default budget 5M env steps (~29 min for Franka alone on an RTX 4090, slower in parallel) with early
  stopping: mean solver SR(2mm) of the last 5 evals >= 98.5% (`--early-stop-sr/--early-stop-window`) or no
  new best model for 15 evals (`--plateau-patience`); both computed from the history, so they survive
  --resume. Evals use 500 fixed targets (less noise than fresh random targets each time).
- Resume: `train.py --resume runs/<method>/<robot>/<run> [--total-steps N | --epochs N]` (method read from the run) loads `checkpoints/resume_state.pt`
  (written every `--resume-interval` 1M steps, at the end, on early stop and on Ctrl+C / SIGTERM, which
  finish the current step first). Verified exact: interrupt + resume == uninterrupted, bit for bit.
  Raising `--total-steps` on resume stretches the cosine LR schedule. `resume_state.pt` (~1.5-2 GB) is
  git-ignored; `best_model.pt` / `latest_model.pt` stay small and are what solver / eval / viewer load.
  `config.json` records `resumes` and `result.stop_reason` (completed / early_stop / interrupted).
- Launch training as a background / daemon task with absolute paths (a launch from another working
  directory failed with exit 127), e.g. `R=/home/congfu/proj/ik-rl; D=$R/runs/<method>/<robot>/$(date +%Y%m%d_%H%M%S);
  mkdir -p $D; cd $R; .venv/bin/python train.py --method <method> --robot <robot> --run-name $(basename $D) > $D/train.log 2>&1`.
  Wait for free VRAM first when the GPU is shared (`until [ $(nvidia-smi --query-gpu=memory.free
  --format=csv,noheader,nounits) -ge 3000 ]; do sleep 30; done`). Plain `nohup ... &` inside a normal
  command gets killed when that command finishes.
- `eval.py` defaults to the newest `runs/<method>/<robot>/*/checkpoints/best_model.pt`
  (by mtime; same for `visualize.py`) and write eval output into that run's `outputs/`; pass `--out-dir` for scratch evaluations.
- Keep only the latest (best) run per method and robot in `runs/<method>/<robot>/` (the user prefers this);
  runs are committed to git (`resume_state.pt` is git-ignored).

## Current reference values

- Default-config env fingerprints (MuJoCo 4.0): franka `ef382c0ae8019d8c`, ur5e `d75bc2421a73a1ab`, so101
  `b26d03e8a6b635a2` (soft-limit all-around sampling; UR5e one turn). MuJoCo 3.17 gave franka `3a4daf3dd0402c72`,
  ur5e `c3cff3c0f15473c2`: FK / collisions are identical (20k configs), only ~1e-16 Jacobian entries differ;
  eval.py results of all SAC checkpoints are unchanged.
- Obs dims (all features): `11n + 35` -> franka 112, ur5e 101, so101 90; without `jacobian` + `jt_direction`: `4n + 35`.
- SAC+DLS results (eval.py, 500 all-around targets, 2 mm / 1.15 deg, seed 42, 5M budget + early stop):
  franka 93.8% (1 start) / 99.2% (4 starts); ur5e 85.6% / 99.2%; so101 99.0% / 99.8% (shared reward,
  no env_overrides). Front-only workspace (earlier) was franka 98.2 / 100, ur5e 91.2 / 99.4, so101 99.8 / 100.
  UR5e failure analysis: not seam / branch switching; wound-up starts (|q| near 2pi, pinned at limits) and
  collision-blocked paths -> sample one turn. Training is deterministic: same seed + same code reproduces a run bit for bit
  (verified on ur5e), so a seed-42 retrain is the strongest check that a refactor preserved training. Jacobian features: +0.6 pt single-start, ~1 fewer step (kept).
- PolicyIK results (runs `runs/policyik/<robot>/20261107_230000`, seed 1, full budget): hold protocol
  (eval.py --protocol hold, 2000 tasks) single-attempt strict franka 95.2 / ur5e 87.4 / so101 98.9% (SAC+DLS 86.8 /
  79.6 / 97.6, DLS baseline 51.4 / 62.5 / 67.0); eval.py 1 start 2 mm 97.6 / 87.8 / 98.8%, 4 starts
  99.6 / 96.6 / 99.4%.
- MuJoCo 4.0 vs 3.17: SAC+DLS so101 `--refine` mean DLS iterations 1.95 / 2.44 (4.0) vs 1.93 / 2.49 recorded
  with 3.17 (r1 / r4); all success rates and errors identical.
