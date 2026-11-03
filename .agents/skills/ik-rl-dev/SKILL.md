---
name: ik-rl-dev
description: >-
  Architecture, conventions and verification workflow for the ik-rl repository (learned 6D IK for
  robot arms in MuJoCo with SAC). Use this skill when modifying ik-rl: adding a robot, changing the
  observation / reward / environment, refactoring shared code in ik_rl/, training or comparing
  models, or checking that a change is behavior-preserving.
---

# ik-rl development

## Architecture

```
configs/<robot>.yaml   # everything robot-specific (RobotSpec); no Python per robot
configs/exp/*.yaml     # experiment overrides (env: / sac: / train: sections), train.py --exp
ik_rl/                 # shared, robot-agnostic code
  robot.py             # RobotSpec + get_robot(name | yaml path), available_robots()
  config.py            # IKConfig, SACConfig (all method defaults) + --exp / --set override helpers
  obs.py               # OBS_FEATURES, obs_dim, compute_obs_batch
  env.py               # RobotKinematics, IKEnv, VectorIKEnv, action scaling, reward
  sac.py replay_buffer.py solver.py (SACIKSolver) math_utils.py runs.py
  README.md            # method details (formulas) + per-robot results
train.py eval.py visualize.py   # --robot <name|yaml>; runs/<robot>/<YYYYmmdd_HHMMSS>/
web/viewer.html        # web viewer page (HTML/CSS/JS) served by visualize.py; __PLACEHOLDERS__ filled at startup
```

- Robot-specific values live **only** in YAML (scene, arm joints, fixed joints, TCP body/offset/quat,
  home, joint sampling box, workspace, camera, viewer figure-8 `fig8_*`). Paths are relative to the repo root. Never hard-code joint
  counts, `qpos[:7]`, finger joints or workspace numbers in shared code; use `RobotKinematics`
  (`arm_qadr`, `arm_dadr`, `q_min/q_max`, `home_q`, `sample_box`, `in_workspace`, `valid_start`, `fk`).
- **Method choices belong in `IKConfig` / `SACConfig` (`ik_rl/config.py`), not robot YAML** (e.g. reward
  weights, `obs_features`). The dataclass defaults are the single source of truth; experiments override
  them via `train.py --exp configs/exp/<name>.yaml` and/or `--set env.<field>=v sac.<field>=v`
  (precedence: defaults < robot `env_overrides` < `--exp` < `--set` / CLI flags). Changing a default
  changes the baseline (and the fingerprints below); prefer an experiment YAML for trials.
  `sac.obs_dim/act_dim/batch_size` are derived by train.py and cannot be overridden.
- The observation is built from feature groups in `OBS_FEATURES` (`ik_rl/obs.py`, `name: (dims per joint,
  fixed dims)`); it is a code schema, not a config value. `obs_dim()` is derived from it, never
  hand-written. To add a feature: add a row to `OBS_FEATURES` and a branch in `compute_obs_batch`. Pass `cfg.obs_features` at every `compute_obs_batch` call
  (env x3, solver, `visualize.py` tracker). `train.py --obs-features ...` selects groups.
- Checkpoint meta stores `robot`, `robot_spec` and `env_config` (incl. `obs_features`).
  `checkpoint_robot(meta)` / `IKConfig.from_dict(meta["env_config"])` rebuild the exact setup, so
  solver / eval / viewer must load the robot and config **from the checkpoint**, not from defaults.

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
3. Sanity-check before training (one-off script): home pose collision-free; TCP equals any reference
   site; Jacobian matches finite differences; target acceptance rate of the sampling box is reasonable
   (UR5e ~17%, SO-101 ~47%); collision rate not dominated by floor hits (tighten the shoulder-lift range
   if so; on SO-101 the floor still causes most of the remaining ~13% collisions, which is acceptable).
4. Train (long-running, see below), then `eval.py --robot <name> --restarts 1` and `--restarts 4`;
   put results in `ik_rl/README.md`, add the robot to the root README table and sparse-checkout line.

## Verifying a change

- **Behavior-preserving refactor**: record [`scripts/fingerprint.py`](scripts/fingerprint.py) for every
  robot before the change and compare after (must be identical). Also re-run `eval.py --restarts 4
  --num-tests 200 --out-dir <scratch>` on existing checkpoints and diff the numbers.
- **Change that alters behavior** (obs / reward / env): existing checkpoints become stale; retrain.
  Compare models with [`scripts/compare_models.py`](scripts/compare_models.py) (2000 fixed targets,
  single/multi-start + 65-step parked accuracy). Use >= 2 seeds per variant: single-run differences of
  1-2 points are within seed noise.
- Jacobian sanity: `RobotKinematics.fk` must call `mj_comPos` before `mj_jacSite`; without it the
  Jacobian is silently all zeros. Check against finite differences or an `mj_forward` path after
  touching FK code.

## Training and runs

- One run (1.6M steps) takes ~12 min on an RTX 4090 (slower when several run in parallel).
- Launch training as a **daemon task** (`IsDaemon: true`), e.g.
  `TS=$(date +%Y%m%d_%H%M%S); mkdir -p runs/<robot>/$TS; exec .venv/bin/python train.py --robot <robot> --run-name $TS > runs/<robot>/$TS/train.log 2>&1`.
  Plain `nohup ... &` inside a normal command gets killed when that command finishes.
- `eval.py` / `visualize.py` default to the newest `runs/<robot>/*/checkpoints/best_model.pt`
  (by mtime) and write eval output into that run's `outputs/`; pass `--out-dir` for scratch evaluations.
- Keep only the latest run per robot in `runs/` (the user prefers this); runs are committed to git.

## Current reference values

- Default-config env fingerprints: franka `943284517ee7d36c`, ur5e `65ec4caf608c5c4a`, so101 `5dbd1cd1d2fcbb97`.
- Obs dims (all features): `11n + 35` -> franka 112, ur5e 101, so101 90; without `jacobian` + `jt_direction`: `4n + 35`.
- Results (eval.py, 500 targets, 2 mm / 1.15 deg, seed 42 runs): franka 98.2% (1 start) / 100% (4 starts);
  ur5e 91.2% / 99.4%; so101 99.8% / 100% (shared reward, no env_overrides). Training is deterministic: same seed + same code reproduces a run bit for bit
  (verified on ur5e), so a seed-42 retrain is the strongest check that a refactor preserved training. Jacobian features: +0.6 pt single-start, ~1 fewer step (kept).

