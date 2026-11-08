# ik-rl

Learned inverse kinematics for robot arms in MuJoCo with reinforcement learning: given a target end-effector position and orientation, a trained policy iteratively moves the joints until the pose is reached. Two methods are implemented side by side:

| Method | `--method` | Idea | Details |
|---|---|---|---|
| SAC+DLS | `sac+dls` | SAC policy to ~1 mm, damped least squares refines to 0.1 mm | [policy/sac_dls/README.md](policy/sac_dls/README.md) |
| PolicyIK | `policyik` | PPO with tolerance-normalized progress / dwell rewards (after Liu et al., arXiv:2611.05372), reaches and holds 1 mm / 0.2° | [policy/policyik/README.md](policy/policyik/README.md) |

| Robot | `--robot` | Config |
|---|---|---|
| Franka Emika Panda (7-DoF) | `franka` | [configs/franka.yaml](configs/franka.yaml) |
| Universal Robots UR5e (6-DoF) | `ur5e` | [configs/ur5e.yaml](configs/ur5e.yaml) |
| The Robot Studio SO-ARM101 (5-DoF) | `so101` | [configs/so101.yaml](configs/so101.yaml) |

Robots, kinematics and DLS refinement are shared ([`core/`](core/README.md), with helpers in `utils/`), the evaluation protocols live in [`evaluation/`](evaluation/README.md), and each method in [`policy/<method>/`](policy). A robot is just a YAML config (model, joints, TCP, home pose, sampling ranges).

```
core/             robots, kinematics (CPU / MuJoCo Warp), DLS refinement, solver building blocks
utils/            config / overrides, runs lookup, rendering, rotation math
evaluation/       benchmark (2 mm early stop) and hold (reach and hold 1 mm / 0.2 deg) protocols, DLS baseline,
                  interactive viewer (viewer/: multi-start tracker, native window / browser)
policy/sac_dls/   SAC+DLS: env, SAC, solver, training
policy/policyik/  PolicyIK: PPO core / env, solver, training
train.py          train.py --method <method> --robot <robot>
eval.py           eval.py --protocol {benchmark,hold}: any checkpoints (method / robot read from them)
visualize.py      visualize.py --method <method> --robot <robot>: interactive viewer of any checkpoint
runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/
```

## Installation

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/Char1esFu/ik-rl.git
cd ik-rl
uv venv && uv pip install -r pyproject.toml
```

Requires MuJoCo >= 4.0 and MuJoCo Warp (GPU kinematics for PolicyIK; installed by the command above).

> The `mujoco_menagerie` submodule (robot models) is ~1.9 GB. To fetch only the models you need instead, see *Sparse checkout* below.

### Sparse checkout (optional)

Clone without `--recurse-submodules`, then populate the submodule with only the required robot models:

```bash
git clone https://github.com/Char1esFu/ik-rl.git
cd ik-rl
git submodule init
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie
git -C mujoco_menagerie sparse-checkout set franka_emika_panda universal_robots_ur5e robotstudio_so101   # ~80 MB
git submodule update   # check out the menagerie commit pinned by this repo
```

Then install the dependencies with `uv venv && uv pip install -r pyproject.toml` as above.

## Usage

```bash
.venv/bin/python train.py --method sac+dls --robot ur5e    # writes runs/sac+dls/ur5e/<YYYYmmdd_HHMMSS>/
.venv/bin/python train.py --method policyik --robot ur5e   # writes runs/policyik/ur5e/<YYYYmmdd_HHMMSS>/
.venv/bin/python eval.py --method policyik --robot ur5e --restarts 4 [--refine]   # benchmark on random reachable targets
.venv/bin/python eval.py --protocol hold --robot ur5e --dls --refine   # reach and hold: newest run of each method + DLS
.venv/bin/python visualize.py --method policyik --robot ur5e   # interactive 3D viewer (web UI at http://localhost:8080 on headless machines)
```

- `train.py` needs `--method` (`sac+dls` or `policyik`) and takes `--robot` (a name from `configs/<name>.yaml` or a YAML path; default `franka`); all other options belong to the method (`train.py --method <method> --help`).
- Each training run writes `config.json`, `checkpoints/` and `outputs/` (curves, eval results) to `runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/`; use `--run-name` to choose the folder name. Ctrl+C stops cleanly; continue a run with `train.py --resume runs/<method>/<robot>/<run>` (method and robot are read from the run).
- `eval.py` / `visualize.py` take `--checkpoint` to pick a specific model (method and robot are read from it); otherwise `--method` / `--robot` select the newest run to use.
- Method settings have their defaults in `policy/<method>/config.py`. For experiments, override them without editing code on the command line (`--set SECTION.KEY=VALUE`) or, for SAC+DLS, from a YAML file that lists only the changed values (see [`configs/exp/example.yaml`](configs/exp/example.yaml)):
  ```bash
  .venv/bin/python train.py --method sac+dls --exp configs/exp/example.yaml
  .venv/bin/python train.py --method sac+dls --robot ur5e --set env.smooth_penalty=0.06 sac.gamma=0.97
  .venv/bin/python train.py --method policyik --robot ur5e --set ppo.entropy_coef=0.005
  ```
  Precedence: defaults < robot `env_overrides` < `--exp` < `--set` / command-line flags. The resolved values are saved in the run's `config.json` and checkpoints.
- Run any script with `--help` for all options.

## Results in Brief

Hold protocol (2000 tasks; reach and hold 1 mm / 0.2° over the final second of 160 steps), single attempt: PolicyIK 95.2 / 87.4 / 98.9% vs SAC+DLS 86.8 / 79.6 / 97.6% (Franka / UR5e / SO-101). With 4 restarts + DLS refinement both reach 97-99.6% at 0.1 mm / 0.001° on this repo's benchmark, SAC+DLS ahead on the UR5e. Full tables: [SAC+DLS](policy/sac_dls/README.md#results), [PolicyIK](policy/policyik/README.md#results).

## Python API

```python
import numpy as np
from policy import load_solver
from utils.runs import find_latest_checkpoint

solver = load_solver(find_latest_checkpoint("ur5e", method="sac+dls"))  # SACIKSolver / PolicyIKSolver; robot from the checkpoint
sol = solver.solve(
    target_pos=np.array([0.45, 0.15, 0.48]),
    target_quat=np.array([0.0, 1.0, 0.0, 0.0]),  # [w, x, y, z], or target_rot=<3x3 matrix>
    q_init=None,       # optional start configuration (n,)
    num_restarts=4,    # parallel start configurations; the best result is returned
    refine=True,       # optional DLS refinement after the policy: ~0.1 mm / 0.001 deg instead of ~1 mm
)
print(sol.qpos, sol.pos_err_m, sol.rot_err_rad, sol.success)
```

`solve()` returns an `IKSolution` with `qpos`, `ee_pos`, `ee_quat`, `pos_err_m`, `rot_err_rad`, `steps`, `success` and `refine_iters`; `solve_batch()` solves many targets in one GPU batch. Use `eval.py --refine` to benchmark with refinement.

## Adding a Robot

1. Make sure the model is available (e.g. `git -C mujoco_menagerie sparse-checkout add <robot_dir>`).
2. Copy [`configs/ur5e.yaml`](configs/ur5e.yaml) to `configs/<name>.yaml` and adapt it (scene, arm joints, TCP, home pose, joint sampling box, workspace; optionally camera and viewer figure-8, see [`configs/so101.yaml`](configs/so101.yaml)).
3. `train.py --method <method> --robot <name>`.
