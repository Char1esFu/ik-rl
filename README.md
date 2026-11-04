# ik-rl

Learned inverse kinematics for robot arms in MuJoCo with reinforcement learning: given a target end-effector position and orientation, a trained policy iteratively moves the joints until the pose is reached.

| Robot | `--robot` | Config |
|---|---|---|
| Franka Emika Panda (7-DoF) | `franka` | [configs/franka.yaml](configs/franka.yaml) |
| Universal Robots UR5e (6-DoF) | `ur5e` | [configs/ur5e.yaml](configs/ur5e.yaml) |
| The Robot Studio SO-ARM101 (5-DoF) | `so101` | [configs/so101.yaml](configs/so101.yaml) |

Training, evaluation, the solver and the viewer are shared across robots ([`ik_rl/`](ik_rl)); a robot is just a YAML config (model, joints, TCP, home pose, sampling ranges). Method details and results: [ik_rl/README.md](ik_rl/README.md).

## Installation

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/Char1esFu/ik-rl.git
cd ik-rl
uv venv && uv pip install -r pyproject.toml
```

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
.venv/bin/python train.py --robot ur5e    # train (up to 5M steps, early stop), writes runs/ur5e/<YYYYmmdd_HHMMSS>/
.venv/bin/python eval.py --robot ur5e --restarts 4   # benchmark on random reachable targets
.venv/bin/python visualize.py --robot ur5e           # interactive 3D viewer (web UI at http://localhost:8080 on headless machines)
```

- `--robot` takes a robot name (`configs/<name>.yaml`) or a path to a YAML file; it defaults to `franka` for training. For `eval.py` / `visualize.py` it only selects which runs to search; without it, the newest checkpoint of any robot is used.
- Each training run writes `config.json`, `checkpoints/` and `outputs/` (curves, eval results) to `runs/<robot>/<YYYYmmdd_HHMMSS>/`; use `--run-name` to choose the folder name.
- `eval.py` and `visualize.py` take `--checkpoint` to pick a specific model; the robot is read from the checkpoint.
- Training runs for at most `--total-steps` (default 5M) and stops early when the in-training success rate stays at the `--early-stop-sr` level (default 99.5%, 3 consecutive evaluations; `--early-stop-patience 0` disables it). Ctrl+C stops cleanly; continue an interrupted or finished run with `train.py --resume runs/<robot>/<run>` (optionally with a larger `--total-steps`).
- Method settings (`IKConfig`: environment / reward / observation, and `SACConfig`) have their defaults in [`ik_rl/config.py`](ik_rl/config.py). For experiments, override them without editing code, either from a YAML file that lists only the changed values (see [`configs/exp/example.yaml`](configs/exp/example.yaml); it can also set train.py options) or on the command line:
  ```bash
  .venv/bin/python train.py --exp configs/exp/example.yaml
  .venv/bin/python train.py --robot ur5e --set env.smooth_penalty=0.06 sac.gamma=0.97
  ```
  Precedence: defaults < robot `env_overrides` < `--exp` < `--set` / command-line flags. The resolved values are saved in the run's `config.json` and checkpoints.
- Run any script with `--help` for all options.

## Python API

```python
import numpy as np
from ik_rl import SACIKSolver
from ik_rl.runs import find_latest_checkpoint

solver = SACIKSolver(find_latest_checkpoint("ur5e"))  # robot is read from the checkpoint
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
3. `train.py --robot <name>`.
