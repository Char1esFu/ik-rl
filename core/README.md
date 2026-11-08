# `core`: robots, kinematics and solver building blocks

The domain core shared by every learning method in [`policy/`](../policy) and by the protocols in
[`evaluation/`](../evaluation/README.md). Layering: `utils` ← `core` ← `evaluation` ← `policy` ← scripts
(`train.py`, `eval.py`, `visualize.py`); `core` imports only `core` and `utils`.

| File | Contents |
|---|---|
| `robot.py` | `RobotSpec`, loaded from `configs/<robot>.yaml` (`get_robot`, `available_robots`) |
| `kinematics.py` | `build_ik_model` (TCP site, axis / target markers), `RobotKinematics`: FK, Jacobian, collision, `apply_joint_step`, joint / workspace boxes, `sample_target` / `sample_start` (the target and start samplers of every method and protocol) |
| `batch_kinematics.py` | `BatchKinematics`: FK + collision for thousands of rows per step on MuJoCo Warp (GPU, CUDA graphs, torch tensors) or CPU worker processes; `sample_valid_configs` |
| `refine.py` | `dls_refine`: damped-least-squares refinement in float64 (the "DLS" of SAC+DLS; `refine=True` of every solver) |
| `solution.py` | `IKSolution`, `checkpoint_robot`, multi-start configurations, target-orientation parsing, refinement over branches |
| `controller.py` | batched closed-loop controller interface (`Controller`, `KinematicController`) and `run_best`, the best-pose / early-stop rollout of any controller |

## `utils`

Generic helpers without robot or method knowledge ([`utils/`](../utils)); `utils` imports nothing from the other
packages at runtime.

| File | Contents |
|---|---|
| `config.py` | dataclass <-> dict helpers (`_ConfigMixin`) and experiment overrides (`--exp` YAML, `--set SECTION.KEY=VALUE`) |
| `runs.py` | `runs/<method>/<robot>/<run>/` lookup (`find_latest_checkpoint`, `run_outputs_dir`) |
| `render.py` | `PoseRenderer`: offscreen images of a configuration with its target (eval galleries) |
| `math_utils.py` | rotation utilities (quaternions, rotation matrices, SO(3) log) |
