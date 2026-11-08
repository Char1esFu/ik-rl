# `evaluation`: method-independent evaluation protocols and the interactive viewer

Two protocols that take any policy's solver or batched controller; they import only [`core`](../core/README.md)
and `utils`. The methods in [`policy/`](../policy) use them for their in-training evaluations,
[`eval.py`](../eval.py) runs them from the command line (`--protocol benchmark | hold`), and
[`visualize.py`](../visualize.py) opens the interactive viewer (`--method` / `--robot` / `--checkpoint`).
The viewer, like the hold protocol, steps any method through the batched controller interface
(`core.controller.Controller`; `policy.make_controller` builds one for a checkpoint).

| File | Contents |
|---|---|
| `benchmark.py` | this repository's benchmark: `EvalTargets` (fixed reachable targets + start configurations, drawn with `RobotKinematics.sample_target` / `sample_start`), `evaluate_solver` (solve with `num_restarts` starts, stop at 2 mm / 0.02 rad, at most `max_steps`; optional DLS refinement), success / error metrics at 5 mm, 2 mm and 0.1 mm |
| `hold.py` | PolicyIK's hold protocol: `HoldTasks`, `evaluate_hold` (160-step attempts; strict = final-second mean errors < 1 mm and < 0.2°, coarse 1 cm / 2°; entry, containment, drift), the deployment protocol (error-corrected goal, 8 transient-goal attempts), `refine_final` (DLS refinement of final configurations) |
| `rollout.py` | `run_horizon`: fixed-horizon rollout of a controller (per-step errors, TCP poses, optional transient goal) |
| `baselines.py` | `DLSController`: classical damped-least-squares controller under the same step rules (a reference, not a trained method) |
| `viewer/` | interactive viewer: `tracker.py` (`MultiStartTracker`: the visible arm plus shadow branches from home / random seeds, rescue by joint-space slew to the best branch, settle / damp rules), `app.py` (native GLFW window or browser UI streamed over WebSocket; auto / manual sliders / figure-8 modes), `viewer.html` |

```bash
.venv/bin/python visualize.py --method policyik --robot franka                          # interactive viewer
.venv/bin/python eval.py --method policyik --robot franka --restarts 4 --refine        # benchmark, newest PolicyIK run
.venv/bin/python eval.py --protocol hold --robot franka --dls --refine                 # hold: newest run of every method + DLS
.venv/bin/python eval.py --protocol hold --checkpoint a=<ckpt> b=<ckpt> --out out.json # hold: chosen checkpoints
```

Fixed seeds keep targets identical across methods and code versions: benchmark seed 2026 (500 targets; the
SAC+DLS training curves use seed + 2000), hold seed 7 (2000 tasks, the same targets as
`.agents/skills/ik-rl-dev/scripts/compare_models.py`).
