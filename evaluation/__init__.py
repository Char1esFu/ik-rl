"""Evaluation protocols, independent of the learning method (they take a solver or a batched controller):

- benchmark.py: this repository's benchmark (EvalTargets, evaluate_solver: stop at 2 mm / 0.02 rad)
- hold.py: PolicyIK's hold protocol (reach and hold 1 mm / 0.2 deg for 1 s, deployment protocol)
- baselines.py: DLS baseline controller; rollout.py: fixed-horizon rollouts
Depends on core/ and utils/; policy/ uses it for in-training evaluation, eval.py drives it from the command line.
"""

from evaluation.benchmark import EvalTargets, evaluate_solver
from evaluation.hold import HoldTasks, evaluate_hold

__all__ = ["EvalTargets", "evaluate_solver", "HoldTasks", "evaluate_hold"]
