"""IK policies, side by side: SAC+DLS (policy/sac_dls) and PolicyIK (policy/policyik).

Both build on the shared, method-independent packages core/ (robots, kinematics, DLS refinement, solver
building blocks), utils/ (config, runs, rendering, rotations) and evaluation/ (benchmark and hold protocols).
A method name is also its runs directory: runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

METHODS = {"sac+dls": "policy.sac_dls", "policyik": "policy.policyik"}  # method name -> package
DEFAULT_METHOD = "sac+dls"


def checkpoint_method(meta: dict[str, Any]) -> str:
    """Method of a checkpoint payload (SAC+DLS checkpoints written before the method field have none)."""
    method = meta.get("method") or meta.get("meta", {}).get("method") or DEFAULT_METHOD
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r} in checkpoint; known: {list(METHODS)}")
    return method


def load_solver(checkpoint: str | Path, device: str = "cuda"):
    """SACIKSolver or PolicyIKSolver for a checkpoint (same solve() / solve_batch() API, robot read from it)."""
    import torch

    meta = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_method(meta) == "policyik":
        from policy.policyik.solver import PolicyIKSolver

        return PolicyIKSolver(checkpoint, device=device)
    from policy.sac_dls.solver import SACIKSolver

    return SACIKSolver(checkpoint, device=device)


def make_controller(solver, bk=None, cpu: bool = False):
    """A new batched closed-loop controller (core.controller.Controller) driven by a solver's policy.

    bk: BatchKinematics to step on (default: the solver's, or an in-process CPU one); cpu=True keeps a
    PolicyIK controller's state and network on the CPU (a few rows, e.g. the viewer: no GPU launch latency).
    """
    from core.batch_kinematics import BatchKinematics

    from policy.policyik.solver import PolicyIKSolver

    if isinstance(solver, PolicyIKSolver):
        from policy.policyik.controller import PolicyIKController

        if cpu:
            return PolicyIKController(solver.cpu_controller.agent, bk or solver.bk, solver.cfg, device="cpu")
        return PolicyIKController(solver.agent, bk or solver.bk, solver.cfg)
    from policy.sac_dls.controller import SACController

    return SACController(solver.agent, bk or BatchKinematics(solver.robot, backend="cpu"), solver.cfg)
