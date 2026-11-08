"""Helpers for locating training runs: runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/{checkpoints,outputs}."""

from __future__ import annotations

from pathlib import Path

DEFAULT_RUNS_DIR = "runs"


def method_runs_dir(method: str, runs_dir: str | Path = DEFAULT_RUNS_DIR) -> Path:
    """runs/<method> (method names as in policy.METHODS, e.g. 'sac+dls', 'policyik')."""
    return Path(runs_dir) / method


def find_latest_checkpoint(
    robot: str | None = None,
    method: str | None = None,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    name: str = "best_model.pt",
) -> Path:
    """Most recently written runs/<method>/<robot>/*/checkpoints/<name> (any method / robot if None)."""
    pattern = f"{method or '*'}/{robot or '*'}/*/checkpoints/{name}"
    candidates = list(Path(runs_dir).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No '{name}' found under {runs_dir}/{method or '*'}/{robot or '*'}/*/checkpoints/. "
            "Train first or pass --checkpoint."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_outputs_dir(checkpoint: str | Path) -> Path:
    """outputs/ dir belonging to a checkpoint stored as <run>/checkpoints/<file>."""
    ckpt = Path(checkpoint).resolve()
    if ckpt.parent.name == "checkpoints":
        return ckpt.parent.parent / "outputs"
    return ckpt.parent / "outputs"
