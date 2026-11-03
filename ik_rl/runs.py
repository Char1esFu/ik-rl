"""Helpers for locating training runs (runs/<robot>/<timestamp>/{checkpoints,outputs})."""

from __future__ import annotations

from pathlib import Path

DEFAULT_RUNS_DIR = "runs"


def find_latest_checkpoint(
    robot: str | None = None,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
    name: str = "best_model.pt",
) -> Path:
    """Most recently written runs/<robot>/*/checkpoints/<name> (any robot if robot is None)."""
    pattern = f"{robot or '*'}/*/checkpoints/{name}"
    candidates = list(Path(runs_dir).glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No '{name}' found under {runs_dir}/{robot or '*'}/*/checkpoints/. "
            "Train first or pass --checkpoint."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_outputs_dir(checkpoint: str | Path) -> Path:
    """outputs/ dir belonging to a checkpoint stored as <run>/checkpoints/<file>."""
    ckpt = Path(checkpoint).resolve()
    if ckpt.parent.name == "checkpoints":
        return ckpt.parent.parent / "outputs"
    return ckpt.parent / "outputs"
