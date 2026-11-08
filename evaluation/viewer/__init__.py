"""Interactive viewers (native window / browser) for real-time IK tracking with any method's controller.

Entry point: visualize.py (chooses the method / checkpoint and passes a controller factory to run_viewer).
"""

from evaluation.viewer.app import run_viewer
from evaluation.viewer.tracker import MultiStartTracker

__all__ = ["run_viewer", "MultiStartTracker"]
