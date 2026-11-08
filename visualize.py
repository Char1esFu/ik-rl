"""Interactive MuJoCo 3D viewer: real-time 6D IK tracking with a trained policy of any method.

  .venv/bin/python visualize.py --method policyik --robot ur5e      # newest PolicyIK run of the UR5e
  .venv/bin/python visualize.py --checkpoint runs/sac+dls/franka/<run>/checkpoints/best_model.pt

Two front ends (evaluation/viewer/):
1. Web viewer (auto-enabled on headless / SSH machines without $DISPLAY, or via --web): EGL offscreen rendering
   streamed over one port (HTTP + WebSocket), with camera orbit / pan / zoom and 6D pose controls in the browser
   (auto targets, manual sliders, figure-8).
2. Native desktop viewer (--native, or default when $DISPLAY is available): GLFW window via mujoco.viewer.
The method and robot are read from the checkpoint; the tracker runs `--restarts` start branches and rescues the
visible arm from local minima by slewing to the best branch.
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--method", type=str, default=None, choices=["sac+dls", "policyik"],
        help="Method whose newest checkpoint to use if --checkpoint is not given (default: any)",
    )
    parser.add_argument(
        "--robot", type=str, default=None,
        help="Used to find the newest checkpoint if --checkpoint is not given (default: any robot)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint of any method (default: newest runs/<method>/<robot>/*/checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=4,
        help="Parallel multi-start IK branches (1 = single-thread from q_cur, 4 = default with backup branches)",
    )
    parser.add_argument("--hold-sec", type=float, default=1.0, help="Pause at solved target before sampling next")
    parser.add_argument("--web", action="store_true", help="Force browser-based interactive 3D viewer")
    parser.add_argument("--native", action="store_true", help="Force native GLFW desktop window")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host for web viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port for web viewer (auto-forwarded in Remote SSH)")
    parser.add_argument("--width", type=int, default=960, help="Initial web render width (the page then requests its own size)")
    parser.add_argument("--height", type=int, default=640, help="Initial web render height")
    parser.add_argument(
        "--max-width", type=int, default=1920,
        help="Web render resolution cap; the page's size is scaled down to fit (higher = sharper but slower, <= 2560)",
    )
    parser.add_argument("--max-height", type=int, default=1200, help="Web render resolution cap (<= 1600)")
    parser.add_argument("--fps", type=int, default=30, help="Target frame rate for web viewer")
    parser.add_argument("--jpeg-quality", type=int, default=82, help="JPEG quality (1-95) for web stream")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    use_web = args.web or (not args.native and not has_display)
    if use_web:
        os.environ.setdefault("MUJOCO_GL", "egl")  # before anything imports mujoco

    import torch

    from evaluation.viewer import run_viewer
    from policy import checkpoint_method, load_solver, make_controller
    from utils.runs import find_latest_checkpoint

    if args.checkpoint is None:
        args.checkpoint = str(find_latest_checkpoint(args.robot, method=args.method))
    method = checkpoint_method(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    print(f"Using checkpoint: {args.checkpoint} ({method})")
    torch.set_num_threads(1)  # the viewer steps a few rows: tiny CPU tensors only contend with more threads
    solver = load_solver(args.checkpoint)
    label = {"sac+dls": "SAC+DLS", "policyik": "PolicyIK"}[method]
    run_viewer(lambda: make_controller(solver, cpu=True), solver.robot, label, args, web=use_web)


if __name__ == "__main__":
    main()
