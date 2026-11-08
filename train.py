"""Train an IK policy: python train.py --method {sac+dls,policyik} --robot <name> [method options].

Runs are written to runs/<method>/<robot>/<YYYYmmdd_HHMMSS>/. Every option after --method / --robot is passed
to the method's trainer (policy/sac_dls/train.py, policy/policyik/train.py); `train.py --method <m> --help`
lists them. With --resume <run dir>, --method may be omitted (read from the run).
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from core.robot import DEFAULT_ROBOT, available_robots
from policy import DEFAULT_METHOD, METHODS


def method_of_run(run_dir: str) -> str:
    """Method of an existing run: config.json 'method', else the runs/<method>/... path, else SAC+DLS."""
    cfg = Path(run_dir) / "config.json"
    if cfg.exists():
        method = json.loads(cfg.read_text()).get("method")
        if method in METHODS:
            return method
    for part in Path(run_dir).resolve().parts:
        if part in METHODS:
            return part
    return DEFAULT_METHOD


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, add_help=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=list(METHODS), default=None, help="learning method (required for new runs)")
    ap.add_argument("--robot", default=None, help=f"{', '.join(available_robots())} or a robot YAML (default "
                    f"{DEFAULT_ROBOT})")
    ap.add_argument("--resume", default=None, help="run directory to continue")
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args()
    if args.method is None and args.resume:
        args.method = method_of_run(args.resume)
    if args.method is None:
        if args.help:
            ap.print_help()
            return
        ap.error("--method is required (sac+dls or policyik)")
    argv = list(rest)
    if args.robot is not None:
        argv += ["--robot", args.robot]
    elif not args.resume:
        argv += ["--robot", DEFAULT_ROBOT]
    if args.resume:
        argv += ["--resume", args.resume]
    if args.help:
        argv.append("--help")
    trainer = importlib.import_module(METHODS[args.method] + ".train")
    sys.argv[0] = f"train.py --method {args.method}"
    trainer.main(argv)


if __name__ == "__main__":
    main()
