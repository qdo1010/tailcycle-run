"""tailcycle-run CLI.

    python -m tailcycle_run plan   --ranges R [--video-dir D]   # GPU-h + TB, no staging
    python -m tailcycle_run plan   --frames N --windows K
    python -m tailcycle_run submit <cluster_run.py args>        # one session
    python -m tailcycle_run batch  <run_sequential.py args>     # sessions one at a time
    python -m tailcycle_run post   yolo-clean|body-span|presence|trials|align <args>
    python -m tailcycle_run guard  --array ID [--merge ID] --cap GPU_H
"""
from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = {
    "submit": HERE / "cluster" / "cluster_run.py",
    "batch": HERE / "cluster" / "run_sequential.py",
    "guard": HERE / "cluster" / "burn_guard.py",
}
POST = {
    "yolo-clean": HERE / "post" / "yolo_clean.py",
    "body-span": HERE / "post" / "body_span.py",
    "presence": HERE / "post" / "yolo_presence.py",
    "trials": HERE / "post" / "trial_windows.py",
    "align": HERE / "post" / "apply_world_alignment.py",
    "align-fit": HERE / "post" / "align_world_frames.py",
}


def _run(script: Path, argv: list[str]) -> None:
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def _plan(argv: list[str]) -> None:
    from tailcycle_run.plan import plan_ranges, projection, staged_tb
    ap = argparse.ArgumentParser(prog="tailcycle-run plan")
    ap.add_argument("--ranges", type=Path)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--windows", type=int)
    ap.add_argument("--video-dir", type=Path, help="session video dir, for the staged TB")
    a = ap.parse_args(argv)
    if a.ranges:
        p = plan_ranges(a.ranges)
    elif a.frames is not None and a.windows is not None:
        p = {"windows": a.windows, "frames": a.frames, "gpu_h": projection(a.frames, a.windows)}
    else:
        ap.error("give --ranges, or --frames and --windows")
    print(f"{p['windows']} windows, {p['frames']:,} frames -> projected {p['gpu_h']:.1f} GPU-h"
          + (f"; staged {staged_tb(a.video_dir):.3f} TB (all cams)" if a.video_dir else ""))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "plan":
        _plan(rest)
    elif cmd in SCRIPTS:
        _run(SCRIPTS[cmd], rest)
    elif cmd == "post":
        if not rest or rest[0] not in POST:
            sys.exit(f"post: one of {', '.join(POST)}")
        _run(POST[rest[0]], rest[1:])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
