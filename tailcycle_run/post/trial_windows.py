#!/usr/bin/env python3
"""Trial windows (door opened -> door closed) from the cbot behavior logs, as
video frame ranges per session.

Logs: <log_root>/<animal>/session_YYYY-MM-DD_HH-MM-SS/robot_manager.log, lines
  2026_09_01_17_33_24_137,  robot_manager_logger, info, door: opened (auto)
in local (America/New_York) time. Video frame times: Cam<ref>_meta.csv
timestamp_sys (epoch ns). Every same-day log is read (a session folder is often
a short app restart with no door events), and each open->closed pair that
overlaps the recording becomes a trial.

Writes <pred_root>/<animal>/<session>/trials.json:
  {"trials": [{"open_s", "close_s", "frame_start", "frame_end", "pretraining", "log"}],
   "n_frames", "in_trial_frames", "logs": [...]}

    trial_windows.py --animals rat tata ... [--sessions ...] [--force]
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg, path as _cfgp  # noqa: E402
LOG_ROOT = _cfgp("local.behavior_log_root")        # <animal>/session_*/robot_manager.log
VIDEO_ROOT = _cfgp("trial_windows.video_root", None)
PRED_ROOT = _cfgp("trial_windows.pred_root", None)
REF_CAM = "Cam2005325"
TZ = ZoneInfo("America/New_York")
LINE = re.compile(r"^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{3}),.*?door: (opened|closed)(.*)$")


def _t(stamp: str) -> float:
    d = dt.datetime.strptime(stamp[:19], "%Y_%m_%d_%H_%M_%S").replace(tzinfo=TZ)
    return d.timestamp() + int(stamp[20:]) / 1000


def door_pairs(log: Path) -> list[dict]:
    out, cur = [], None
    for ln in log.read_text(errors="ignore").splitlines():
        m = LINE.match(ln.strip())
        if not m:
            continue
        t, kind, rest = _t(m.group(1)), m.group(2), m.group(3)
        if kind == "opened":
            cur = (t, "pretraining" in rest)
        elif cur is not None:
            out.append({"open_s": cur[0], "close_s": t, "pretraining": cur[1], "log": log.parent.name})
            cur = None
    return out


def frame_times(vdir: Path) -> np.ndarray | None:
    meta = vdir / f"{REF_CAM}_meta.csv"
    if not meta.exists():
        metas = sorted(vdir.glob("Cam*_meta.csv"))
        if not metas:
            return None
        meta = metas[0]
    a = np.loadtxt(meta, delimiter=",", skiprows=1, usecols=(0, 2), dtype=np.int64)
    ft = np.full(int(a[:, 0].max()) + 1, np.nan)
    ft[a[:, 0]] = a[:, 1] / 1e9
    return ft


def session(animal: str, sess: str, force: bool) -> str:
    out = PRED_ROOT / animal / sess / "trials.json"
    if out.exists() and not force:
        return "exists"
    day = sess[:10].replace("_", "-")
    logs = sorted((LOG_ROOT / animal).glob(f"session_{day}_*/robot_manager.log"))
    if not logs:
        return "no log"
    ft = frame_times(VIDEO_ROOT / animal / sess)
    if ft is None:
        return "no meta"
    t0, t1 = np.nanmin(ft), np.nanmax(ft)
    trials = []
    for log in logs:
        for p in door_pairs(log):
            if p["close_s"] < t0 or p["open_s"] > t1:
                continue
            idx = np.flatnonzero((ft >= p["open_s"]) & (ft <= p["close_s"]))
            if idx.size:
                trials.append({**p, "frame_start": int(idx[0]), "frame_end": int(idx[-1])})
    trials.sort(key=lambda p: p["open_s"])
    n_in = sum(p["frame_end"] - p["frame_start"] + 1 for p in trials)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"trials": trials, "n_frames": int(ft.size), "in_trial_frames": n_in,
                               "logs": [l.parent.name for l in logs], "ref_cam": REF_CAM}, indent=1))
    return f"{len(trials)} trials, {n_in / ft.size:.0%} of {ft.size} frames in trial"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animals", nargs="+", required=True)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--min-cams", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for an in a.animals:
        for v in sorted(p for p in (VIDEO_ROOT / an).iterdir() if p.is_dir()):
            if a.sessions and v.name not in a.sessions:
                continue
            if len(list(v.glob("Cam*.mp4"))) < a.min_cams:
                continue
            print(f"{an}/{v.name}: {session(an, v.name, a.force)}", flush=True)


if __name__ == "__main__":
    main()
