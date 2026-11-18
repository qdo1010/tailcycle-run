#!/usr/bin/env python3
"""YOLO-gated tailcycle pose for orma: <session>/yolo_clean/data3D.csv.

Port of climb's analyzeMiceTrajectory/filter_jarvis_with_yolo.py (same keep
rule, same helpers, imported from that repo) onto tailcycle output and the
yolo_track.csv written by post/yolo_presence.py:

  * KEEP a frame iff the top-down YOLO track (Cam2005325) has the mouse there
    AND the pose body centre (Neck/SpineL/TailBase), projected into that cam,
    is within --max-px of the YOLO centroid; then drop 3D teleports.
  * CONTINUITY FILL: unvalidated frames continuous with a kept frame are kept,
    but only inside a presence.json interval -- orma's stand-in for climb's
    shelter rule (outside the intervals the mouse is not in the arena, so a
    pose there is a false detection however smooth it is).

Writes <pred_root>/<animal>/<session>/yolo_clean/{data3D.csv,gate_stats.json},
the path the dashboard's 'yolo_clean' overlay source reads. --source
tailcycle47_v2 gates the 47-kp output instead (-> yolo_clean47_v2/); the
allen set has neck/spine-mid/tail-base at the same indices 3/4/5 as BODY.

    yolo_clean.py --pred-root <pred_root> --animal <animal> --sessions s1 ...
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailcycle_run.config import path as _cfgp  # noqa: E402
from io3d import load_any, write_any  # noqa: E402,F401
# external: analyzeMiceTrajectory/filter_jarvis_with_yolo.py (local.analyze_mice_trajectory)
sys.path.insert(0, str(_cfgp("local.analyze_mice_trajectory")))
from filter_jarvis_with_yolo import (BODY, continuity_filter_track,  # noqa: E402
                                     load_data3d, load_P, velocity_gate_3d,
                                     write_csv)

REF_CAM = "Cam2005325"


def process(pred_root: Path, animal: str, session: str, a) -> None:
    sd = pred_root / animal / session
    src = a.source
    pose, track, pres = (sd / src / "data3D.csv", sd / "yolo_track.csv",
                         sd / "presence.json")
    for p in (pose, track, pres):
        if not p.exists():
            print(f"{animal}/{session}: missing {p.name}, skipping", flush=True)
            return
    calib = yaml.safe_load((sd / src / "info.yaml").read_text())["dataset_name"]
    P = load_P(calib, REF_CAM)
    if src == "tailcycle":
        data = load_data3d(pose)
    else:
        head, data = load_any(pose)
    F = len(data)
    stride = json.loads(pres.read_text())["stride"]
    intervals = json.loads(pres.read_text())["intervals"]

    yf, yx, yy = [], [], []
    with open(track) as fh:
        for row in csv.DictReader(fh):
            if row["mouse_x"] and float(row["mouse_conf"]) >= a.mouse_conf:
                yf.append(int(row["frame"])); yx.append(float(row["mouse_x"]))
                yy.append(float(row["mouse_y"]))
    yf, yx, yy = np.array(yf), np.array(yx), np.array(yy)
    with open(track) as fh:   # the track's own stride (climb's yolo_track/ is 3, ours 6)
        tf = np.array([int(r.split(",", 1)[0]) for r in fh.read().split("\n")[1:] if r])
    if tf.size > 1:
        stride = int(np.median(np.diff(tf)))
    yok = (continuity_filter_track(yf, yx, yy, max_jump_px=130.0, max_gap=4 * stride)
           if len(yf) else np.array([], bool))
    ymouse = {int(yf[i]): (yx[i], yy[i]) for i in range(len(yf)) if yok[i]}

    bc = np.nanmean(data[:, BODY, :3], axis=1)
    proj = bc @ P[:, :3].T + P[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = proj[:, :2] / proj[:, 2:3]
    keep = np.zeros(F, bool)
    for fi in np.flatnonzero(~np.isnan(bc[:, 0]) & (proj[:, 2] > 0)):
        s = (fi // stride) * stride
        m = ymouse.get(s) or ymouse.get(s + stride) or ymouse.get(s - stride)
        if m is not None and np.hypot(uv[fi, 0] - m[0], uv[fi, 1] - m[1]) <= a.max_px:
            keep[fi] = True
    n_agree = int(keep.sum())
    keep = velocity_gate_3d(bc, keep, max_mm_per_frame=15.0, max_gap=60)

    present = np.zeros(F, bool)
    for s, e in intervals:
        present[max(0, s):min(F, e)] = True
    has = ~np.isnan(bc[:, 0])
    n_fill = 0
    for rng, step in ((range(1, F), -1), (range(F - 2, -1, -1), 1)):
        for i in rng:
            j = i + step
            if keep[i] or not has[i] or not keep[j] or not present[i]:
                continue
            if np.linalg.norm(bc[i] - bc[j]) <= a.fill_mm_per_frame:
                keep[i] = True; n_fill += 1

    # behavior-log trial gate (post/trial_windows.py): door opened -> closed
    yolo_keep = keep.copy()
    in_trial = np.ones(F, bool)
    tj = sd / "trials.json"
    trials = json.loads(tj.read_text())["trials"] if tj.exists() else []
    if trials and not a.no_trial_gate:
        pad = int(a.trial_pad_s * 180)
        in_trial[:] = False
        for t in trials:
            in_trial[max(0, t["frame_start"] - pad):min(F, t["frame_end"] + pad + 1)] = True
        keep &= in_trial
    trial_gate = bool(trials) and not a.no_trial_gate

    cleaned = data.copy()
    cleaned[~keep] = np.nan
    out = sd / ("yolo_clean" if src == "tailcycle"
                else "yolo_clean" + src.removeprefix("tailcycle")) / "data3D.csv"
    if src == "tailcycle":
        write_csv(cleaned, out)
    else:
        write_any(head, cleaned, out)
    stats = dict(frames=F, detected=float(has.mean()), yolo_mouse_samples=len(ymouse),
                 agree=n_agree, keep=float(keep.mean()), kept_frames=int(keep.sum()),
                 continuity_fill=n_fill, outside_presence_dropped=int((has & ~present).sum()),
                 max_px=a.max_px, mouse_conf=a.mouse_conf, source=src, stride=stride,
                 trial_gate=trial_gate, n_trials=len(trials),
                 yolo_keep_frames=int(yolo_keep.sum()),
                 dropped_outside_trials=int((yolo_keep & ~in_trial).sum()),
                 trial_pad_s=a.trial_pad_s)
    np.savez_compressed(out.parent / "masks.npz", yolo_keep=yolo_keep, in_trial=in_trial)
    (out.parent / "gate_stats.json").write_text(json.dumps(stats, indent=1))
    # the viewer reprojects with <source>/info.yaml's calibration (orma
    # sessions have no root info.yaml), so carry tailcycle's over
    (out.parent / "info.yaml").write_text((sd / src / "info.yaml").read_text())
    print(f"{animal}/{session}: {src} {100 * has.mean():.0f}% of frames -> "
          f"KEEP {100 * keep.mean():.0f}% ({keep.sum():,}/{F:,}; agree {n_agree:,}, "
          f"fill {n_fill:,}; " + (f"{len(trials)} trials, -{stats['dropped_outside_trials']:,} "
          f"outside" if trial_gate else "no trial gate") + f") -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True, type=Path)
    ap.add_argument("--animal", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--source", default="tailcycle",
                    help="pose dir to gate: tailcycle (24 kp) or tailcycle47_v2")
    ap.add_argument("--max-px", type=float, default=220.0, help="as climb")
    ap.add_argument("--mouse-conf", type=float, default=0.25, help="as climb")
    ap.add_argument("--fill-mm-per-frame", type=float, default=15.0, help="as climb")
    ap.add_argument("--no-trial-gate", action="store_true",
                    help="ignore trials.json (keep YOLO-validated frames between trials too)")
    ap.add_argument("--trial-pad-s", type=float, default=1.0)
    a = ap.parse_args()
    for s in a.sessions:
        process(a.pred_root, a.animal, s, a)


if __name__ == "__main__":
    main()
