#!/usr/bin/env python3
"""Ranges file of frames where the mouse is near a mapped hurdle sheet.

Reads <pred_root>/<animal>/<session>/yolo_track.csv (post/yolo_presence.py,
top cam Cam2005325, full-res px) and objects/hurdle_map.json (the per-session
sheet map: objects[] with height, half, line_px; its producer is not part of
this package). It keeps the
track frames whose mouse centre lies within --near-mm of a chosen sheet's
top-cam line. The sheet is chosen by --min-height, or else it is the tallest.
mm/px comes from that sheet's own mapped length over its pixel length.
Each kept frame is padded by --pad-s, overlapping windows are merged, and
windows longer than --max-frames are split, so every array element stays
well inside its wall. The output is one 'FS FE' line per window (half-open
source frames), the format of cluster_run --ranges-file.

    tailcycle-run post hurdle-windows --pred-root <pred_root> --animal <animal> \
        --session <session> --out ranges.txt [--near-mm 100 --pad-s 1 | --crossings]
--crossings: keep only frames around sheet CROSSINGS instead (the mouse centre
changes side of the sheet's line within its span, both samples within
--cross-mm of it and <= 1 s apart; flips < 0.5 s after the previous one are
jitter). "Near" (100 mm) kept 92 % of one test session -- the mouse is by the
block most of the time -- so the smoke used crossings (2026-11-17).
Prints windows, frames, and the GPU-h projection (frames/5000 + 0.045 x windows,
refit on gpu_l4 for univ2_47, 2026-11-18).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.plan import projection  # noqa: E402


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    t = np.clip(((p - a) @ d) / (d @ d), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * d), axis=1)


def windows_from_frames(frames, pad: int, n_total: int, max_frames: int) -> list[tuple[int, int]]:
    """Pad each kept frame by `pad`, merge overlapping windows, split windows
    longer than `max_frames`. Half-open (FS, FE) source frames."""
    win: list[list[int]] = []
    for f in sorted(int(f) for f in frames):
        s, e = max(0, f - pad), min(n_total, f + pad + 1)
        if win and s <= win[-1][1]:
            win[-1][1] = max(win[-1][1], e)
        else:
            win.append([s, e])
    out = []
    for s, e in win:
        while e - s > max_frames:
            out.append((s, s + max_frames))
            s += max_frames
        out.append((s, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--animal", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--near-mm", type=float, default=100.0)
    ap.add_argument("--pad-s", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=180.0)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--min-height", type=float, default=None,
                    help="use every sheet at least this tall (default: the tallest only)")
    ap.add_argument("--max-frames", type=int, default=20000)
    ap.add_argument("--crossings", action="store_true")
    ap.add_argument("--cross-mm", type=float, default=60.0)
    a = ap.parse_args()
    sd = a.pred_root / a.animal / a.session
    objs = json.loads((sd / "objects" / "hurdle_map.json").read_text())["objects"]
    if not objs:
        print(f"{a.animal}/{a.session}: no mapped sheet in hurdle_map.json -> no windows")
        return
    sheets = ([o for o in objs if o["height"] >= a.min_height] if a.min_height is not None
              else [max(objs, key=lambda o: o["height"])])
    rows = list(csv.DictReader(open(sd / "yolo_track.csv")))
    fr = np.array([int(r["frame"]) for r in rows])
    xy = np.array([[float(r["mouse_x"] or "nan"), float(r["mouse_y"] or "nan")] for r in rows])
    conf = np.array([float(r["mouse_conf"] or 0) for r in rows])
    near = np.zeros(len(fr), bool)
    for o in sheets:
        p0, p1 = (np.array(p, float) for p in o["line_px"])
        mm_per_px = 2 * o["half"][0] / np.linalg.norm(p1 - p0)
        ok = (conf >= a.min_conf) & np.isfinite(xy[:, 0])
        d = np.full(len(fr), np.inf)
        d[ok] = seg_dist(xy[ok], p0, p1) * mm_per_px
        if a.crossings:
            u = (p1 - p0) / np.linalg.norm(p1 - p0)
            nrm = np.array([-u[1], u[0]])
            q = xy - p0
            sd_mm = (q @ nrm) * mm_per_px
            t = (q @ u) / np.linalg.norm(p1 - p0)
            vi = np.nonzero(ok)[0]
            last = -10 ** 9
            for i, j in zip(vi[:-1], vi[1:]):
                if (np.sign(sd_mm[i]) != np.sign(sd_mm[j]) and fr[j] - fr[i] <= a.fps
                        and -0.05 < t[j] < 1.05 and max(abs(sd_mm[i]), abs(sd_mm[j])) < a.cross_mm):
                    if fr[j] - last > a.fps / 2:
                        near[j] = True
                    last = fr[j]
            print(f"  {int(near.sum())} crossings")
        else:
            near |= d <= a.near_mm
        print(f"sheet {o['height']:.0f} mm L {2 * o['half'][0]:.0f} mm: {mm_per_px:.3f} mm/px, "
              f"{int((d <= a.near_mm).sum())} of {len(fr)} track frames within {a.near_mm:.0f} mm")
    pad = int(round(a.pad_s * a.fps))
    n_total = int(fr.max()) + 1
    out = windows_from_frames(fr[near], pad, n_total, a.max_frames)
    a.out.write_text("".join(f"{s} {e}\n" for s, e in out))
    frames = sum(e - s for s, e in out)
    md5 = hashlib.md5(a.out.read_bytes()).hexdigest()
    print(f"{len(out)} windows, {frames:,} frames ({100 * frames / n_total:.1f}% of {n_total:,}), "
          f"projected {projection(frames, len(out)):.1f} GPU-h; {a.out} md5 {md5}")


if __name__ == "__main__":
    main()
