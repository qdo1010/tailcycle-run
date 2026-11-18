#!/usr/bin/env python3
"""Top-down YOLO mouse track + "mouse in arena" intervals for a session.

Same detector and CSV schema as climb's analyzeMiceTrajectory/yolo_track_cam.py
(YOLO11m, classes Mouse / SideCyl, reference cam Cam2005325), so the climb
gating logic ports. Used to skip / restrict tailcycle where the mouse isn't in
the arena (most orma trials have long empty stretches).

Writes, per session under <pred_root>/<animal>/<session>/:
    yolo_track.csv   frame, mouse_x, mouse_y, mouse_conf, cyl_x, cyl_y, cyl_conf
                     (every `stride`-th frame; x/y in full-resolution pixels)
    presence.json    mouse-present intervals [start, end) in frame indices,
                     present seconds, fraction of the recording

Frames are decoded by ONE sequential ffmpeg pipe from frame 0 and indexed by
decode order -- never by seeking: orma video is MPEG-TS, where seeks land on
the wrong keyframe. Decode order is the same index tailcycle uses.

    yolo_presence.py --video-root <video_root> \
        --pred-root <pred_root> --animal <animal> --sessions s1 s2 ...

Runs in the `yolo` conda env (ultralytics), which has no FastAPI -- hence
explicit roots instead of importing app.SOURCES.
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg  # noqa: E402
from tailcycle_run.vendor.ffprobe import _ffprobe_frames   # noqa: E402

WEIGHTS  = _cfg("local.yolo_weights")
REF_CAM  = "Cam2005325"
WIDTH    = 960          # decode-side downscale = the model's imgsz
MOUSE, CYL = 0, 1


def _frames(video: Path, stride: int, w: int, h: int):
    """Yield (frame_index, BGR image) for every stride-th frame, sequentially."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"select='not(mod(n\\,{stride}))',scale={w}:{h}",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 3 * 8)
    size = w * h * 3
    k = 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield k * stride, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.kill()
        proc.wait()


def _best(boxes, cls_id, sx, sy):
    best = None
    for b in boxes:
        if int(b.cls) == cls_id:
            c = float(b.conf)
            if best is None or c > best[0]:
                x, y = b.xywh[0].tolist()[:2]
                best = (c, x * sx, y * sy)
    return best


def presence_intervals(frames, conf, min_conf, gap, pad, n_frames):
    """[start, end) frame intervals where the mouse is detected: detections
    >= min_conf, gaps shorter than `gap` frames bridged, `pad` frames added
    on each side (clipped to the recording)."""
    hit = frames[conf >= min_conf]
    if hit.size == 0:
        return []
    out = []
    s = e = int(hit[0])
    for f in hit[1:]:
        f = int(f)
        if f - e > gap:
            out.append([s, e])
            s = f
        e = f
    out.append([s, e])
    return [[max(0, a - pad), min(n_frames, b + 1 + pad)] for a, b in out]


def run_session(model, animal: str, session: str, a) -> dict | None:
    video = a.video_root / animal / session / f"{REF_CAM}.mp4"
    out_dir = a.pred_root / animal / session
    if not video.exists():
        print(f"{animal}/{session}: no {REF_CAM}.mp4, skipping", flush=True)
        return None
    track = out_dir / "yolo_track.csv"
    if track.exists() and not a.force:
        print(f"{animal}/{session}: yolo_track.csv exists (use --force)", flush=True)
    else:
        import cv2  # noqa: F401  (ultralytics wants it; import early for errors)
        W0, H0 = map(int, subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=30   # TS lists the stream
            ).stdout.split()[0].split(",")[:2])              # twice (program + stream)
        h = int(round(H0 * WIDTH / W0 / 2)) * 2
        sx, sy = W0 / WIDTH, H0 / h
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = track.with_suffix(".csv.tmp")
        t0 = time.time()
        n = 0
        with open(tmp, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "mouse_x", "mouse_y", "mouse_conf",
                        "cyl_x", "cyl_y", "cyl_conf"])
            batch = []

            def flush():
                res = model.predict([im for _, im in batch], imgsz=WIDTH,
                                    conf=a.conf, verbose=False, device=a.device)
                for (fi, _), r in zip(batch, res):
                    mb = _best(r.boxes, MOUSE, sx, sy)
                    cb = _best(r.boxes, CYL, sx, sy)
                    w.writerow([fi,
                                f"{mb[1]:.1f}" if mb else "", f"{mb[2]:.1f}" if mb else "",
                                f"{mb[0]:.3f}" if mb else "",
                                f"{cb[1]:.1f}" if cb else "", f"{cb[2]:.1f}" if cb else "",
                                f"{cb[0]:.3f}" if cb else ""])
                batch.clear()

            for fi, im in _frames(video, a.stride, WIDTH, h):
                batch.append((fi, im))
                if len(batch) == a.batch:
                    flush()
                n += 1
                if n % 5000 == 0:
                    print(f"  {animal}/{session}: frame {fi:,} "
                          f"({n / (time.time() - t0):.0f} samples/s)", flush=True)
            if batch:
                flush()
        tmp.replace(track)
        print(f"  {animal}/{session}: {n:,} samples in {time.time() - t0:.0f}s "
              f"-> {track}", flush=True)

    fps = a.fps
    n_frames = _ffprobe_frames(video) or 0
    fr, cf = [], []
    with open(track) as fh:
        for row in csv.DictReader(fh):
            fr.append(int(row["frame"]))
            cf.append(float(row["mouse_conf"]) if row["mouse_conf"] else 0.0)
    fr, cf = np.asarray(fr), np.asarray(cf)
    n_frames = max(n_frames, int(fr[-1]) + 1 if fr.size else 0)
    iv = presence_intervals(fr, cf, a.min_conf, int(a.gap_s * fps),
                            int(a.pad_s * fps), n_frames)
    present = sum(b - s for s, b in iv)
    summary = {"session": session, "animal": animal, "ref_cam": REF_CAM,
               "n_frames": n_frames, "fps": fps, "stride": a.stride,
               "min_conf": a.min_conf, "gap_s": a.gap_s, "pad_s": a.pad_s,
               "intervals": iv, "present_s": round(present / fps, 1),
               "present_frac": round(present / n_frames, 4) if n_frames else 0.0,
               "weights": WEIGHTS}
    (out_dir / "presence.json").write_text(json.dumps(summary, indent=1))
    print(f"{animal}/{session}: mouse present {summary['present_s']:.0f} s "
          f"({100 * summary['present_frac']:.1f}%) in {len(iv)} interval(s)",
          flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-root", required=True, type=Path)
    ap.add_argument("--pred-root", required=True, type=Path)
    ap.add_argument("--animal", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=6, help="every Nth frame (6 = 30 Hz at 180 fps)")
    ap.add_argument("--fps", type=float, default=180.0)
    ap.add_argument("--conf", type=float, default=0.25, help="detector threshold (as climb)")
    ap.add_argument("--min-conf", type=float, default=0.4, help="mouse conf that counts as present")
    ap.add_argument("--gap-s", type=float, default=2.0, help="bridge absences shorter than this")
    ap.add_argument("--pad-s", type=float, default=1.0, help="pad each interval by this")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--force", action="store_true", help="re-run YOLO even if yolo_track.csv exists")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    for s in a.sessions:
        try:
            run_session(model, a.animal, s, a)
        except Exception as e:   # one unreadable video must not end the batch
            print(f"{a.animal}/{s}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
