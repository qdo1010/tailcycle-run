"""Per-camera blur detection for a recording session.

A blurry / out-of-focus camera (e.g. Cam2006515 on some climb days) injects
noise into tailcycle's multi-view triangulation. This module measures each
camera's sharpness (variance of the Laplacian, the standard focus metric)
over a few sampled frames and flags cameras far below the session median.

Reading 16 videos is slow (~seconds/session), so results are cached in a
single central JSON keyed by "animal/session". Videos are immutable once
recorded, so the cache never needs invalidation. `session_blur` returns the
cached entry if present and computes+caches otherwise. Cache path:
config local.blur_cache (default <local.log_dir>/blur_cache.json); point it
at a dashboard's cache to share results.
"""
from __future__ import annotations
import json
import threading
from pathlib import Path

import cv2
import numpy as np

from tailcycle_run.config import path as _cfgp

CACHE = _cfgp("local.blur_cache",
              str(_cfgp("local.log_dir", "/tmp/tailcycle_run") / "blur_cache.json"))
_lock = threading.Lock()

# A camera below this fraction of the session-median sharpness is "blurry".
# Calibrated on 2026_08_03 climb (Cam2006515 = 0.41x, clear outlier) vs a
# normally-dim session where the softest cam was 0.61x.
BLUR_FRACTION = 0.5
N_SAMPLE_FRAMES = 6


def _load_cache() -> dict:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2))
    tmp.replace(CACHE)


def _camera_sharpness(mp4: Path) -> float | None:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    vals = []
    for f in np.linspace(n * 0.1, n * 0.9, N_SAMPLE_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vals.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    cap.release()
    return float(np.median(vals)) if vals else None


def compute_blur(video_dir: Path) -> dict:
    """Measure sharpness of every Cam*.mp4 in video_dir. Returns
    {sharpness: {cam: val}, median: float, blurry: [cam,...]}."""
    sharp = {}
    for mp4 in sorted(video_dir.glob("Cam*.mp4")):
        v = _camera_sharpness(mp4)
        if v is not None:
            sharp[mp4.stem] = v
    if not sharp:
        return {"sharpness": {}, "median": None, "blurry": []}
    med = float(np.median(list(sharp.values())))
    blurry = sorted(cam for cam, v in sharp.items() if v < BLUR_FRACTION * med)
    return {"sharpness": sharp, "median": med, "blurry": blurry}


def session_blur(animal: str, session: str, video_dir: Path,
                  compute_if_missing: bool = False,
                  force: bool = False) -> dict:
    """Cached blur entry for one session. When compute_if_missing is False
    (the /api/sessions path) returns {'checked': False} if not yet scanned,
    so the row render never pays the video-read cost.

    force=True recomputes even if cached — for sessions whose videos were
    unreadable at scan time (still copying / not yet finalized) and have
    since healed. Videos are otherwise immutable, so this is the only
    invalidation path."""
    key = f"{animal}/{session}"
    if not force:
        with _lock:
            cache = _load_cache()
            if key in cache:
                return {**cache[key], "checked": True}
    if force:
        compute_if_missing = True
    if not compute_if_missing:
        return {"checked": False, "blurry": []}
    if not video_dir.is_dir() or not any(video_dir.glob("Cam*.mp4")):
        return {"checked": False, "blurry": []}
    entry = compute_blur(video_dir)
    with _lock:
        cache = _load_cache()
        cache[key] = entry
        _save_cache(cache)
    return {**entry, "checked": True}
