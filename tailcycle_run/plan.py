"""GPU-h projection before any staging.

Refit on gpu_l4 for univ2_47 (2026-11-17/18): 1.4-1.7 fps per element plus
about 0.045 GPU-h (~2.7 min) of model/ffmpeg startup per window/chunk:

    GPU-h ~= frames / 5000 + 0.045 * windows

(The earlier frames/5000 + windows/60 under-counted startup: a 263-window,
119,618-frame smoke projects 35.8 GPU-h and measured 35.7.)
"""
from __future__ import annotations
from pathlib import Path

FRAMES_PER_GPU_H = 5000.0
STARTUP_GPU_H_PER_WINDOW = 0.045


def parse_ranges(path: Path | str) -> list[tuple[int, int]]:
    """A cluster_run --ranges-file: one 'FS FE' line per window, half-open
    source frames. Blank lines and '#' comments are ignored; FE must be > FS."""
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"{path}:{n}: expected 'FS FE', got {line!r}")
        fs, fe = int(parts[0]), int(parts[1])
        if fs < 0 or fe <= fs:
            raise ValueError(f"{path}:{n}: bad window {fs} {fe}")
        out.append((fs, fe))
    return out


def projection(frames: int, windows: int) -> float:
    return frames / FRAMES_PER_GPU_H + windows * STARTUP_GPU_H_PER_WINDOW


def plan_ranges(path: Path | str) -> dict:
    w = parse_ranges(path)
    frames = sum(fe - fs for fs, fe in w)
    return {"windows": len(w), "frames": frames, "gpu_h": projection(frames, len(w))}


def staged_tb(video_dir: Path, cams: list[str] | None = None) -> float:
    """Bytes the session's videos would stage (all Cam*.mp4, or only `cams`)."""
    vids = [p for p in Path(video_dir).glob("Cam*.mp4") if cams is None or p.stem in cams]
    return sum(p.stat().st_size for p in vids) / 1e12
