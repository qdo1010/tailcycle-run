"""Frame count from a video container header (vendored from the dashboard's
health_audit). ffprobe binary: config local.ffprobe (default "ffprobe" on PATH)."""
from __future__ import annotations
import subprocess
from pathlib import Path

from tailcycle_run.config import get as _cfg

FFPROBE = _cfg("local.ffprobe", "ffprobe")


def _ffprobe_frames(p: Path) -> int | None:
    """Frame count from the container header. MPEG-TS recordings (saved as
    .mp4) carry no nb_frames, so fall back to duration x fps -- matches the
    _meta.csv row count exactly on those."""
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,duration,avg_frame_rate",
             "-of", "default=nw=1", str(p)],
            capture_output=True, text=True, timeout=15,
        )
        kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        if kv.get("nb_frames", "").isdigit():
            return int(kv["nb_frames"])
        num, _, den = kv.get("avg_frame_rate", "").partition("/")
        dur = float(kv["duration"])
        return round(dur * int(num) / int(den or 1))
    except Exception:
        return None
