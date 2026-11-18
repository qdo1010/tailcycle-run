#!/usr/bin/env python3
"""Per-frame body span for 47-kp tailcycle output (v2 has no quality score;
this is the exploded-skeleton filter's input).

Writes <session>/<source>/body_span.csv:
    frame, span_mm, span_notail_mm, n_kp
  span_mm         max pairwise distance over all valid keypoints
  span_notail_mm  same without tail-1/3, tail-2/3, tail-end (a normal mouse
                  without tail is < ~120 mm; a few hundred mm = exploded)
  n_kp            valid keypoints in the frame
NaN rows where the frame has < 2 keypoints.

    body_span.py --pred-root <pred_root> [--source tailcycle47_v2] [--force]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from io3d import load_any  # noqa: E402

TAIL = {"tail-1/3", "tail-2/3", "tail-end"}


def max_pairwise(x: np.ndarray) -> np.ndarray:
    """x: F x K x 3 with NaN for missing -> F max pairwise distances."""
    out = np.full(len(x), np.nan, np.float32)
    for i in range(0, len(x), 5000):
        c = x[i:i + 5000]
        d = np.linalg.norm(c[:, :, None, :] - c[:, None, :, :], axis=-1)
        with np.errstate(invalid="ignore"):
            out[i:i + 5000] = np.nanmax(d.reshape(len(c), -1), axis=1)
    return out


def session(csv: Path, force: bool) -> str:
    out = csv.with_name("body_span.csv")
    if out.exists() and out.stat().st_mtime > csv.stat().st_mtime and not force:
        return "up to date"
    head, data = load_any(csv)
    names = head[0].rstrip("\n").split(",")[::4]
    xyz = data[:, :, :3]
    notail = [i for i, n in enumerate(names) if n not in TAIL]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s_all = max_pairwise(xyz)
        s_nt = max_pairwise(xyz[:, notail])
    n = (~np.isnan(xyz[:, :, 0])).sum(1)
    tmp = out.with_suffix(".tmp")
    np.savetxt(tmp, np.column_stack([np.arange(len(n)), s_all, s_nt, n]),
               delimiter=",", fmt=["%d", "%.1f", "%.1f", "%d"],
               header="frame,span_mm,span_notail_mm,n_kp", comments="")
    tmp.replace(out)
    ok = s_nt[n >= 2]
    return (f"{len(n):,} frames, span_notail median {np.nanmedian(ok):.0f} mm, "
            f">200 mm in {100 * np.mean(ok > 200):.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", type=Path, required=True)
    ap.add_argument("--source", default="tailcycle47_v2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for csv in sorted(a.pred_root.glob(f"*/*/{a.source}/data3D.csv")):
        print(f"{csv.parent.parent.parent.name}/{csv.parent.parent.name}: "
              f"{session(csv, a.force)}", flush=True)


if __name__ == "__main__":
    main()
