"""Cross-day world-frame alignment.

Each calibration day defines its own world frame from wherever the static
charuco board happened to sit (position + rotation vary day to day), so 3D
predictions are NOT comparable across dates even though the physical camera
rig is fixed. Because the rig IS fixed, each date's camera constellation is
the same set of points up to a rigid transform — so we can recover, per date,
the rigid transform T that maps that date's world frame onto a single
CANONICAL frame, by fitting the camera centers.

Measured: with a rigid (no-scale) fit + robust outlier rejection, clean dates
align to the canonical frame at 2-4 mm median residual (the rig is stable to
a few mm). A date that will NOT align rigidly (all cameras off by >>10 mm,
e.g. 2026_09_04) has a BAD SOLVE, not a moved rig — flagged for re-solve.

This is NON-DESTRUCTIVE: it writes a per-date 4x4 transform to
world_alignment/<date>.json. Native predictions stay in their own frame (so
reprojection + IK keep working); cross-day analysis applies T on demand via
canonical_xyz().

Usage:
  align_world_frames.py [--reference 2026_08_05] [--apply-report]
"""
from __future__ import annotations
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg, path as _cfgp  # noqa: E402
CALIB_ROOT = _cfgp("local.opencv_calib_root")
OUT_DIR = _cfgp("local.jarvis_calib_root") / "world_alignment"
DEFAULT_REF = "2026_08_05"
INLIER_TOL_MM = 15.0        # a camera further than this after fit = outlier
MAX_ALIGN_RESID_MM = 15.0   # date median above this = solve suspect


def cam_centers(date: str) -> dict[str, np.ndarray]:
    out = {}
    for y in sorted(glob.glob(str(CALIB_ROOT / date / "calibration" / "Cam*.yaml"))):
        fs = cv2.FileStorage(y, 0)
        rc, tc = fs.getNode("rc_ext").mat(), fs.getNode("tc_ext").mat()
        fs.release()
        if rc is not None and tc is not None:
            out[Path(y).stem] = (-rc.T @ tc.reshape(3, 1)).ravel()
    return out


def rigid_fit(A: np.ndarray, B: np.ndarray):
    """B ~ R@A + t (no scale). Returns R(3x3), t(3,), per-point residual."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(Ac.T @ Bc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = B.mean(0) - R @ A.mean(0)
    return R, t, np.linalg.norm((A @ R.T + t) - B, axis=1)


def robust_align(C: dict, C_ref: dict):
    """Rigid transform date->reference with iterative outlier rejection.
    Returns (T 4x4, inlier_cams, median_resid, outliers) or None."""
    common = sorted(set(C) & set(C_ref))
    if len(common) < 4:
        return None
    A = np.array([C[c] for c in common])
    B = np.array([C_ref[c] for c in common])
    keep = np.ones(len(common), bool)
    R = np.eye(3); t = np.zeros(3)
    for _ in range(4):
        R, t, _ = rigid_fit(A[keep], B[keep])
        allres = np.linalg.norm((A @ R.T + t) - B, axis=1)
        thr = max(INLIER_TOL_MM, 3 * np.median(allres[keep]))
        keep = allres < thr
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    inliers = [common[i] for i in range(len(common)) if keep[i]]
    outliers = [common[i] for i in range(len(common)) if not keep[i]]
    return T, inliers, float(np.median(allres[keep])), outliers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=DEFAULT_REF)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    C_ref = cam_centers(args.reference)
    print(f"canonical reference: {args.reference} ({len(C_ref)} cams)")
    # reference maps to itself
    (OUT_DIR / f"{args.reference}.json").write_text(json.dumps({
        "date": args.reference, "reference": args.reference,
        "T": np.eye(4).tolist(), "median_resid_mm": 0.0,
        "n_inliers": len(C_ref), "outliers": [], "status": "reference"}))

    dates = sorted(d.name for d in CALIB_ROOT.iterdir()
                   if re.fullmatch(r"\d{4}_\d{2}_\d{2}", d.name)
                   and (d / "calibration").is_dir())
    ok = suspect = 0
    for d in dates:
        if d == args.reference:
            continue
        C = cam_centers(d)
        r = robust_align(C, C_ref)
        if r is None:
            print(f"  {d}: too few shared cams — SKIP")
            continue
        T, inliers, med, outliers = r
        status = "ok" if med <= MAX_ALIGN_RESID_MM else "SOLVE-SUSPECT"
        if med <= MAX_ALIGN_RESID_MM:
            ok += 1
        else:
            suspect += 1
        (OUT_DIR / f"{d}.json").write_text(json.dumps({
            "date": d, "reference": args.reference, "T": T.tolist(),
            "median_resid_mm": med, "n_inliers": len(inliers),
            "outliers": outliers, "status": status}))
        flag = "" if status == "ok" else "   <<< bad solve, re-solve this date"
        bumped = f" bumped={outliers}" if outliers else ""
        print(f"  {d}: {med:5.1f}mm  {len(inliers)} inliers{bumped}{flag}")
    print(f"\n{ok} aligned, {suspect} solve-suspect. transforms -> {OUT_DIR}")


def canonical_xyz(xyz: np.ndarray, date: str) -> np.ndarray:
    """Apply a date's stored transform to (N,3) mm points -> canonical frame.
    Returns points unchanged if no transform exists."""
    p = OUT_DIR / f"{date}.json"
    if not p.exists():
        return xyz
    T = np.array(json.loads(p.read_text())["T"])
    return (xyz @ T[:3, :3].T) + T[:3, 3]


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
