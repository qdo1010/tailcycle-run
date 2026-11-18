"""Bake the canonical world frame into calibs + existing predictions.

For each date with a GOOD alignment transform T (date->canonical, from
align_world_frames.py):

  1. Transform every camera's extrinsics so the calibration expresses cameras
     in the canonical frame. new R_c = R_c @ R_T^T ; new t_c = t_c - R_c' @ t_T.
     This is exactly reprojection-invariant (a world point's 2D projection is
     unchanged), so FUTURE predictions run against it land natively in the
     canonical frame. Applied to red_data OpenCV yamls; jarvis dir is then
     regenerated from them.
  2. Transform every EXISTING prediction's 3D on that date by T (X_canon =
     R_T @ X_date + t_T). Because the calib was transformed consistently,
     canonical-3D reprojected through the aligned calib is pixel-identical to
     before. IK / bad-frames are rigid-invariant, so they need no rerun.

Idempotent: a `.world_aligned` marker (calib dir) and an `aligned_frame:` key
(session info.yaml) prevent double-application. Native copies are backed up
(calibration_dateframe_bak / data3D_dateframe.csv). Bad-solve dates
(status != ok) are SKIPPED and left in their own frame.

Usage: apply_world_alignment.py [--dry-run]
"""
from __future__ import annotations
import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg, path as _cfgp  # noqa: E402
JARVIS_ROOT = _cfgp("local.jarvis_calib_root")
ALIGN_DIR = JARVIS_ROOT / "world_alignment"
CALIB_ROOT = _cfgp("local.opencv_calib_root")
# prediction roots whose stored 3D is transformed with their calibration date
PRED_ROOTS = [Path(p).expanduser() for p in _cfg("local.align_pred_roots", [])]
TOOLS = Path(__file__).resolve().parents[1] / "vendor"


def align_calib(date: str, T: np.ndarray, dry: bool) -> bool:
    cdir = CALIB_ROOT / date / "calibration"
    if not cdir.is_dir():
        return False
    if (cdir / ".world_aligned").exists():
        return False   # already done
    R_T, t_T = T[:3, :3], T[:3, 3]
    yamls = sorted(cdir.glob("Cam*.yaml"))
    if dry:
        print(f"  [dry] would align {len(yamls)} cams in {cdir}")
        return True
    shutil.copytree(cdir, CALIB_ROOT / date / "calibration_dateframe_bak",
                    dirs_exist_ok=True)
    for y in yamls:
        fs = cv2.FileStorage(str(y), 0)
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        rc = fs.getNode("rc_ext").mat(); tc = fs.getNode("tc_ext").mat()
        w = int(fs.getNode("image_width").real())
        h = int(fs.getNode("image_height").real())
        fs.release()
        if rc is None:
            continue
        Rc2 = rc @ R_T.T
        tc2 = tc.reshape(3) - Rc2 @ t_T
        s = cv2.FileStorage(str(y), cv2.FileStorage_WRITE)
        s.write("image_width", w); s.write("image_height", h)
        s.write("camera_matrix", K)
        s.write("distortion_coefficients", D if D is not None else np.zeros((1, 5)))
        s.write("tc_ext", tc2.reshape(3, 1)); s.write("rc_ext", Rc2)
        s.release()
    # Transform the solve's stored 3D landmarks too, else the reproj gate
    # (_calib_reproj_errors projects these through the now-canonical extrinsics)
    # sees a huge mismatch and wrongly refuses every aligned date.
    lg = CALIB_ROOT / date / "output" / "landmarks_global.json"
    if lg.exists():
        d = json.loads(lg.read_text())
        if "landmarks_global" in d and not d.get("_world_aligned"):
            pts = np.array(d["landmarks_global"], float)
            d["landmarks_global"] = ((pts @ R_T.T) + t_T).tolist()
            d["_world_aligned"] = True
            lg.write_text(json.dumps(d))
    (cdir / ".world_aligned").write_text(json.dumps({"T": T.tolist()}))
    # regenerate jarvis from the aligned OpenCV calib
    jd = JARVIS_ROOT / date
    shutil.rmtree(jd, ignore_errors=True)
    subprocess.run([_cfg("local.python", sys.executable),
                    str(TOOLS / "convert_calib_generic.py"), str(cdir), str(jd)],
                   capture_output=True)
    return True


def align_predictions(date: str, T: np.ndarray, dry: bool) -> int:
    R_T, t_T = T[:3, :3], T[:3, 3]
    n = 0
    for root in PRED_ROOTS:
        for info in root.glob(f"*/{date}_*/tailcycle*/info.yaml"):
            sess = info.parent
            txt = info.read_text()
            # skip if already canonical (either transformed here before, or
            # predicted natively against an aligned calib)
            if "aligned_frame: canonical" in txt:
                continue
            csv = sess / "data3D.csv"
            if not csv.exists():
                continue
            if dry:
                print(f"  [dry] would transform {csv}")
                n += 1
                continue
            df = pd.read_csv(csv, header=[0, 1], dtype=np.float32)
            names = list(dict.fromkeys(c[0] for c in df.columns))
            shutil.copyfile(csv, sess / "data3D_dateframe.csv")
            for nm in names:
                xyz = df[[(nm, "x"), (nm, "y"), (nm, "z")]].to_numpy()
                fin = np.isfinite(xyz).all(1)
                xyz[fin] = xyz[fin] @ R_T.T + t_T
                df[(nm, "x")], df[(nm, "y")], df[(nm, "z")] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            df.to_csv(csv, index=False, float_format="%.4f", na_rep="NaN")
            # A human-proofread source (<source>_proofread/, written by RED
            # via /api/session_corrections) keeps its corrections in the
            # prediction's frame too — move them with the CSV, or the next
            # correction rebuild would mix frames.
            cj = sess / "corrections.json"
            if cj.exists():
                corr = json.loads(cj.read_text())
                for kps in corr.get("frames", {}).values():
                    for nm, xyz in kps.items():
                        kps[nm] = [round(float(v), 4) for v in
                                   (np.asarray(xyz, float) @ R_T.T + t_T)]
                cj.write_text(json.dumps(corr, indent=1))
            # replace any existing aligned_frame line, else append
            lines = [l for l in txt.splitlines()
                     if not l.startswith("aligned_frame:")]
            lines.append("aligned_frame: canonical")
            info.write_text("\n".join(lines) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dates", nargs="+", default=None,
                    help="only these YYYY_MM_DD dates (default: every aligned date)")
    args = ap.parse_args()
    cal_done = pred_done = skipped = 0
    for jf in sorted(ALIGN_DIR.glob("*.json")):
        meta = json.loads(jf.read_text())
        date, status = meta["date"], meta.get("status")
        if args.dates and date not in args.dates:
            continue
        if status not in ("ok", "reference"):
            print(f"{date}: {status} — SKIP (left in own frame)")
            skipped += 1
            continue
        T = np.array(meta["T"])
        if align_calib(date, T, args.dry_run):
            cal_done += 1
        p = align_predictions(date, T, args.dry_run)
        pred_done += p
        if p:
            print(f"{date}: calib aligned, {p} predictions -> canonical "
                  f"(resid {meta['median_resid_mm']:.1f}mm)")
    print(f"\n{'[dry] ' if args.dry_run else ''}calibs aligned: {cal_done}, "
          f"predictions transformed: {pred_done}, dates skipped: {skipped}")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
