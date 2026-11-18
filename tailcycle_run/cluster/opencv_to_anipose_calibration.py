"""Convert our OpenCV `Cam*.yaml` calibration dir to a single
aniposelib-format `calibration.toml` that the NEW tailcyclenet inference
pipeline (`--videos` + `--calibration`) reads directly.

This replaces the old `session_to_tailcycle.py` flow which built a whole
tailcycle-format session directory (session.toml + calibration.toml +
groups.pq + symlinked videos). The new tailcyclenet reads raw videos in
memory and only needs the calibration.toml.

Input:  <calib-dir>/Cam*.yaml   (OpenCV FileStorage; our red_data format)
Output: <out.toml>              (aniposelib `[cam_i]` blocks)

Uses the same conversion code as before -- just skips the .pq/session.toml/
symlink emission.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_calib(yaml_path: Path) -> dict:
    """Read a per-camera calib yaml in EITHER format and return the OpenCV
    convention (K, D, R=world→cam, T). Two formats occur on disk:

      OpenCV (red_data/calib/<date>/calibration/): camera_matrix (=K),
        distortion_coefficients, rc_ext (=R), tc_ext (=T).
      JARVIS (jarvis_calibrations/<date>/): intrinsicMatrix (=Kᵀ),
        distortionCoefficients (1×5), R (=Rᵀ), T. Produced by
        convert_calib_generic.py, which transposes K and R — so we
        transpose them back here.
    """
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    W = int(fs.getNode("image_width").real()) if not fs.getNode("image_width").empty() else 3208
    H = int(fs.getNode("image_height").real()) if not fs.getNode("image_height").empty() else 2200
    if not fs.getNode("camera_matrix").empty():
        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        R = fs.getNode("rc_ext").mat()
        T = fs.getNode("tc_ext").mat()
    else:                                   # JARVIS format
        K = fs.getNode("intrinsicMatrix").mat().T          # Kᵀ → K
        D = fs.getNode("distortionCoefficients").mat()
        R = fs.getNode("R").mat().T                        # Rᵀ → R
        T = fs.getNode("T").mat()
    fs.release()
    if K is None or R is None or T is None:
        raise SystemExit(f"missing K/R/T in {yaml_path}")
    return {"size": (W, H), "K": K, "D": D, "R": R, "T": T}


def _cam_serial(yaml_path: Path) -> str:
    m = re.match(r"Cam(\d+)\.yaml$", yaml_path.name)
    if not m:
        raise SystemExit(f"unexpected calib filename: {yaml_path.name}")
    return m.group(1)


def build(calib_dir: Path, out_toml: Path,
          exclude: set[str] | None = None) -> Path:
    """exclude: camera names (e.g. {'Cam2006515'}) to leave OUT of the toml.
    tailcyclenet matches videos to calibration by name, so a camera absent
    from the toml is cleanly ignored — this is how blurry/broken cameras are
    dropped from triangulation per session."""
    yamls = sorted(calib_dir.glob("Cam*.yaml"))
    if exclude:
        dropped = [y.stem for y in yamls if y.stem in exclude]
        if dropped:
            print(f"[calib-toml] excluding {len(dropped)} cams: {dropped}")
        yamls = [y for y in yamls if y.stem not in exclude]
    if not yamls:
        raise SystemExit(f"no Cam*.yaml in {calib_dir}")

    lines = []
    for i, y in enumerate(yamls):
        c    = _load_calib(y)
        K    = c["K"]
        D    = c["D"].ravel()
        rvec = cv2.Rodrigues(np.asarray(c["R"], dtype=np.float64))[0].ravel()
        tvec = c["T"].ravel()
        W, H = c["size"]
        serial = _cam_serial(y)
        lines.append(f"[cam_{i}]")
        lines.append(f'name = "Cam{serial}"')
        lines.append(f"size = [ {W}, {H},]")
        lines.append("matrix = [ "
                     f"[ {K[0,0]}, {K[0,1]}, {K[0,2]},], "
                     f"[ {K[1,0]}, {K[1,1]}, {K[1,2]},], "
                     f"[ {K[2,0]}, {K[2,1]}, {K[2,2]},],]")
        dd = list(D) + [0.0] * max(0, 5 - len(D))
        lines.append(f"distortions = [ {dd[0]}, {dd[1]}, {dd[2]}, {dd[3]}, {dd[4]},]")
        lines.append(f"rotation = [ {rvec[0]}, {rvec[1]}, {rvec[2]},]")
        lines.append(f"translation = [ {tvec[0]}, {tvec[1]}, {tvec[2]},]")
        lines.append("n_dist = 5")
        lines.append("offset = [ 0.0, 0.0,]")
        lines.append("moving = false")
        lines.append("")
    lines.append("[metadata]")
    lines.append("")
    out_toml.parent.mkdir(parents=True, exist_ok=True)
    out_toml.write_text("\n".join(lines))
    print(f"[calib-toml] wrote {out_toml}  ({len(yamls)} cameras)")
    return out_toml


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True, type=Path,
                    help="dir with 16 Cam*.yaml (OpenCV FileStorage)")
    ap.add_argument("--out", required=True, type=Path,
                    help="path for the aniposelib calibration.toml")
    ap.add_argument("--exclude", default="",
                    help="comma-separated camera names to leave out of the "
                          "toml (e.g. Cam2006515,Cam2002487)")
    args = ap.parse_args()
    excl = {c.strip() for c in args.exclude.split(",") if c.strip()}
    build(args.calib_dir, args.out, exclude=excl or None)
