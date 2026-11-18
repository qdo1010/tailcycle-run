"""Convert one of our mouse recording sessions into a tailcycle-dataset
inference input, so tailcyclenet's `scripts/infer.py` can consume it.

Input (either Free or Climb tab):
  <video_root>/<animal>/<session>/Cam*.mp4        # 16 videos
  <opencv_calib_root>/<date>/calibration/   # 16 OpenCV Cam*.yaml

Output (a tailcycle single-session dataset at --out):
  <out>/                              (a tailcycle "session directory")
    session.toml                      keypoint identity + labels="tracked"
    calibration.toml                  aniposelib CameraGroup + offset/moving
    groups.pq                         one row for the whole recording
    groups/000/
      Cam*.mp4                        symlinks to source Cam*.mp4

`--out` is the SESSION dir directly (not a dataset root). tailcyclenet's
`sessions_for()` treats a path containing session.toml as one session.

Notes on the format transformation:
- Our `Cam*.yaml` is OpenCV FileStorage: `camera_matrix` is 3x3, but JARVIS's
  own convention stores it as intrinsicMatrix.T. Here we output aniposelib's
  standard [[fx,0,cx],[0,fy,cy],[0,0,1]] — un-transpose if needed.
- `rc_ext` is a 3x3 rotation matrix; aniposelib wants a Rodrigues 3-vector.
- All 5 distortion coefficients are used (the reference convert_johnson.py comment:
  truncating to k1,k2 costs 7.7 px on 2025_10_20).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SESSION_KEYPOINTS = [
    "Snout", "EarL", "EarR", "Neck", "SpineL", "TailBase",
    "ShoulderL", "ElbowL", "WristL", "HandL",
    "ShoulderR", "ElbowR", "WristR", "HandR",
    "KneeL", "AnkleL", "FootL",
    "KneeR", "AnkleR", "FootR",
    "TailTip", "TailMid", "Tail1Q", "Tail3Q",
]

SESSION_SKELETON = [
    ["Snout", "EarL"], ["Snout", "EarR"], ["EarL", "Neck"], ["EarR", "Neck"],
    ["Neck", "SpineL"], ["SpineL", "TailBase"],
    ["Neck", "ShoulderL"], ["ShoulderL", "ElbowL"], ["ElbowL", "WristL"], ["WristL", "HandL"],
    ["Neck", "ShoulderR"], ["ShoulderR", "ElbowR"], ["ElbowR", "WristR"], ["WristR", "HandR"],
    ["SpineL", "KneeL"], ["KneeL", "AnkleL"], ["AnkleL", "FootL"],
    ["SpineL", "KneeR"], ["KneeR", "AnkleR"], ["AnkleR", "FootR"],
    ["TailBase", "Tail1Q"], ["Tail1Q", "TailMid"], ["TailMid", "Tail3Q"], ["Tail3Q", "TailTip"],
]

FLIP_PAIRS = [
    ["EarL", "EarR"], ["ShoulderL", "ShoulderR"], ["ElbowL", "ElbowR"],
    ["WristL", "WristR"], ["HandL", "HandR"], ["KneeL", "KneeR"],
    ["AnkleL", "AnkleR"], ["FootL", "FootR"],
]

SESSION_RE = re.compile(r"^(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}$")


def _date_of(session_name: str) -> str:
    m = SESSION_RE.match(session_name)
    if not m:
        raise SystemExit(f"unrecognised session name: {session_name!r}")
    return m.group(1)


def _load_opencv_calib(yaml_path: Path) -> dict:
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    W = int(fs.getNode("image_width").real()) if not fs.getNode("image_width").empty() else None
    H = int(fs.getNode("image_height").real()) if not fs.getNode("image_height").empty() else None
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rc_ext").mat()
    T = fs.getNode("tc_ext").mat()
    fs.release()
    if K is None or R is None or T is None:
        raise SystemExit(f"missing K/R/T in {yaml_path}")
    return {"size": (W, H), "K": K, "D": D, "R": R, "T": T}


def _rmat_to_rvec(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    return rvec.ravel()


def _cam_serial_from_filename(mp4: Path) -> str:
    """Cam2005325.mp4 -> 2005325."""
    m = re.match(r"Cam(\d+)\.mp4$", mp4.name)
    if not m:
        raise SystemExit(f"unexpected video filename: {mp4.name}")
    return m.group(1)


def _frame_count(mp4: Path) -> int:
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _video_size(mp4: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(mp4))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def _emit_calibration_toml(out: Path, calibs: dict[str, dict]) -> None:
    """Emit aniposelib-format calibration.toml. Cams are sorted by serial for
    reproducibility; the pose model treats cameras as a bag, so ordering is
    inference-side only (matches how convert_johnson.py does it)."""
    lines = []
    for i, serial in enumerate(sorted(calibs.keys())):
        c    = calibs[serial]
        K    = c["K"]
        D    = c["D"].ravel()
        rvec = _rmat_to_rvec(c["R"])
        tvec = c["T"].ravel()
        W, H = c["size"]
        lines.append(f"[cam_{i}]")
        lines.append(f'name = "Cam{serial}"')
        lines.append(f"size = [ {W}, {H},]")
        lines.append("matrix = [ "
                     f"[ {K[0,0]}, {K[0,1]}, {K[0,2]},], "
                     f"[ {K[1,0]}, {K[1,1]}, {K[1,2]},], "
                     f"[ {K[2,0]}, {K[2,1]}, {K[2,2]},],]")
        # aniposelib expects distortions as a length-5 list. Include all 5;
        # per the reference convert_johnson.py, truncating to k1,k2 costs 7.7 px.
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
    (out / "calibration.toml").write_text("\n".join(lines))


def _emit_session_toml(out: Path, source: str) -> None:
    lines = [
        'mode = "3d"',
        'units = "mm"',
        'labels = "tracked"',
        f'names = [ {", ".join(f"{n!r}" for n in SESSION_KEYPOINTS)},]',
        f'skeleton = [ {", ".join("[ " + ", ".join(f"{n!r}" for n in edge) + ",]" for edge in SESSION_SKELETON)},]',
        f'flip_pairs = [ {", ".join("[ " + ", ".join(f"{n!r}" for n in pair) + ",]" for pair in FLIP_PAIRS)},]',
        "",
        "[provenance]",
        f'source = "{source}"',
        'annotator = ""',
        'annotator_tool = "tailcycle_run/cluster/session_to_tailcycle.py"',
        'animal_id_source = "single animal per session"',
        "",
    ]
    (out / "session.toml").write_text("\n".join(lines))


def _emit_groups_pq(out: Path, group_id: str, n_frames: int) -> None:
    """One group covering the whole recording. tailcycle windowing runs
    internally; we just declare the one contiguous clip."""
    df = pd.DataFrame({"group_id": [group_id], "n_frames": [n_frames]})
    df.to_parquet(out / "groups.pq", index=False)


def convert(video_dir: Path, calib_dir: Path, out: Path,
             source_label: str, group_id: str = "000") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "groups").mkdir(exist_ok=True)
    group_dir = out / "groups" / group_id
    group_dir.mkdir(exist_ok=True)

    # Symlink Cam*.mp4 (per the format spec: "a converter is encouraged to
    # symlink whole directories rather than copy or link individual frames").
    mp4s = sorted(video_dir.glob("Cam*.mp4"))
    if not mp4s:
        raise SystemExit(f"no Cam*.mp4 in {video_dir}")
    for mp4 in mp4s:
        link = group_dir / mp4.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(mp4.resolve())

    # Load matching Cam*.yaml for each Cam*.mp4 (dropping any cam with no calib).
    calibs = {}
    calib_sizes = {}
    for mp4 in mp4s:
        serial = _cam_serial_from_filename(mp4)
        yaml_path = calib_dir / f"Cam{serial}.yaml"
        if not yaml_path.exists():
            print(f"[convert] WARN {yaml_path} missing — dropping Cam{serial} "
                  f"from the tailcycle input", file=sys.stderr)
            (group_dir / mp4.name).unlink(missing_ok=True)
            continue
        c = _load_opencv_calib(yaml_path)
        # Video vs calib size sanity check; calib usually stores 3208x2200
        # which matches the sensor -- videos might be downscaled. Prefer the
        # video's actual dimensions in calibration.toml `size`.
        vw, vh = _video_size(mp4)
        calib_sizes[serial] = (vw, vh)
        c["size"] = (vw, vh)
        calibs[serial] = c
    if len(calibs) < 2:
        raise SystemExit(f"only {len(calibs)} cam(s) with calib — need at least 2")

    _emit_calibration_toml(out, calibs)
    _emit_session_toml(out, source=source_label)

    # groups.pq — one row for the whole recording (n_frames from first cam).
    first_mp4 = next(iter(sorted(group_dir.glob("Cam*.mp4"))))
    n_frames  = _frame_count(first_mp4)
    _emit_groups_pq(out, group_id=group_id, n_frames=n_frames)

    print(f"[convert] built tailcycle session at {out}")
    print(f"          cams: {len(calibs)}   frames: {n_frames}   group_id: {group_id}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-dir", required=True, type=Path,
                    help="dir with 16 Cam*.mp4 (e.g. <video_root>/<animal>/<session>)")
    ap.add_argument("--calib-dir", required=True, type=Path,
                    help="dir with 16 Cam*.yaml (OpenCV, e.g. <opencv_calib_root>/<date>/calibration)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output tailcycle session dir")
    ap.add_argument("--source-label", default="",
                    help="text stamped into session.toml [provenance].source")
    args = ap.parse_args()
    convert(args.video_dir, args.calib_dir, args.out,
             source_label=args.source_label or str(args.video_dir))
