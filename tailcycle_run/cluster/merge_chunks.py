"""Concatenate every per-chunk session directory produced by predict_chunk.sh
into ONE JARVIS-shaped data3D.csv covering the whole recording.

Each chunk writes a tailcycle-format session dir. points3d.pq schema
(verified against a real run):
    group_id:    category
    frame:       int32     -- SOURCE frame index (start-frame offset already applied)
    animal_id:   category  -- 'det00', 'det01', ...
    bodypart:    category  -- keypoint NAME ('Snout', 'EarL', ...)
    status:      category  -- 'visible' | ...
    x, y, z:     float32
    score:       float32
    score_logit: float32

Because `frame` is the global source index and `bodypart` is the keypoint
name directly, the merge is a straight concat + pivot -- no kp-id lookup,
no per-chunk offset math needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_to_tailcycle import SESSION_KEYPOINTS  # noqa: E402

# 47-kp allen-mouse schema (the mega-mouse multi-dataset model run with
# --dataset-name allen-mouse-combined). Order = the model's
# keypoint_registry order with the "allen-mouse-combined-" prefix stripped,
# which is how bodypart names appear in points3d.pq.
ALLEN_KEYPOINTS = [
    "nose", "L-ear", "R-ear", "neck", "spine-mid", "tail-base",
    "tail-1/3", "tail-2/3", "tail-end",
    "LF-shoulder", "LF-elbow", "LF-wrist", "LF-index", "LF-middle",
    "LF-ring", "LF-pinkie", "LF-index-base", "LF-pinkie-base",
    "LH-hip", "LH-knee", "LH-wrist", "LH-thumb", "LH-index", "LH-middle",
    "LH-ring", "LH-pinkie", "LH-thumb-base", "LH-pinkie-base",
    "RF-shoulder", "RF-elbow", "RF-wrist", "RF-index", "RF-middle",
    "RF-ring", "RF-pinkie", "RF-index-base", "RF-pinkie-base",
    "RH-hip", "RH-knee", "RH-wrist", "RH-thumb", "RH-index", "RH-middle",
    "RH-ring", "RH-pinkie", "RH-thumb-base", "RH-pinkie-base",
]

KP_SETS = {"johnson": SESSION_KEYPOINTS, "allen": ALLEN_KEYPOINTS}


def merge(chunks_dir: Path, num_chunks: int, out_csv: Path,
          kp_set: str = "johnson") -> Path:
    keypoints = KP_SETS[kp_set]
    frames = []
    for i in range(1, num_chunks + 1):
        pq = chunks_dir / f"chunk_{i:03d}" / "points3d.pq"
        # .range is written only after infer exits 0; a points3d.pq without
        # it is a partial left by a crashed chunk, not a result.
        if not (pq.exists() and (chunks_dir / f"chunk_{i:03d}.range").exists()):
            # chunk ran but had zero detections (dark/empty segment) — the
            # worker leaves a .EMPTY marker. Skip it; its frames stay NaN.
            if (chunks_dir / f"chunk_{i:03d}.EMPTY").exists():
                print(f"[merge] chunk_{i:03d} empty (no detections) -- "
                      f"frames stay NaN")
                continue
            raise SystemExit(f"missing {pq}")
        frames.append(pd.read_parquet(pq))
    if not frames:
        raise SystemExit("every chunk is empty -- nothing to merge")
    df = pd.concat(frames, ignore_index=True)

    # Single-mouse rig: keep the first animal id (lexicographically 'det00').
    animals = sorted(df["animal_id"].astype(str).unique())
    if len(animals) > 1:
        print(f"[merge] {len(animals)} animals present {animals}; keeping "
              f"{animals[0]} (single-mouse rig)")
    df = df[df["animal_id"].astype(str) == animals[0]]

    df["bodypart"] = df["bodypart"].astype(str)
    seen = set(df["bodypart"].unique())
    unknown = seen - set(keypoints)
    if unknown:
        print(f"[merge] WARNING: {len(unknown)} bodyparts not in the "
              f"'{kp_set}' kp set are dropped: {sorted(unknown)[:6]}…")
    F_total = int(df["frame"].max()) + 1
    K = len(keypoints)
    kp_col = {name: j for j, name in enumerate(keypoints)}

    pts  = np.full((F_total, K, 3), np.nan, dtype=np.float32)
    conf = np.full((F_total, K),    np.nan, dtype=np.float32)

    f = df["frame"].to_numpy(dtype=np.int64)
    k = df["bodypart"].map(kp_col).to_numpy()          # NaN for unknown names
    keep = ~pd.isna(k)
    f, kk = f[keep], k[keep].astype(np.int64)
    pts[f, kk, 0] = df["x"].to_numpy()[keep]
    pts[f, kk, 1] = df["y"].to_numpy()[keep]
    pts[f, kk, 2] = df["z"].to_numpy()[keep]
    conf[f, kk]   = df["score"].to_numpy()[keep] if "score" in df.columns else 1.0

    cols_row0, cols_row1 = [], []
    for kp in keypoints:
        cols_row0 += [kp, kp, kp, kp]
        cols_row1 += ["x", "y", "z", "confidence"]
    data = np.empty((F_total, 4 * K), dtype=np.float32)
    for j in range(K):
        data[:, 4*j+0] = pts[:, j, 0]
        data[:, 4*j+1] = pts[:, j, 1]
        data[:, 4*j+2] = pts[:, j, 2]
        data[:, 4*j+3] = conf[:, j]
    out = pd.DataFrame(data, columns=pd.MultiIndex.from_arrays([cols_row0, cols_row1]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.4f", na_rep="NaN")
    print(f"merged {num_chunks} chunks -> {out_csv}  ({F_total:,} frames, {K} kps)")

    # Scorer output (tailcyclenet infer --scorer). Despite "framewise" in the
    # docs, quality_score is PER KEYPOINT (24 distinct values per frame,
    # measured 2026-10-06: e.g. TailMid -0.09 while the spine is ~1.1), so keep
    # every keypoint — a per-frame mean would hide the weak ones. Side file so
    # data3D.csv keeps its JARVIS schema. Columns: frame, quality_mean,
    # quality_min, worst_kp, precision_mean, then q_<kp> per keypoint. Model
    # outputs, not calibrated probabilities; poses are not filtered by them.
    if "quality_score" in df.columns:
        qs = np.full((F_total, K), np.nan, dtype=np.float32)
        qs[f, kk] = df["quality_score"].to_numpy()[keep]
        import warnings
        with warnings.catch_warnings():     # unscored frames are all-NaN rows
            warnings.simplefilter("ignore", RuntimeWarning)
            q = pd.DataFrame({
                "quality_mean": np.nanmean(qs, axis=1),
                "quality_min":  np.nanmin(qs, axis=1),
            }, index=pd.RangeIndex(F_total, name="frame"))
        has = ~np.isnan(qs).all(axis=1)
        worst = np.full(F_total, "", dtype=object)
        worst[has] = np.array(keypoints, dtype=object)[
            np.nanargmin(np.where(np.isnan(qs[has]), np.inf, qs[has]), axis=1)]
        q["worst_kp"] = worst
        if "quality_precision" in df.columns:
            qp = np.full((F_total, K), np.nan, dtype=np.float32)
            qp[f, kk] = df["quality_precision"].to_numpy()[keep]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q["precision_mean"] = np.nanmean(qp, axis=1)
        for j, kp in enumerate(keypoints):
            q[f"q_{kp}"] = qs[:, j]
        q_csv = out_csv.with_name("quality.csv")
        q.to_csv(q_csv, float_format="%.4f", na_rep="NaN")
        print(f"scorer quality -> {q_csv}  ({int(has.sum()):,} scored frames, "
              f"{K} kp)")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-dir", required=True, type=Path)
    ap.add_argument("--num-chunks", required=True, type=int)
    ap.add_argument("--out-csv",    required=True, type=Path)
    ap.add_argument("--kp-set", choices=sorted(KP_SETS), default="johnson",
                    help="keypoint schema of the model output: johnson (24) "
                          "or allen (47, mega-mouse model)")
    args = ap.parse_args()
    merge(args.chunks_dir, args.num_chunks, args.out_csv, kp_set=args.kp_set)
