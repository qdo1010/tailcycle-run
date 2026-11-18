#!/bin/bash
# Cluster-side per-chunk worker for tailcyclenet inference (NEW pipeline).
#
# Replaces the old flow (ffmpeg-trim + build session dir + infer.py --data)
# with tailcyclenet's own --videos/--calibration/--start-frame/--end-frame
# path, which reads raw video containers through PyAV IN MEMORY. Wins:
#   - no ffmpeg step (~30 s/chunk × K = several minutes saved)
#   - no session-dir dance on scratch
#   - PyAV decoding avoids the decord "container into RAM" pattern that
#     OOM-killed our 16-cam runs at 120 GB on gpu_l4 (see tailcyclenet
#     commit 0127511)
#   - --max-ram 24 sets an actual host-RAM ceiling
#
# Args (positional):
#   1  STAGED_ID    dir name under $CF/predictions/ (output identity)
#   2  NUM_CHUNKS   how many chunks the recording is divided into
#   3  CALIB_DIR    date name under $CF/tailcycle_calibs/  (holds calibration.toml)
#   4  DATASET_NAME registry dataset (e.g. johnson-mouse-combined-aug, or
#                   allen-mouse-combined for the 47-kp mega model)
#   5  MODEL_OVERRIDE optional: full path to an alternate model run dir
#                   (e.g. the 47-kp mega-mouse model). Empty/absent = default.
#   6  VIDEO_ID     optional: dir name under $CF/tailcycle_videos/ to read
#                   videos from. Defaults to STAGED_ID — pass explicitly when
#                   two model variants share one staged upload.
#   7  DETECTOR_OVERRIDE optional: detector run folder (e.g. the universal
#                   mouse detector). Empty/absent = the johnson v3best detector.
#                   A run folder resolves to its latest detector_it*.pth.
#   8  SCORER_RUN   optional: framewise scorer run folder. When set, infer adds
#                   quality_score / quality_precision to points3d.pq (poses are
#                   not changed); merge writes them to quality.csv. The scorer
#                   must share the pose run's video encoder.
#   9  REPO_OVERRIDE optional: tailcyclenet checkout to run (default
#                   $HOME/tailcyclenet). A model trained on newer code (e.g. the
#                   univ-mouse-v2 export) runs from its own checkout so the v1
#                   production checkout is never touched.
#  10  DET_SCORE_OVERRIDE optional: detector objectness floor (default 0.5,
#                   0.1 for allen); per-detector, set by the variant.
#
# Env:
#   LSB_JOBINDEX   LSF array index [1..NUM_CHUNKS]
#
# Cluster layout (relative to $CF = the work dir):
#   $CF/tailcycle_videos/$STAGED_ID/Cam*.mp4                   (uploaded once)
#   $CF/tailcycle_calibs/$CALIB_DIR/calibration.toml           (uploaded once)
#   $CF/predictions/$STAGED_ID/chunks/chunk_NNN/               (this script writes here)
#     session.toml + calibration.toml + groups.pq +
#     points3d.pq + keypoints.pq + instances.pq + windows.pq
#
set -uo pipefail

STAGED_ID=$1; NUM_CHUNKS=$2; CALIB_DIR=$3; DATASET_NAME=$4
MODEL_OVERRIDE=${5:-}; VIDEO_ID=${6:-$STAGED_ID}
DETECTOR_OVERRIDE=${7:-}; SCORER_RUN=${8:-}
REPO_OVERRIDE=${9:-}; DET_SCORE_OVERRIDE=${10:-}
i=${LSB_JOBINDEX:-1}; [ "$i" = "0" ] && i=1

# site paths come from cluster_run (_site_env): TCR_WORK, TCR_REPO, TCR_PIXI,
# TCR_DEFAULT_MODEL, TCR_DEFAULT_DETECTOR
CF=${TCR_WORK:-$HOME/mouse_cluster}
REPO=${REPO_OVERRIDE:-${TCR_REPO:-$HOME/tailcyclenet}}
PIXI=${TCR_PIXI:-$HOME/.pixi/bin/pixi}
export PIXI_FROZEN=true   # never rewrite pixi.lock: concurrent array jobs corrupted it (2026-11-12)
REC=$CF/tailcycle_videos/$VIDEO_ID
CALIB=$CF/tailcycle_calibs/$CALIB_DIR/calibration.toml
CHUNKS=$CF/predictions/$STAGED_ID/chunks
OUT_DIR=$CHUNKS/chunk_$(printf '%03d' "$i")
# 500k-repaired: user's own training run (dataset mouse-annotated-aug, same
# 24 kp names/order as johnson-mouse-combined-aug; trained on v0.1.1 code)
MODEL_RUN=${MODEL_OVERRIDE:-$TCR_DEFAULT_MODEL}
DETECTOR_RUN=${DETECTOR_OVERRIDE:-$TCR_DEFAULT_DETECTOR}

# The detector is trained on johnson-mouse-combined-aug. Any other dataset
# name (mouse-annotated-aug = the 500k-repaired 24-kp model's registry name;
# allen-mouse-combined = the mega 47-kp output set) needs the explicit
# transfer flag — the videos are johnson-rig either way, the detector's true
# domain. Only the ALLEN case needs the lower score threshold (its transfer
# confidence measured near zero at 0.5); same-rig johnson-family models keep
# the normal 0.5.
DET_SCORE=0.5
EXTRA_FLAGS=""
if [ "$DATASET_NAME" != "johnson-mouse-combined-aug" ]; then
  EXTRA_FLAGS="--allow-detector-transfer"
fi
if [ "$DATASET_NAME" = "allen-mouse-combined" ]; then
  DET_SCORE=0.1
fi
[ -n "$DET_SCORE_OVERRIDE" ] && DET_SCORE=$DET_SCORE_OVERRIDE

mkdir -p "$CHUNKS"
# In smoke mode the driver polls for smoke.FAILED; otherwise chunk_NNN.FAILED.
fail() {
  if [ -n "${TAILCYCLE_SMOKE_END:-}" ]; then
    echo "$1" > "$CHUNKS/smoke.FAILED"
  else
    echo "$1" > "$CHUNKS/chunk_$(printf '%03d' "$i").FAILED"
  fi
  echo "FAILED chunk $i: $1"; exit 1
}

# A run folder or a self-contained packaged .pth (tailcyclenet package_checkpoint.py).
[ -e "$MODEL_RUN" ] || fail "no model run dir or .pth at $MODEL_RUN"
[ -e "$DETECTOR_RUN" ] || fail "no detector run dir or .pth at $DETECTOR_RUN"
[ -d "$REPO" ] || fail "no tailcyclenet checkout at $REPO"
[ -z "$SCORER_RUN" ] || [ -d "$SCORER_RUN" ] || fail "no scorer dir at $SCORER_RUN"
[ -f "$CALIB" ] || fail "no calibration.toml at $CALIB"

FIRSTCAM=$(ls "$REC"/Cam*.mp4 2>/dev/null | sort | head -1)
[ -n "$FIRSTCAM" ] || fail "no Cam*.mp4 in $REC"

# Total frame count (openable via cv2 head). Retried: on a busy node the probe
# occasionally prints nothing (transient FS / pixi hiccup) — seen as
# "bad frame count ''" on 5/32 chunks while the other 27 read the same file.
# An empty result is retried; a real 0 (unreadable video) still fails fast.
N=""
for attempt in 1 2 3 4; do
  N=$("$PIXI" run --manifest-path "$REPO/pyproject.toml" python -c \
      "import cv2,sys; print(int(cv2.VideoCapture(sys.argv[1]).get(cv2.CAP_PROP_FRAME_COUNT)))" \
      "$FIRSTCAM" | tail -1)
  [ -n "$N" ] && break
  echo "frame-count probe returned nothing (attempt $attempt) — retrying in $((attempt*15))s"
  sleep $((attempt*15))
done
[ "$N" -gt 0 ] 2>/dev/null || fail "bad frame count '$N'"

# Half-open [FS, FE) in source indices. `-–end-frame 0` in tailcycle means
# "to end", but we always compute an explicit FE.
W=$(( N / NUM_CHUNKS ))
FS=$(( (i-1) * W ))
if [ "$i" -eq "$NUM_CHUNKS" ]; then FE=$N; else FE=$(( i * W )); fi
# Mouse-presence gate: the driver writes ranges.txt ("FS FE" per line, one per
# chunk) when the session has a YOLO presence track, so only frames with the
# mouse in the arena are predicted. Line i overrides the equal split.
RANGES=$CF/predictions/$STAGED_ID/ranges.txt
if [ -f "$RANGES" ]; then
  read -r FS FE < <(sed -n "${i}p" "$RANGES")
  [ -n "$FE" ] || fail "no line $i in $RANGES"
  [ "$FE" -gt "$N" ] && FE=$N
fi

# Smoke mode: TAILCYCLE_SMOKE_END clamps chunk 1 to a tiny frame range so
# the driver can validate the ENTIRE real path (cam-regex, calibration,
# model load, detector, output) in ~5 min on ONE gpu before committing the
# full K-way array. Only chunk index 1 runs in smoke mode; the driver
# submits a single-element array for it.
# Skip prior successful chunks (fresh runs delete the .FAILED file but the
# session dir may still be here from a partial). The .range marker (written
# below only after infer exits 0) or the legit-empty marker is the completion
# signal -- NOT points3d.pq, which infer writes incrementally, so a crashed
# chunk leaves a partial one behind.
if [ -f "$OUT_DIR.range" ] || [ -f "$OUT_DIR.EMPTY" ]; then
  echo "chunk $i already produced -- skipping"
  exit 0
fi

run_infer() {  # $1=FS $2=FE $3=OUT_DIR ; returns infer's exit code
  rm -rf "$3"
  cd "$REPO"
  "$PIXI" run python scripts/infer.py \
    --run "$MODEL_RUN" \
    --videos "$REC" \
    --calibration "$CALIB" \
    --cam-regex 'Cam[0-9]+' \
    --group-id "000" \
    --dataset-name "$DATASET_NAME" \
    --anchor none --track --overlap 4 \
    --detector "$DETECTOR_RUN" \
    --det-score "$DET_SCORE" \
    --max-animals 1 \
    --trim-to-shortest \
    --start-frame "$1" --end-frame "$2" \
    --max-ram 24 \
    --units mm \
    --out "$3" \
    ${SCORER_RUN:+--scorer "$SCORER_RUN"} \
    $EXTRA_FLAGS
}

# Retrying wrapper. Startup races (30+ array elements hit pixi/NFS at once:
# "Failed to load lock file", "missing `version` field", stale handles) kill
# infer within seconds; they're transient, so a FAST failure (< 180 s) is
# retried up to 4 attempts with jitter + backoff so chunks stop colliding in
# lockstep. A failure after real work started (>= 180 s) is retried only once
# (it may be a mid-run stale handle, but a genuine error shouldn't burn ~35 min
# of GPU four times). 2026-09-29: blossom 09_17_14_17_45 chunk 13 lost both of
# its old single-retry attempts to "Failed to load lock file".
run_infer_retry() {  # same args as run_infer; returns 0 on success
  local attempt=1 slow_fails=0 t_start dt wait_s rc out
  out=$(mktemp)
  while :; do
    t_start=$(date +%s)
    run_infer "$@" 2>&1 | tee "$out"; rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] && { rm -f "$out"; return 0; }
    dt=$(( $(date +%s) - t_start ))
    if [ "$dt" -ge 180 ]; then
      slow_fails=$((slow_fails + 1))
      [ "$slow_fails" -ge 2 ] && { echo "infer failed after ${dt}s (twice past startup) -- giving up"; rm -f "$out"; return 1; }
    elif ! grep -qiE "lock file|stale file handle|Errno 116|missing .version. field|Input/output error|Resource temporarily unavailable" "$out"; then
      # Fast AND not a known race: a deterministic error (bad flag, missing
      # file, API change...). Retrying just burns GPU time (2026-10-06: a
      # missing --units was retried 4x over 7 min).
      echo "infer failed after ${dt}s with a non-transient error -- not retrying"
      rm -f "$out"; return 1
    fi
    [ "$attempt" -ge 4 ] && { echo "infer failed $attempt times -- giving up"; rm -f "$out"; return 1; }
    wait_s=$(( 20 + RANDOM % 70 + attempt * 30 ))
    echo "infer attempt $attempt failed after ${dt}s -- retrying in ${wait_s}s (transient NFS/pixi race guard)"
    sleep "$wait_s"
    attempt=$((attempt + 1))
  done
}

if [ -n "${TAILCYCLE_SMOKE_END:-}" ]; then
  # Smoke mode: validate the ENTIRE real path (cam-regex, calibration, model
  # load, detector, output) on a tiny frame range before the driver commits
  # the full K-way array. A single window can be a false failure — the mouse
  # may be off-camera at the start, and some recordings have LIGHTS-OFF
  # segments (measured: pitch-black middle in wasabi/2026_08_13_16_37_17) —
  # so try several windows across the recording before giving up.
  SMOKE_N=${TAILCYCLE_SMOKE_END}
  OUT_DIR=$CHUNKS/smoke
  if [ -f "$RANGES" ]; then        # centres of the middle / first / last range
    K=$(wc -l < "$RANGES")        # (range edges are padding, mouse not yet seen)
    STARTS=""
    for L in $(( (K + 1) / 2 )) 1 "$K"; do
      read -r RS RE < <(sed -n "${L}p" "$RANGES")
      C=$(( (RS + RE) / 2 - SMOKE_N / 2 )); [ "$C" -lt 0 ] && C=0
      STARTS="$STARTS $C"
    done
  else                              # N/2, N/4, 3N/4
    STARTS="$(( N * 2 / 4 )) $(( N / 4 )) $(( N * 3 / 4 ))"
  fi
  for FS in $STARTS; do
    FE=$(( FS + SMOKE_N ))
    [ "$FE" -gt "$N" ] && FE=$N
    echo "SMOKE MODE: frames [$FS, $FE)"
    run_infer_retry "$FS" "$FE" "$OUT_DIR" || fail "infer failed"
    if [ -f "$OUT_DIR/points3d.pq" ]; then
      echo "$FS $FE" > "$OUT_DIR.range"
      echo "CHUNK_DONE idx=smoke frames=$((FE-FS))"
      exit 0
    fi
    echo "smoke window [$FS, $FE) produced no detections (dark/empty?) -- trying next window"
  done
  fail "no detections in any smoke window (N/2, N/4, 3N/4) -- dark recording or detector failure"
fi

echo "WINDOW idx=$i/$NUM_CHUNKS host=$(hostname) N=$N start=$FS end=$FE"
t0=$(date +%s)
# Bounded retry of the transient startup races — see run_infer_retry.
run_infer_retry "$FS" "$FE" "$OUT_DIR" || fail "infer failed (after retries)"
t1=$(date +%s)

if [ ! -f "$OUT_DIR/points3d.pq" ]; then
  # infer exited 0 but produced no 3D points: every frame in this window had
  # zero detections (lights-off segment, empty arena). That is a valid EMPTY
  # result, not a failure — mark it so merge skips this chunk instead of
  # failing the whole session.
  echo "$FS $FE" > "$OUT_DIR.EMPTY"
  echo "CHUNK_EMPTY idx=$i frames=$((FE-FS)) (no detections -- dark/empty segment)"
  exit 0
fi
echo "$FS $FE" > "$OUT_DIR.range"
echo "CHUNK_DONE idx=$i frames=$((FE-FS)) seconds=$((t1-t0))"
