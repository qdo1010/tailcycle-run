#!/usr/bin/env bash
# Predict ONE frame chunk of a recording with tailcyclenet, on one GPU.
#
#   infer_chunk.sh [INDEX]
#
# INDEX (1-based line of $TC_OUT/chunks.txt) is the argument, else the
# scheduler's array index: LSB_JOBINDEX (LSF) or SLURM_ARRAY_TASK_ID (Slurm).
# Settings come from the environment; see example.env.
# Output: $TC_OUT/chunk_NNN/ (a tailcyclenet prediction session).
# A chunk whose session.toml already says `complete = true` is skipped, so
# re-running the whole set only redoes the chunks that did not finish.
set -uo pipefail

: "${TC_REPO:?set TC_REPO (tailcyclenet checkout)}"
: "${TC_POSE:?set TC_POSE (pose run folder or .pth)}"
: "${TC_DETECTOR:?set TC_DETECTOR (detector run folder or .pth)}"
: "${TC_VIDEOS:?set TC_VIDEOS (directory of one recording, one video per camera)}"
: "${TC_CALIB:?set TC_CALIB (aniposelib calibration.toml)}"
: "${TC_CAM_REGEX:?set TC_CAM_REGEX (capture group = camera name)}"
: "${TC_UNITS:?set TC_UNITS (units of the calibration, e.g. mm)}"
: "${TC_MAX_ANIMALS:?set TC_MAX_ANIMALS}"
: "${TC_OUT:?set TC_OUT (output directory, holds chunks.txt)}"
PIXI=${TC_PIXI:-pixi}
# Many jobs starting at once must not each rewrite pixi.lock (it corrupts it).
export PIXI_FROZEN=true

i=${1:-${LSB_JOBINDEX:-${SLURM_ARRAY_TASK_ID:-}}}
[[ "$i" =~ ^[0-9]+$ ]] && [ "$i" -ge 1 ] || { echo "no chunk index (arg, LSB_JOBINDEX or SLURM_ARRAY_TASK_ID)"; exit 2; }

read -r FS FE < <(sed -n "${i}p" "$TC_OUT/chunks.txt")
[ -n "${FE:-}" ] || { echo "no line $i in $TC_OUT/chunks.txt"; exit 2; }

OUT=$TC_OUT/$(printf 'chunk_%03d' "$i")
if grep -qE '^complete *= *true' "$OUT/session.toml" 2>/dev/null; then
  echo "chunk $i already complete, skipping"; exit 0
fi
rm -rf "$OUT"     # a partial from a crashed run; infer is not a resume

echo "chunk $i frames [$FS, $FE) on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-default}"
t0=$(date +%s)
cd "$TC_REPO" || exit 2
# TC_EXTRA_ARGS is split on spaces on purpose (extra infer flags).
# shellcheck disable=SC2086
"$PIXI" run python scripts/infer.py \
  --run "$TC_POSE" \
  --detector "$TC_DETECTOR" \
  --videos "$TC_VIDEOS" \
  --calibration "$TC_CALIB" \
  --cam-regex "$TC_CAM_REGEX" \
  --units "$TC_UNITS" \
  --max-animals "$TC_MAX_ANIMALS" \
  --start-frame "$FS" --end-frame "$FE" \
  --out "$OUT" \
  ${TC_EXTRA_ARGS:-}
rc=$?
echo "chunk $i exit $rc after $(( $(date +%s) - t0 )) s"
exit $rc
