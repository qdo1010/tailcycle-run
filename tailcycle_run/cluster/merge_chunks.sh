#!/bin/bash
# Cluster-side merge: concat every chunk_NNN.npz produced by predict_chunk.sh
# into ONE data3D.csv covering the whole session. Writes DONE / FAILED so the
# driver's poll loop knows to pull.
#
# Args:
#   1  STAGED_ID   dir name (same as predict_chunk.sh's arg 1)
#   2  NUM_CHUNKS  expected count (fail early if a chunk went missing)
#   3  KP_SET      optional: johnson (default, 24 kp) | allen (47 kp, mega)
#
set -uo pipefail

STAGED_ID=$1; NUM_CHUNKS=$2; KP_SET=${3:-johnson}
CF=${TCR_WORK:-$HOME/mouse_cluster}
REPO=${TCR_REPO:-$HOME/tailcyclenet}
PIXI=${TCR_PIXI:-$HOME/.pixi/bin/pixi}
export PIXI_FROZEN=true   # never rewrite pixi.lock: concurrent array jobs corrupted it (2026-11-12)
CHUNKS=$CF/predictions/$STAGED_ID/chunks
OUT=$CF/predictions/$STAGED_ID
# merge_chunks.py ships beside this script. The driver deploys each harness
# version to its own content-hashed dir (<tools_dir>/tailcycle/v_<hash>)
# so a running job's scripts are never replaced; resolve our own dir, never
# a fixed shared path.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DONE=$OUT/DONE
FAILED=$OUT/FAILED

# Fail fast if any chunk didn't produce. NEW pipeline emits chunk_NNN/points3d.pq.
# A chunk_NNN.EMPTY marker means the chunk RAN cleanly but had zero detections
# (lights-off / empty-arena segment) — that's a valid empty result, not a
# missing chunk.
missing=()
for i in $(seq 1 "$NUM_CHUNKS"); do
  c=chunk_$(printf '%03d' "$i")
  # .range = infer exited 0; points3d.pq alone may be a crashed chunk's partial.
  [ -f "$CHUNKS/$c/points3d.pq" ] && [ -f "$CHUNKS/$c.range" ] && continue
  [ -f "$CHUNKS/$c.EMPTY" ] && { echo "note: $c is empty (no detections)"; continue; }
  missing+=("$c/points3d.pq")
done
if [ "${#missing[@]}" -gt 0 ]; then
  { echo "merge FAILED: missing chunks: ${missing[*]}"; \
    echo "check for chunk_*.FAILED files in $CHUNKS"; } > "$FAILED"
  ls "$CHUNKS"/chunk_*.FAILED 2>/dev/null | while read f; do
    echo "--- $f ---" >> "$FAILED"; head -20 "$f" >> "$FAILED"
  done
  exit 1
fi

"$PIXI" run --manifest-path "$REPO/pyproject.toml" python \
  "$HERE/merge_chunks.py" \
  --chunks-dir "$CHUNKS" \
  --num-chunks "$NUM_CHUNKS" \
  --out-csv    "$OUT/data3D.csv" \
  --kp-set     "$KP_SET" \
  || { echo "merge_chunks.py failed" > "$FAILED"; exit 1; }

touch "$DONE"
echo "MERGE_DONE $STAGED_ID -> $OUT/data3D.csv"
