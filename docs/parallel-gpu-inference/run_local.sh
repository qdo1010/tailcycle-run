#!/usr/bin/env bash
# Run every chunk in $TC_OUT/chunks.txt on THIS machine, one worker per GPU.
#
#   run_local.sh 0,1,2,3        # GPU ids to use
#
# Worker g runs chunks g+1, g+1+G, g+1+2G, ... one after another on its GPU.
# Logs: $TC_OUT/logs/chunk_NNN.log. Re-run it to redo only unfinished chunks.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${TC_OUT:?set TC_OUT}"
[ $# -eq 1 ] || { echo "usage: $0 GPU_IDS (e.g. 0,1,2,3)"; exit 2; }
IFS=, read -ra GPUS <<< "$1"
G=${#GPUS[@]}
K=$(grep -c . "$TC_OUT/chunks.txt") || { echo "no $TC_OUT/chunks.txt (run chunks.py split)"; exit 2; }
mkdir -p "$TC_OUT/logs"
echo "$K chunks on $G GPUs (${GPUS[*]})"

for g in "${!GPUS[@]}"; do
  (
    for ((i = g + 1; i <= K; i += G)); do
      log=$TC_OUT/logs/$(printf 'chunk_%03d' "$i").log
      if CUDA_VISIBLE_DEVICES=${GPUS[$g]} bash "$HERE/infer_chunk.sh" "$i" > "$log" 2>&1; then
        echo "chunk $i done (GPU ${GPUS[$g]})"
      else
        echo "chunk $i FAILED (GPU ${GPUS[$g]}), see $log"
      fi
    done
  ) &
done
wait
echo "all workers finished; next: python $HERE/chunks.py merge $TC_OUT"
