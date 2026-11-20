# Faster tailcyclenet inference on many GPUs

`tailcyclenet infer` runs one recording on one GPU, start to finish. A long
multi-camera recording can take many GPU-hours, so on one GPU it takes that
many hours of wall clock. It has no built-in multi-GPU mode.

To use more GPUs, **split the recording into frame chunks and run one `infer`
per chunk, on as many GPUs as you have, then concatenate the outputs.** This
works because `infer` can predict any `[--start-frame, --end-frame)` range
straight from the raw videos, and its output `frame` column is always the source
frame index. The merge is then a plain concatenation.

```
recording (N frames, C cameras)
 ├── chunk 1  [0,    N/K)   ─► GPU a ─► chunk_001/points3d.pq ─┐
 ├── chunk 2  [N/K,  2N/K)  ─► GPU b ─► chunk_002/points3d.pq ─┼─► merge ─► points3d.pq (+ .csv)
 ├── ...                                                       │
 └── chunk K  [..., N)      ─► GPU z ─► chunk_K/points3d.pq ───┘
```

The wall clock drops roughly by the number of GPUs running at once. The total
GPU-hours rise slightly, because each chunk pays its own startup cost (see
[Choosing K](#choosing-the-number-of-chunks-k)).

(A "chunk" here is a frame range given to one `infer` process. It is not
tailcyclenet's own internal *window*, the short clip the model sees at a time,
which `--overlap` and `--prefetch-windows` refer to.)

This folder holds the whole recipe. It needs only a tailcyclenet checkout and
does not depend on anything else in this repo:

| file | what |
|---|---|
| `chunks.py` | `split`: writes `chunks.txt` (K chunks over the shortest camera, or one short smoke chunk). `merge`: checks every chunk finished and concatenates them |
| `infer_chunk.sh` | predicts one chunk (index from the argument, LSF or Slurm) |
| `run_local.sh` | runs all chunks on one machine, one worker per GPU |
| `example.env` | the settings both scripts read |

`infer_chunk.sh` calls `scripts/infer.py` with `--run`, `--detector`,
`--videos`, `--calibration`, `--cam-regex`, `--units`, `--max-animals`,
`--start-frame`, `--end-frame` and `--out`. Old and new tailcyclenet checkouts
accept all of these (`--run` is the older spelling of `--pose`). Check
`pixi run python scripts/infer.py --help` if yours differs.

## Before you start

1. **Install tailcyclenet once** (`pixi install` in the checkout).
2. **Check the containers.** Some rigs save MPEG-TS streams with a `.mp4`
   extension. Exact frame seeking does not work on TS, so a chunk would start at
   the wrong frame. `chunks.py split` refuses TS files. Rewrap them losslessly,
   without re-encoding:
   ```bash
   ffprobe -v error -show_entries format=format_name -of csv=p=0 Cam1.mp4   # "mpegts" = rewrap
   ffmpeg -i Cam1.mp4 -map 0 -c copy Cam1.rewrap.mp4                        # then check the frame count, and replace
   ```
3. **Fill in the settings:** `cp example.env my.env`, edit it, then `source my.env`.
   Anything else your model needs (e.g. `--dataset-name`, `--det-score`) goes in
   `TC_EXTRA_ARGS`.

## Run

```bash
source my.env

# 1. Smoke test: ONE short chunk from the middle of the recording, in its own
#    directory, on one GPU. A few hundred frames take minutes. Never merge it.
python chunks.py split --videos "$TC_VIDEOS" --smoke 300 --out "$TC_OUT.smoke"
TC_OUT="$TC_OUT.smoke" CUDA_VISIBLE_DEVICES=0 bash infer_chunk.sh 1
#    It should end with "exit 0" and $TC_OUT.smoke/chunk_001/points3d.pq.
#    Its run time gives you the rate for the cost estimate below.

# 2. Split for real. N = the shortest camera; K chunks of about N/K frames.
python chunks.py split --videos "$TC_VIDEOS" --chunks 16 --out "$TC_OUT"

# 3a. One machine with several GPUs:
bash run_local.sh 0,1,2,3

# 3b. Or a cluster: one array element per chunk (see below).

# 4. Merge. Lists the chunks to re-run if any did not finish.
python chunks.py merge "$TC_OUT" --csv
```

Run `chunks.py` with any Python >= 3.11 that has pyarrow. The tailcyclenet
environment works: `pixi run --manifest-path "$TC_REPO/pyproject.toml" python chunks.py ...`.

**Re-running is safe.** A chunk whose `session.toml` says `complete = true` is
skipped. Re-run `run_local.sh`, or the array, to redo only the failed ones.

## On a cluster

Each array element runs `infer_chunk.sh`, which reads its chunk from the
scheduler's index. Both LSF and Slurm pass the submitting shell's environment by
default, so `source my.env` first. K is the number of lines in `chunks.txt`,
and `%N` caps how many run at once.

**LSF** (the lab's own driver uses this array pattern in production; this exact
command has not been run):
```bash
K=$(grep -c . "$TC_OUT/chunks.txt"); mkdir -p "$TC_OUT/logs"
bsub -J "tc[1-$K]%8" -q <gpu-queue> -gpu "num=1" -n 8 -W 8:00 \
     -o "$TC_OUT/logs/%J_%I.out" bash "$PWD/infer_chunk.sh"
```

**Slurm** (not tested):
```bash
K=$(grep -c . "$TC_OUT/chunks.txt"); mkdir -p "$TC_OUT/logs"
sbatch --array=1-$K%8 --gres=gpu:1 --cpus-per-task=8 --time=08:00:00 \
       --output="$TC_OUT/logs/%A_%a.out" --wrap "bash $PWD/infer_chunk.sh"
```

Then run `chunks.py merge` once the array has ended. You can submit it as a
dependent job: LSF `-w "ended(tc)"`, Slurm `--dependency=afterany:<jobid>`.

## Choosing the number of chunks K

- **Each chunk pays a fixed startup cost** (environment, model and detector
  load, opening every camera), whatever its length. More chunks give shorter wall
  clock and smaller, cheaper-to-retry pieces, but more total GPU-hours.
- One measurement as a guide (an L4 GPU, one 47-keypoint model, the lab's
  multi-camera rig). Each element ran at 1.4–1.7 frames/s, and startup cost about
  2.7 min per chunk. That gives GPU-h ≈ frames / 5000 + 0.045 × K. Your model, GPU
  and camera count will differ, so time your smoke test and refit.
- A reasonable start: K = the number of GPUs you can get at once (a small
  multiple of it on a cluster, for retries and stragglers), with chunks of at
  least several thousand frames.
- **Memory.** Host RAM is sized from the job's limit (cgroup or LSF) by default.
  `--max-ram GB` caps it and changes speed only, never the predictions.

## What to know about the result

- **Identity resets at every chunk.** A ranged run is not a slice of the
  whole-clip run: tracking starts fresh at each `--start-frame`. Animal ids
  (`det00`, ...) are not comparable across chunks; the merge adds a `chunk`
  column so you can see where. With one animal this does not matter. With
  several, re-link identities across the seams, or use fewer, longer chunks.
- **Seams.** The first frames of each chunk have no history from the previous
  chunk. If a seam matters, check the frames on either side of it.
- **Cameras of different lengths.** Chunks cover the shortest camera. If
  infer refuses because the cameras differ in length, add `--trim-to-shortest`
  to `TC_EXTRA_ARGS`, but first check that the difference is a few frames at
  the end and not a desync.
- **Empty chunks are valid.** A chunk where nothing was detected (lights off,
  empty arena) finishes with no rows. The merge reports it and carries on.
- **Already on by default**, so no flag is needed: `--precision bf16`,
  `--camera-batch` (all cameras encoded in one call) and `--prefetch-windows 1`
  (decode ahead while the GPU works).

## Pitfalls seen in practice

- **Many jobs starting at once** in one pixi environment on a shared filesystem
  corrupted `pixi.lock`. `infer_chunk.sh` sets `PIXI_FROZEN=true`; run
  `pixi install` once before submitting.
- The same burst can cause short-lived startup errors (`Failed to load lock
  file`, stale file handle). Re-running the failed chunks fixes them; the
  `complete = true` check makes that cheap.
- A chunk that crashed partway leaves a partial `points3d.pq` (it is written as
  inference runs). Trust `session.toml: complete = true`, not the file's
  existence. The merge and the skip check both do. (Older tailcyclenet checkouts
  do not write `complete`: there the skip never fires, so every chunk re-runs,
  and the merge needs `--no-complete-check`.)
- **A GPU budget:** project the cost (formula above) before submitting a large
  array. Keep `%N` small enough to stop the array in time if the first elements
  run slower than planned.
