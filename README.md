# tailcycle-run

Run tailcyclenet multi-camera 3D mouse pose inference on an
LSF GPU cluster. The package stages a session's videos and calibration, submits
a chunked LSF array plus a merge job, polls, pulls `data3D.csv` and
post-processes it. It was extracted from a lab dashboard. The code
behaves as it did there, except that every site value now comes from a config
file.

**Just want tailcyclenet faster on several GPUs, without this lab setup?** See
[docs/parallel-gpu-inference](docs/parallel-gpu-inference/README.md): a
self-contained recipe (split into frame windows, one `infer` per GPU, merge)
that needs only a tailcyclenet checkout.

tailcyclenet itself (the model code) is **not** in this repo. This repo is the
driver around it: it moves data, builds the calibration, talks to LSF, guards
the GPU budget and cleans the output.

## How it works

```
 local workstation                              LSF cluster
 ─────────────────                              ───────────
 <video_root>/<animal>/<session>/Cam*.mp4  ──►  <work_dir>/tailcycle_videos/<animal>__<session>/
 <opencv_calib_root>/<date>/calibration/   ──►  <work_dir>/tailcycle_calibs/<date>/calibration.toml
         (both copied over the SMB mount of the cluster home; ssh is for bsub/bjobs/df only)

 tailcycle-run submit ── bsub ──►  smoke job (24 frames)
                                   array [1-K]: one element = one frame window
                                       predict_chunk.sh -> tailcyclenet infer.py -> chunk_NNN.npz
                                   merge job (-w ended(array)): merge_chunks.sh -> predictions/<sid>/data3D.csv + DONE

 <pred_root>/<animal>/<session>/<out_name>/data3D.csv  ◄── pulled over the mount, plus info.yaml
        │
        └── post: yolo-clean, body-span, presence, trials, hurdle-windows, align
```

Status is read from marker files (`DONE`, `FAILED`, `smoke.FAILED`, `REMUX_OK`)
on the SMB mount. The driver never polls the login node.

## Prerequisites

You need all of these before the first run:

| need | why |
|---|---|
| Python >= 3.10 | the driver and post-processing |
| `ffprobe` on the PATH | container check (MPEG-TS vs MP4), frame counts |
| `ssh` to the LSF login node, key-based | `bsub`, `bjobs`, `bkill`, `df` only |
| the cluster home mounted locally over SMB/CIFS | all bulk data and status markers move here; submit stops when it is down |
| a `tailcyclenet` checkout with `pixi` on the cluster | the model code the array jobs run (`cluster.tailcycle_repo`, `cluster.pixi`) |
| model checkpoints on the cluster filesystem | one set per variant you use (`[models]`) |
| per-date OpenCV calibration yamls locally | `<opencv_calib_root>/<date>/calibration/Cam*.yaml` |
| a GPU queue and an LSF project you may use | `cluster.queue`, `cluster.project` |
| (post-processing only) a YOLO weights file and the lab's `analyzeMiceTrajectory` checkout | see [Known couplings](#known-couplings--todo) |

`config.example.toml` is the full list of site values, with comments.

## Install

```bash
pip install -e .            # numpy, pandas, opencv-python, pyyaml (+ tomli on py3.10)
pip install -e '.[yolo]'    # + ultralytics, for the YOLO presence track
cp config.example.toml ~/.config/tailcycle-run/config.toml   # then fill in <placeholders>
```

The config path can also be given with `TAILCYCLE_RUN_CONFIG=/path/config.toml`.
**Keep the filled config out of git.** It names hosts, users and paths.
Nothing site-specific is hard-coded in the package; `tests/test_scrub.py`
enforces that. `.gitignore` already excludes `config.toml`.

## Quick start: one session

Expected inputs:
- videos at `<video_root>/<animal>/<session>/Cam<serial>.mp4`, one file per camera
  (session names look like `2026_11_17_14_30_05`);
- a calibration for the session's date, or an adjacent one (the nearest date is
  used when the exact date is missing; force one with `--calib-date YYYY_MM_DD`).

```bash
# 1. Project the cost. Nothing is staged. Get the number approved first.
tailcycle-run plan --ranges windows.txt --video-dir /path/to/<animal>/<session>
#    windows.txt: one "FS FE" per line, half-open source frames, '#' comments ok
#    or, without a file:  tailcycle-run plan --frames 119618 --windows 263

# 2. Run it: stage -> smoke -> array -> merge -> poll -> pull.
tailcycle-run submit --animal A --session S --variant univ2_47 --ranges-file windows.txt

# 3. Post-process the result.
tailcycle-run post yolo-clean --help
tailcycle-run post body-span --help
```

A `submit` for a session that already has `DONE` on the cluster skips straight
to the pull, and the staging directory name is deterministic
(`<animal>__<session>`), so re-running after an interruption is safe.

Many sessions: `batch` (one at a time, from `[sources.<tab>]`) or `run` (an
unattended plan file with a total GPU-h cap). See [Plan batches](#plan-batches-run).

## Where things land

| what | where |
|---|---|
| staged videos | `<work_dir>/tailcycle_videos/<animal>__<session>/` (cluster; reclaimed after DONE) |
| calibration | `<work_dir>/tailcycle_calibs/<date>/calibration.toml` (built from the OpenCV yamls) |
| harness scripts | `<tools_dir>/...`, a content-hashed directory per version, so running jobs never see a change |
| per-chunk output and markers | `<work_dir>/predictions/<sid>/` (cluster) |
| **result** | `<pred_root>/<animal>/<session>/<out_name>/data3D.csv` (local) |
| run record | `<out_name>/info.yaml` next to it: calibration, model, array id, world frame |
| scorer quality | `quality.csv`, only for variants with a scorer (`univ24`, `univ47`) |
| gated pose | `yolo_clean*/data3D.csv` and `masks.npz`, from `post yolo-clean` |
| driver logs | `<local.log_dir>/tailcycle_<animal>_<session>[_<variant>].log` |

`data3D.csv` has one row per video frame. Frames outside the predicted windows
stay NaN.

## Stages

| stage | command | what |
|---|---|---|
| plan | `tailcycle-run plan --ranges R [--video-dir D]` | GPU-h projection and staged TB, with **no staging** |
| submit | `tailcycle-run submit --animal A --session S [--variant univ2_47] [--ranges-file R]` | stage → smoke test (24 frames) → array → merge → poll → pull |
| batch | `tailcycle-run batch --tab T --animal A --sessions S1 S2 ...` | sessions one at a time, with a consecutive-fail stop |
| run | `tailcycle-run run --plan P.tsv --total-cap GPU_H [--bwlimit KB/s] [--max-concurrent N]` | unattended `univ2_47` batch over a plan: per session submit + guard + post, with a total GPU-h cap (alias `batch-plan`) |
| guard | `tailcycle-run guard --array ID [--merge ID] --cap GPU_H` | GPU-h cap, kills the array when it would be exceeded |
| post | `tailcycle-run post presence\|trials\|hurdle-windows\|yolo-clean\|body-span\|align ...` | YOLO track, trial windows, hurdle windows, gated pose, body span, world frame |

`submit` runs `tailcycle_run/cluster/cluster_run.py`. Its flow:
1. Stage `Cam*.mp4` to `<work_dir>/tailcycle_videos/<animal>__<session>/` over
   the **SMB mount of the cluster home**. The login node is never a data
   path; when the mount is down, submit stops. A `.staging` lock is touched
   first (and `predictions/<sid>/` created), so the reclaim sweep leaves the
   dir alone. `TAILCYCLE_SKIP_VIDEO_STAGE=1` reuses copies already staged.
2. **Rewrap (v2 variants only).** ffprobe checks each staged container. The
   v2 reader seeks exact frame indices, which MPEG-TS (saved as `.mp4` by some
   rigs) cannot give. TS cameras are rewrapped to MP4 on the cluster by
   `remux_ts.sh` (a CPU job on `cluster.cpu_queue`, `ffmpeg -c copy`; a file
   is replaced only when its TS packet count equals the MP4 frame count). The
   driver waits on the `REMUX_OK` marker and the job log (in
   `cluster.log_dir`) over the mount, one check every 3 min, for at most 7 h.
   It then checks again and refuses to submit if any camera is still TS.
3. Build `calibration.toml` (aniposelib) from the date's OpenCV yamls and
   upload it.
4. Deploy the harness (`predict_chunk.sh`, `merge_chunks.*`, converters) into
   a content-hashed dir, so running jobs never see a script change.
5. Run a 24-frame smoke job, then the K-way array (one element per
   `--ranges-file` window), then a merge job (`-w ended(array)`). The index
   list is compressed into ranges (`[1-263]`, `[1-5,9,12-20]`); LSF rejects
   a job name with hundreds of explicit indices. `TAILCYCLE_MAX_CONCURRENT=N`
   adds an LSF `%N` slot limit, read from the env or, for drivers already
   running, from `<local.log_dir>/tailcycle_max_concurrent.<animal>__<session>`.
6. Poll the DONE / FAILED markers through the SMB mount with backoff
   (1 → 10 min). It does not poll the login node.
7. Pull `data3D.csv` into `<pred_root>/<animal>/<session>/<out_name>/` and
   write `info.yaml` (calibration, model, array id, world frame).

Camera gates run before staging:
- blurry cameras (variance of the Laplacian, vs the session median);
- calibration reprojection error;
- frame-count desync (more than 2 frames from the median);
- cameras that have no video.

The submit refuses to run with fewer than 8 usable cameras.

Variants (`--variant`): `univ2_47` (47-kp allen set, `tailcycle47_v2/`),
`univ2_24`, `univ24` (with scorer → `quality.csv`), `univ47`, `mega24`,
`mega47`, `johnson24`. Checkpoint paths are in `[models]`.

## Spend and stop rules

- **Plan first.** GPU-h ≈ `frames/5000 + 0.045 × windows`. This was refit
  on gpu_l4 with `univ2_47` (2026-11-17/18): 1.4–1.7 fps per element plus
  about 0.045 GPU-h (~2.7 min) of startup per window. The earlier
  `windows/60` under-counted startup: a 263-window, 119,618-frame smoke
  projects 35.8 GPU-h and measured 35.7. Get the projection approved before
  staging.
- **Ranges trim the cost.** `--ranges-file` predicts only the given
  `FS FE` windows (half-open source frames), e.g. trial windows or object
  crossings.
- **Disk floor.** Submit aborts below 500 GB free on `cluster.df_path`
  (50 GB with `TAILCYCLE_SKIP_VIDEO_STAGE`, which only writes predictions).
- **Staged videos are reclaimed** once every predictions run of a session
  has DONE. A fresh `.staging` lock protects a session while it is still
  being staged.
- **Pacing.** `TAILCYCLE_BWLIMIT=<KB/s>` caps the upload, for sources whose
  owner limits the read rate.
- **burn_guard** does one batched `bjobs` call every 15 min. It kills the
  array and its merge job when measured GPU-h reaches 90 % of the cap, or
  when (after ≥ 20 % of elements are done) the per-element mean projects
  past the cap. The guard can only stop elements that have not started yet,
  so **keep K concurrent × element time well inside the guard interval**:
  without a `%N` limit, 255 of 263 elements once finished inside one 15-min
  interval. `run` starts the guard every 5 min with
  `TAILCYCLE_MAX_CONCURRENT` set.
- **Budget override.** `GATE_OVERRIDE_BUDGET=<reason>` (env, or the reason in
  `<local.log_dir>/gate_override_budget` for drivers already running) is
  passed to the spend gate. Set it only on the budget owner's word, and
  remove the file when the batch is done.
- **Spend gate.** Every bsub gets the `[gate]` env (session, purpose,
  optional sandbox waiver), for a site bsub wrapper.
- **No polling loops on the login node.** Cluster-side status is read from
  the SMB mount.

## Plan batches (`run`)

`run` reads a TSV with a header row and the columns
`tab animal session ranges proj_gpuh`. `tab` names a `[sources.<tab>]`
config table, `ranges` is a `--ranges-file`, and `proj_gpuh` comes from
`plan`. For each row it:
1. skips the row if `<pred_root>/<animal>/<session>/tailcycle47_v2/data3D.csv`
   exists (it also skips sealed animals unless `--allow-sealed` is given);
2. stops the batch if GPU-h spent so far plus this projection exceeds
   `--total-cap`;
3. waits, rechecking every 30 min, until the cluster has room for the
   session's videos plus 500 GB;
4. runs `submit --variant univ2_47` with `TAILCYCLE_BWLIMIT` and
   `TAILCYCLE_MAX_CONCURRENT`. Once the array and merge ids appear in the
   driver log, it starts `guard` on them with a per-session cap of
   `max(1.5 × proj, proj + 2)` GPU-h, checking every 5 min;
5. post-processes: `yolo-clean --source tailcycle47_v2 --no-trial-gate`, then
   `body-span` on `tailcycle47_v2` and `yolo_clean47_v2`.

The batch stops after `--max-consecutive-fail` failures in a row (default 2),
or when a guard kills a session's jobs. Logs go to
`<local.log_dir>/tailcycle_<animal>_<session>_univ2_47.log`.

## Sealed animals

Animals held out as a final test set go in `sealed_animals` (config).
`submit` and `batch` refuse them unless `--allow-sealed` is given. In the
lab's use, one animal is sealed this way. Keep its frames out of every
training, labelling and tuning set, and never tune filters or thresholds on
it.

## Post-processing

- `post presence`: top-camera YOLO track (`yolo_track.csv`) and the
  mouse-present intervals (`presence.json`). Frames are decoded sequentially,
  never by seeking (MPEG-TS seeks land on the wrong keyframe).
- `post trials`: door-open → door-close trial windows from behaviour logs
  (`trials.json`).
- `post yolo-clean`: the YOLO gate, a ±1 s trial gate, a 3D velocity gate
  and continuity fill, written to `yolo_clean*/data3D.csv` plus `masks.npz`.
  **External dependency:** `filter_jarvis_with_yolo.py` from the lab's
  `analyzeMiceTrajectory` repo (`local.analyze_mice_trajectory`).
- `post hurdle-windows`: a `--ranges-file` of the frames where the mouse is
  near (`--near-mm`), or crossing (`--crossings`), a mapped hurdle sheet. It
  reads `yolo_track.csv` and `objects/hurdle_map.json`; the tool that writes
  the map is not in this repo. A session with an empty map gives no windows.
  It prints the GPU-h projection.
- `post body-span`: per-frame max pairwise keypoint distance, with and
  without the tail. This is the exploded-skeleton filter input.
- `post align` / `align-fit`: fit and apply a per-date rigid transform into
  one canonical world frame (calibrations and stored predictions).

## Tests

```bash
pip install -e '.[test]' && pytest
```

The tests need no cluster:
- config loading;
- the ranges parser;
- the plan formula (the 263-window, 119,618-frame smoke gives 35.8 GPU-h;
  measured 35.7);
- the LSF index ranges and the `%N` limit, and the budget-override env and
  file;
- the `run` skip and total-cap stop, which happen before any cluster call;
- `hurdle-windows` window merging and splitting, and the empty-map guard;
- `body_span.max_pairwise` on a hand-made array;
- the sealed-animal refusal;
- the scrub test.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `no config at ...` | copy `config.example.toml` to `~/.config/tailcycle-run/config.toml` or set `TAILCYCLE_RUN_CONFIG` |
| `config key '...' missing` | fill that key; the message names it |
| submit stops at once | the SMB mount of the cluster home is down. Remount it; the login node is not used as a data path |
| `only N usable cameras` | fewer than 8 cameras passed the gates (blur, reprojection error, frame-count desync, no video). Fix the data; do not lower the floor |
| refuses to submit, camera still MPEG-TS | the rewrap job failed or timed out (7 h). Read its log in `cluster.log_dir`, fix, re-run |
| smoke test FAILED | the driver prints the real infer error from the smoke log. Usually a cluster checkout, model path or flag mismatch (preflight checks the `infer.py` flags) |
| `<animal> is a SEALED animal` | by design. `--allow-sealed` only for the final held-out test |
| aborted: below 500 GB free | the disk floor on `cluster.df_path`. Wait for reclaim, or free space |
| bsub rejects the job name | too many explicit array indices; the driver compresses them to ranges, so this means a custom change |
| guard killed the array | measured or projected GPU-h passed the cap. Check the projection, then re-run with a new cap on the budget owner's word |
| stale `data3D.csv`, run not finished | completion is keyed on `DONE`, not on the CSV, which persists across runs |

## Collaborating

- The GitHub repo is **private**. A collaborator needs to be added on GitHub
  (Settings -> Collaborators) before they can clone it.
- Each person keeps their own filled `config.toml`. Never commit it.
- Run `plan` and have the GPU-h approved before any `submit` or `run`.
- Run `pytest` before pushing. `test_scrub.py` fails if a host, user, path or
  similar site string reaches a tracked file.

## Known couplings / TODO

- `yolo_clean` imports `filter_jarvis_with_yolo` from `analyzeMiceTrajectory`.
  Vendor it, or package it, to drop the path.
- The YOLO weights (2-class Mouse/SideCyl) are not distributed. Their path is
  `local.yolo_weights`.
- The cluster side expects a `tailcyclenet` checkout with pixi at
  `cluster.tailcycle_repo` / `cluster.pixi`. The univ2 variants use their own
  checkout (`models.univ2_repo`). Preflight warns when the local and cluster
  checkouts differ or lag `origin/main`.
- The reference camera serial (`Cam2005325`, top-down) is fixed in
  `yolo_presence`, `yolo_clean` and `trial_windows`.
- `run_sequential` tabs come from `[sources.<tab>]`. The dashboard this came
  from also tracks per-tab rules (min cameras, animal discovery) that are not
  ported.
- The dashboard's realigned-video tool (`fix_desync_video.py`) is not
  included. `cluster_run` still honours `<pred>/<session>/aligned/Cam*.mp4`
  overrides when they exist.
- `cluster.work_dir` and `cluster.log_dir` must be home-relative (`~/...`).
