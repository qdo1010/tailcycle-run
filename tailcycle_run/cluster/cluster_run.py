#!/usr/bin/env python3
"""Drive a cluster tailcyclenet inference on one (animal, session).

Flow:
  1. Stage full-resolution videos    <video_root>/<animal>/<session>/Cam*.mp4
                                  →  <cluster>:<work>/tailcycle_videos/<sid>/
  2. Stage OpenCV Cam*.yaml calib    <opencv_calib_root>/<date>/calibration/
                                  →  <cluster>:<work>/tailcycle_calibs/<date>/
  3. Deploy harness scripts          tailcycle_run/cluster/{predict_chunk,merge_chunks}.{sh,py}
                                  →  <cluster>:<tools_remote>/tailcycle/
  4. Submit LSF array + merge dep    K-way; each element ffmpeg-trims its own
                                     N/K-frame chunk, converts to tailcycle format,
                                     runs pixi + scripts/infer.py, drops chunk_iii.npz.
                                     Merge job concats -> data3D.csv on cluster.
  5. Poll DONE / FAILED             via the CIFS mount (ssh only as fallback), backoff 1->10 min
  6. Pull                            <cluster>:<work>/predictions/<sid>/data3D.csv
                                  →  <PRED_ROOT>/<animal>/<session>/tailcycle/data3D.csv

STAGED_ID is deterministic: "<animal>__<session>" so re-invocations idempotently
reuse the staging dir. If DONE already exists on cluster we skip straight to pull.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg, path as _cfgp  # noqa: E402

LOGIN            = _cfg("cluster.login")                 # user@login-host
CF_REMOTE        = _cfg("cluster.work_dir", "~/mouse_cluster")      # must start with "~/"
WORK_REL         = CF_REMOTE[2:]                          # the same, relative to the cluster home
TOOLS_REMOTE     = _cfg("cluster.tools_dir", "~/tailcycle_run_tools")
LOCAL_CIFS       = _cfgp("cluster.smb_mount")            # local mount of the cluster home
CLUSTER_HOME     = _cfg("cluster.home")                  # the same home, as compute nodes see it
CLUSTER_DF_PATH  = _cfg("cluster.df_path", CLUSTER_HOME) # filesystem whose free space gates submits

# JARVIS-format calib isn't used by tailcycle (aniposelib format is built on the
# cluster from the OpenCV yamls we upload), but we still key by date.
LOCAL_OPENCV_CALIB_ROOT = _cfgp("local.opencv_calib_root")

# Defaults when --video-root / --pred-root are not given.
DEFAULT_VIDEO_ROOT = _cfgp("local.default_video_root")
DEFAULT_PRED_ROOT  = _cfgp("local.default_pred_root")
# pred root used to find realigned overrides when the video root is not the default one
FALLBACK_PRED_ROOT = _cfgp("local.fallback_pred_root", str(DEFAULT_PRED_ROOT))
LOCAL_PYTHON       = _cfg("local.python", sys.executable)
LOCAL_TAILCYCLENET = _cfgp("local.tailcyclenet_repo")
JARVIS_CALIB_ROOT  = _cfgp("local.jarvis_calib_root")

DEFAULT_QUEUE     = _cfg("cluster.queue", "gpu_l4")
DEFAULT_PROJECT   = _cfg("cluster.project")
DEFAULT_WALL      = _cfg("cluster.wall", "8:00")
DEFAULT_SLOTS     = int(_cfg("cluster.slots", 8))     # gpu_l4 is 8 slots per node -- larger allocs
                            # (tried -n 16) hang bsub for >5 min without
                            # scheduling. Stay single-node.
DEFAULT_CHUNKS    = 32     # K in the LSF array. K=32 keeps each chunk under
                            # the 120 GB gpu_l4 slot cap: 69k frames / 32
                            # ≈ 2.2k per chunk halves peak memory vs K=16's
                            # 4.3k. Trade: +30-45 s fixed overhead per chunk
                            # (model load, ffmpeg), so total wall-clock is
                            # comparable but reliability is much higher.
DATASET_NAME      = "mouse-annotated-aug"   # 500k-repaired model's registry name

# Model variants. "mega24" is the production default (24-kp output into
# tailcycle/ — IK, bad frames, overlay skeleton, RED all assume 24 kp),
# produced by the mega-mouse model; "johnson24" is the previous 24-kp model
# (same output dir), kept as a fallback. "mega47" is the
# multi-dataset mega-mouse model run with the allen 47-kp output set —
# same videos/calib, separate predictions identity (<sid>__47) and a
# separate local output dir (tailcycle47/) so the two never collide.
# Multi-dataset mega-mouse model. Its keypoint registry holds BOTH schemas:
# allen-mouse-combined (47 kp, idx 0-46) and johnson-mouse-combined-aug
# (24 kp, idx 47-70 — same names/order as our KEYPOINTS). infer.py picks the
# block via --dataset-name (tailcyclenet/infer/driver.py -> adopt.dataset_name),
# so the same checkpoint serves the 24-kp production output natively.
MEGA_MODEL_RUN = _cfg("models.mega_run", "")

# Universal mouse models (Oct 2026). Pose registry holds
# allen-mouse-combined(47), calms21(7), facemap(25), johnson-mouse-combined-aug
# (24, same names/order as our KEYPOINTS) and muss-17m(22). The detector is a
# multi-root detector trained on johnson-mouse-combined-aug (+ muss-17m,
# facemap), so a johnson session is IN its domain: no --allow-detector-transfer.
# The scorer was trained on top of this pose run's checkpoint with the video
# encoder frozen, so pose+scorer share one encoder pass. All three need
# tailcyclenet >= fc03b31 (2026-10-05) on the cluster.
UNIV_POSE_RUN = _cfg("models.univ_pose_run", "")
UNIV_DETECTOR_RUN = _cfg("models.univ_detector_run", "")
UNIV_SCORER_RUN = _cfg("models.univ_scorer_run", "")

# Universal mouse v2 (2026-10-29/30), packaged self-contained
# checkpoints (config + keypoint registry inside the .pth; registry includes
# johnson-mouse-combined-aug, 24 kp, same names/order). Trained on tailcyclenet
# e6d72bd, ~240 commits past the v1 checkout's fc03b31, so it runs from its own
# checkout (models.univ2_repo, at e6d72bd); the v1 checkout is untouched. No v2
# scorer exists (the v1 scorer shares v1's encoder), so no quality.csv.
# Detector objectness quantiles on GT boxes: q01 0.11, q10 0.58, so the v1
# floor of 0.5 would drop ~10% of true boxes; use 0.1.
UNIV2_POSE = _cfg("models.univ2_pose", "")
UNIV2_DETECTOR = _cfg("models.univ2_detector", "")
UNIV2_REPO = _cfg("models.univ2_repo", "")

VARIANTS = {
    # v2 universal mouse model, 24-kp johnson output into tailcycle_v2/ next to
    # the v1 tailcycle/ (predictions identity <sid>__v2), so both coexist.
    "univ2_24": {
        "dataset":  "johnson-mouse-combined-aug",
        "model":    UNIV2_POSE,
        "detector": UNIV2_DETECTOR,
        "repo":     UNIV2_REPO,
        "det_score": "0.1",
        "kp_set":   "johnson",
        "out_name": "tailcycle_v2",
        "sid_suffix": "__v2",
    },
    # v2 universal mouse model, 47-kp allen output (its registry holds
    # allen-mouse-combined too) into tailcycle47_v2/, next to v1's tailcycle47/.
    "univ2_47": {
        "dataset":  "allen-mouse-combined",
        "model":    UNIV2_POSE,
        "detector": UNIV2_DETECTOR,
        "repo":     UNIV2_REPO,
        "det_score": "0.1",
        "kp_set":   "allen",
        "out_name": "tailcycle47_v2",
        "sid_suffix": "__47v2",
    },
    # Production 24-kp output (tailcycle/) from the universal mouse pose model
    # + universal detector, with the framewise scorer's quality per frame
    # (-> quality.csv). Default for the "▶ tailcycle (cluster)" button.
    "univ24": {
        "dataset":  "johnson-mouse-combined-aug",
        "model":    UNIV_POSE_RUN,
        "detector": UNIV_DETECTOR_RUN,
        "scorer":   UNIV_SCORER_RUN,
        "kp_set":   "johnson",
        "out_name": "tailcycle",
        "sid_suffix": "",
    },
    # 24-kp output (tailcycle/) from the MEGA model, queried with
    # the Johnson keypoint set. Default for the "▶ tailcycle (cluster)" button.
    # johnson-mouse-combined-aug is also the detector's own dataset, so no
    # --allow-detector-transfer and the normal det-score 0.5 apply.
    "mega24": {
        "dataset":  "johnson-mouse-combined-aug",
        "model":    MEGA_MODEL_RUN,
        "kp_set":   "johnson",
        "out_name": "tailcycle",
        "sid_suffix": "",
    },
    # Previous production model (Johnson 500k-repaired, 24 kp). Kept as a
    # fallback: --variant johnson24.
    "johnson24": {
        "dataset":  DATASET_NAME,
        "model":    "",                 # predict_chunk.sh default
        "kp_set":   "johnson",
        "out_name": "tailcycle",
        "sid_suffix": "",
    },
    # 47-kp allen output from the universal mouse model (its registry holds
    # allen-mouse-combined too) + universal detector/scorer: newer than mega47
    # and trained with the johnson rigs in-domain. Same output dir and
    # predictions identity as mega47 -- they're alternatives, not companions.
    "univ47": {
        "dataset":  "allen-mouse-combined",
        "model":    UNIV_POSE_RUN,
        "detector": UNIV_DETECTOR_RUN,
        "scorer":   UNIV_SCORER_RUN,
        "kp_set":   "allen",
        "out_name": "tailcycle47",
        "sid_suffix": "__47",
    },
    "mega47": {
        "dataset":  "allen-mouse-combined",
        "model":    MEGA_MODEL_RUN,
        "kp_set":   "allen",
        "out_name": "tailcycle47",
        "sid_suffix": "__47",
    },
}
POLL_SECONDS      = 30

SESSION_RE = re.compile(r"^(\d{4}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}$")

LOCAL_TOOLS_DIR = Path(__file__).resolve().parent          # this file's dir


HARNESS_FILES = ["predict_chunk.sh", "merge_chunks.sh", "merge_chunks.py",
                 "session_to_tailcycle.py", "opencv_to_anipose_calibration.py"]


def _harness_dir() -> str:
    """Cluster dir for THIS version of the harness scripts, named by a hash of
    their content: <tools_dir>/tailcycle/v_<hash>.

    Every submit used to rsync the scripts into one shared dir while other
    sessions' arrays were executing them. bash reads a script incrementally,
    so replacing it mid-run kills the job at a stale offset (exit 127) or with
    ESTALE across the SMB->NFS mounts (cluster-job failure mode
    #1; happened-in-waiting here when predict_chunk.sh was edited on 09-24
    while a resume array ran). A content-hashed dir is write-once: a changed
    script gets a NEW dir, so no running job ever sees its file change."""
    import hashlib
    h = hashlib.sha1()
    for name in HARNESS_FILES:
        h.update(name.encode() + b"\0" + (LOCAL_TOOLS_DIR / name).read_bytes())
    return f"{TOOLS_REMOTE}/tailcycle/v_{h.hexdigest()[:12]}"


def _ssh(cmd: str, timeout: float = 60.0, check: bool = True) -> str:
    r = subprocess.run(["ssh", LOGIN, cmd], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh {cmd!r} failed rc={r.returncode}\n{r.stderr}")
    return r.stdout.strip()


def _cifs_available() -> bool:
    """True iff the cluster-home CIFS mount answers within 10 s. A dropped
    mount is an empty dir; a hung one blocks stat forever, hence the
    subprocess timeout instead of Path.is_dir()."""
    try:
        r = subprocess.run(["stat", "-f", str(LOCAL_CIFS / WORK_REL)],
                           capture_output=True, timeout=10)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _remote_rel(rel: str) -> str:
    """Cluster path for `rel`, relative to the cluster home (what both CIFS
    and ssh-rsync resolve against): bare names live under the work dir,
    '~/x' is home-relative, absolute paths pass through."""
    rel = str(rel)
    if rel.startswith("~/"):
        return rel[2:]
    if rel.startswith("/"):
        return rel
    return f"{WORK_REL}/{rel}"


def _run_rsync_retry(cmd: list[str], tries: int = 3) -> None:
    """rsync with retry. Both endpoints are network mounts (sshfs source,
    CIFS dest) that hiccup transiently — a 30 s I/O stall once killed three
    drivers mid-stage (exit 11) after hundreds of GB were already up. rsync
    is idempotent/delta-resuming, so retrying after a backoff is safe and
    resumes where it left off instead of losing the whole upload."""
    for attempt in range(1, tries + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as e:
            if attempt == tries:
                raise
            wait = 60 * attempt
            print(f"[tailcycle-cluster] rsync exit {e.returncode} "
                  f"(attempt {attempt}/{tries}) — retrying in {wait}s",
                  flush=True)
            time.sleep(wait)


def _require_cifs() -> None:
    """Bulk data moves ONLY over the SMB mount of the cluster home. The
    cluster forbids transfers through the login nodes (they killed ssh-rsync
    uploads on 2026-10-21), so a down mount is a hard stop, never a fallback
    to the login node."""
    if not _cifs_available():
        raise RuntimeError(
            f"cluster-home SMB mount {LOCAL_CIFS} is not available; data "
            "transfers through the login node are not allowed. Mount it "
            f"(cluster.smb_mount = {LOCAL_CIFS}) and retry.")


def _rsync_up(local: Path, remote_rel: str, extra: list[str] | None = None,
               checksum: bool = False) -> None:
    """Upload over the SMB mount of the cluster home (see _require_cifs)."""
    rel  = _remote_rel(remote_rel)
    if rel.startswith("/"):
        raise RuntimeError(f"{rel}: only cluster-home paths are reachable over SMB")
    _require_cifs()
    opts = ["-aL", "--info=progress2"]
    # TAILCYCLE_BWLIMIT (KB/s) paces uploads, e.g. 50000 when the source is
    # a network share whose owner caps the read rate
    if os.environ.get("TAILCYCLE_BWLIMIT"):
        opts.append(f"--bwlimit={int(os.environ['TAILCYCLE_BWLIMIT'])}")
    if checksum: opts.append("--checksum")
    if extra:    opts.extend(extra)
    dest = LOCAL_CIFS / rel
    dest.mkdir(parents=True, exist_ok=True)
    _run_rsync_retry(["rsync", *opts, str(local).rstrip("/") + "/",
                      str(dest).rstrip("/") + "/"])


def _rsync_down(remote_rel: str, local: Path) -> None:
    """Download over the SMB mount of the cluster home (see _require_cifs)."""
    local.parent.mkdir(parents=True, exist_ok=True)
    rel = _remote_rel(remote_rel)
    if rel.startswith("/"):
        raise RuntimeError(f"{rel}: only cluster-home paths are reachable over SMB")
    _require_cifs()
    subprocess.run(["rsync", "-a", str(LOCAL_CIFS / rel), str(local)], check=True)


def _date_of(session: str) -> str:
    m = SESSION_RE.match(session)
    if not m: raise SystemExit(f"bad session name {session!r}")
    return m.group(1)


def _staged_id(animal: str, session: str) -> str:
    return f"{animal}__{session}"




def _resolve_calib(date: str, calib_root: Path) -> tuple[Path | None, str | None]:
    """(calib_dir, calib_date) resolving in the same order the dashboard uses.
    The aniposelib toml builder reads OpenCV OR JARVIS-format Cam*.yaml, so any
    of these works as a source. Returns (None, None) if nothing is found."""
    # 1. OpenCV red_data for this exact date
    p = calib_root / date / "calibration"
    if p.is_dir() and any(p.glob("Cam*.yaml")):
        return p, date
    # 2. JARVIS-format for this exact date
    p = JARVIS_CALIB_ROOT / date
    if p.is_dir() and any(p.glob("Cam*.yaml")):
        return p, date
    # 3. nearest JARVIS-format date (rig is stationary; adjacent-day calib is
    #    a faithful stand-in — same rule as tools/cluster/run.py)
    def _ord(s: str) -> int:
        y, m, d = s.split("_")[:3]
        return int(y) * 10000 + int(m) * 100 + int(d)
    cands = []
    if JARVIS_CALIB_ROOT.is_dir():
        for d in JARVIS_CALIB_ROOT.iterdir():
            if d.is_dir() and re.match(r"^\d{4}_\d{2}_\d{2}$", d.name) \
               and any(d.glob("Cam*.yaml")):
                cands.append(d.name)
    if cands:
        nearest = min(cands, key=lambda s: abs(_ord(s) - _ord(date)))
        return JARVIS_CALIB_ROOT / nearest, nearest
    return None, None


def _gate_env(purpose: str) -> str:
    """Env prefix for every bsub, for a site spend gate (a bsub wrapper on
    the cluster PATH). Submissions come over ssh with no terminal, so the gate
    requires GATE_SESSION; GATE_PURPOSE covers the gate's projection check for
    a full array priced at -W. Values come from the [gate] config table:
    session, nosandbox (reason string, or "" to omit), extra (more env)."""
    env = {"GATE_SESSION": _cfg("gate.session", "tailcycle-run")}
    nosb = _cfg("gate.nosandbox", "")
    if nosb:
        env["GATE_NOSANDBOX"] = nosb
    env.update(_cfg("gate.extra", {}))
    env["GATE_PURPOSE"] = purpose
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "


def _site_env() -> str:
    """`VAR=value ...` for the cluster-side scripts (predict_chunk.sh,
    merge_chunks.sh): the work dir, default tailcyclenet checkout and pixi, and
    the default (johnson24) model + detector checkpoints. Everything comes from
    the config so the scripts hold no site paths."""
    env = {
        "TCR_WORK": "$HOME/" + WORK_REL,
        "TCR_REPO": _cfg("cluster.tailcycle_repo", "$HOME/tailcyclenet"),
        "TCR_PIXI": _cfg("cluster.pixi", "$HOME/.pixi/bin/pixi"),
        "TCR_DEFAULT_MODEL": _cfg("models.default_model", ""),
        "TCR_DEFAULT_DETECTOR": _cfg("models.default_detector", ""),
    }
    # values may reference $HOME on the cluster: double quotes, not shlex
    return " ".join(f'{k}="{v}"' for k, v in env.items()) + " "


def _parse_bsub_id(out: str) -> str:
    m = re.search(r"<(\d+)>", out)
    if not m: raise RuntimeError(f"cannot parse bsub id from {out!r}")
    return m.group(1)


def preflight_infer_flags(predict_chunk_sh: Path,
                           cli_py: Path) -> None:
    """Static check: every `--foo` flag our predict_chunk.sh passes to
    infer.py must appear as an add_argument('--foo', ...) in cli.py.

    Runs BEFORE the LSF array submit so a rename/removal upstream (like
    `--det-cache` disappearing or a new required `--max-animals`) fails in
    2 seconds locally instead of burning 30 seconds per array element on
    the cluster and only surfacing at chunk-log time. Cheap enough to run
    every submit.
    """
    if not cli_py.exists():
        print(f"[preflight] cli.py not found at {cli_py} — skipping flag check",
              flush=True)
        return
    valid = set()
    # every option string, aliases included (v2: add_argument('--pose', '--run', ...))
    for m in re.finditer(r"add_argument\(((?:\s*'--[a-z0-9-]+'\s*,)+)",
                         cli_py.read_text()):
        valid.update(re.findall(r"'(--[a-z0-9-]+)'", m.group(1)))
    if not valid:
        raise SystemExit(f"[preflight] failed to parse any flags from {cli_py}")

    text = predict_chunk_sh.read_text()
    # Extract every `--foo` of the infer.py invocation. Line-based: start at
    # the line containing scripts/infer.py and consume continuation lines
    # while they end with a backslash (works whether the call sits inline or
    # inside a helper function like run_infer()).
    lines = text.splitlines()
    block: list[str] = []
    for j, ln in enumerate(lines):
        if "scripts/infer.py" in ln:
            block.append(ln)
            k = j
            while k < len(lines) - 1 and lines[k].rstrip().endswith("\\"):
                k += 1
                block.append(lines[k])
            break
    if not block:
        raise SystemExit(f"[preflight] cannot find infer.py invocation in "
                          f"{predict_chunk_sh}")
    used = set(re.findall(r"--[a-z-]+", "\n".join(block)))
    unknown = sorted(used - valid)
    if unknown:
        raise SystemExit(
            f"[preflight] predict_chunk.sh uses infer.py flags that are NOT "
            f"in cli.py: {unknown}\n"
            f"  cli.py: {cli_py}\n"
            f"  predict_chunk.sh: {predict_chunk_sh}\n"
            f"Fix the shell script or wait for a pull/pixi install.")
    print(f"[preflight] {len(used)} infer.py flags in predict_chunk.sh all "
          f"valid against cli.py", flush=True)
    _preflight_repo_sync()


def _preflight_repo_sync() -> None:
    """Warn when the tailcyclenet checkouts drift.

    Model/detector checkpoints are trained against NEWER tailcyclenet code
    than our checkouts — running a new checkpoint on an old repo fails at
    load time (e.g. yolox 'hybrid' unknown on a 78-commit-stale repo,
    2026-08-26, which silently killed a full 11-session batch's smokes).
    Two cheap checks, both WARN-only (never block a submit on github being
    slow):
      1. local checkout vs origin (fetch, count commits behind)
      2. local HEAD vs cluster HEAD (they run different halves of the same
         pipeline and must match)
    """
    local_repo = LOCAL_TAILCYCLENET
    try:
        subprocess.run(["git", "-C", str(local_repo), "fetch", "-q",
                        "origin", "main"], timeout=20, check=False)
        behind = subprocess.run(
            ["git", "-C", str(local_repo), "rev-list", "--count",
             "HEAD..origin/main"],
            capture_output=True, text=True, timeout=10)
        n = int(behind.stdout.strip() or 0)
        if n > 0:
            print(f"[preflight] WARNING: local tailcyclenet is {n} commits "
                  f"behind origin/main — a checkpoint trained on newer code "
                  f"may fail to load. Update BOTH repos:\n"
                  f"  git -C {local_repo} pull\n"
                  f"  ssh {LOGIN} 'cd ~/tailcyclenet && git pull && "
                  f"~/.pixi/bin/pixi install --manifest-path "
                  f"~/tailcyclenet/pyproject.toml'", flush=True)
        local_head = subprocess.run(
            ["git", "-C", str(local_repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        cluster_head = _ssh(
            "cd ~/tailcyclenet && git rev-parse --short HEAD",
            check=False).strip()
        if local_head and cluster_head and local_head != cluster_head:
            print(f"[preflight] WARNING: tailcyclenet version SPLIT — local "
                  f"{local_head} vs cluster {cluster_head}. Preflight "
                  f"validates against local cli.py but the cluster runs its "
                  f"own checkout; sync them before trusting this run.",
                  flush=True)
    except Exception as e:
        print(f"[preflight] repo-sync check skipped ({e})", flush=True)


REPROJ_MAX_PX = 8.0     # exclude a camera whose median landmark reprojection
                         # error exceeds this (good cams measure < 1 px)


def _calib_reproj_errors(calib_dir: Path) -> dict[str, float]:
    """Median landmark-reprojection error per camera, in px, computed from
    the calibration solve's own artifacts (output/landmarks*.json next to
    the exported calibration/ dir). Empty dict when artifacts are absent
    (borrowed/jarvis-only dates) — callers treat that as 'unverified'."""
    import json as _json
    import cv2 as _cv2
    import numpy as _np
    root = calib_dir.parent
    lg_p = root / "output" / "landmarks_global.json"
    lm_p = root / "output" / "landmarks.json"
    if not (lg_p.exists() and lm_p.exists()):
        return {}
    lg = _json.loads(lg_p.read_text())
    lm = _json.loads(lm_p.read_text())
    ids3d = {i: p for i, p in zip(lg["ids"], lg["landmarks_global"])}
    out: dict[str, float] = {}
    for cam_serial, entry in lm.items():
        yaml = calib_dir / f"Cam{cam_serial}.yaml"
        if not yaml.exists():
            continue
        fs = _cv2.FileStorage(str(yaml), 0)
        K = fs.getNode("camera_matrix").mat()
        Dd = fs.getNode("distortion_coefficients").mat()
        rc = fs.getNode("rc_ext").mat()
        tc = fs.getNode("tc_ext").mat()
        fs.release()
        if K is None or rc is None:
            continue
        if Dd is None:
            Dd = _np.zeros(5)
        sel = [(ids3d[i], p2) for i, p2 in
               zip(entry["ids"], entry["landmarks"]) if i in ids3d]
        if not sel:
            continue
        P3 = _np.array([s[0] for s in sel], float)
        P2 = _np.array([s[1] for s in sel], float)
        rvec, _ = _cv2.Rodrigues(rc)
        proj, _ = _cv2.projectPoints(P3.reshape(-1, 1, 3), rvec,
                                      tc.astype(float), K,
                                      _np.asarray(Dd).ravel())
        err = _np.linalg.norm(proj.reshape(-1, 2) - P2[:, :2], axis=1)
        out[f"Cam{cam_serial}"] = float(_np.median(err))
    return out


def _reclaim_completed_staging() -> None:
    """rm tailcycle_videos/<sid> for every sid whose merged data3D.csv exists.
    A completed session's staged videos are dead weight; the normal cleanup
    runs in wait_and_fetch, but a killed/re-run driver skips it, so we also
    sweep here at the start of every submit. Also drops per-chunk session
    dirs once the merge is done (data3D.csv present) since those are only
    needed by the merge step."""
    # A staged video dir <sid> may be shared by MULTIPLE prediction runs
    # (<sid> for the 24-kp default and <sid>__47 for the mega variant), so
    # only reclaim the videos when EVERY predictions/<sid>* run that exists
    # has finished. Completion is keyed on the per-run DONE marker, NOT
    # data3D.csv: the merge writes DONE (merge_chunks.sh) and a RESUBMIT clears
    # it (submit(): rm -f .../DONE), whereas the old data3D.csv PERSISTS across
    # runs. Keying on data3D.csv let a concurrent submit's sweep delete the
    # staged videos out from under a re-submitted session's in-flight array
    # (its stale CSV read as "done") — which corrupted a live run. DONE is
    # absent for any active/re-submitted run, so it's the safe signal.
    cmd = (
        "for vd in " + CF_REMOTE + "/tailcycle_videos/*/; do "
        "  [ -d \"$vd\" ] || continue; sid=$(basename \"$vd\"); "
        # a fresh .staging lock = videos being staged for a run whose own
        # prediction dir/DONE state isn't set yet (2026-11-17: a sweep deleted
        # taco's pre-staged videos because its OLD 24-kp run had DONE)
        "  [ -n \"$(find \"$vd\" -maxdepth 1 -name .staging -mmin -1440)\" ] && continue; "
        "  all_done=1; any=0; "
        "  for pd in " + CF_REMOTE + "/predictions/$sid "
        "            " + CF_REMOTE + "/predictions/${sid}__*; do "
        "    [ -d \"$pd\" ] || continue; any=1; "
        "    [ -f \"$pd/DONE\" ] || all_done=0; "
        "  done; "
        "  if [ $any -eq 1 ] && [ $all_done -eq 1 ]; then "
        "    echo \"reclaim $sid\"; rm -rf \"$vd\"; "
        "  fi; "
        "  for pd in " + CF_REMOTE + "/predictions/$sid "
        "            " + CF_REMOTE + "/predictions/${sid}__*; do "
        "    [ -d \"$pd\" ] || continue; "
        "    [ -f \"$pd/DONE\" ] && rm -rf \"$pd/chunks\"; "
        "  done; "
        "done"
    )
    out = _ssh(cmd, check=False)
    for line in out.splitlines():
        if line.startswith("reclaim "):
            print(f"[tailcycle-cluster] reclaimed staging for {line[8:]}",
                  flush=True)


def _require_free_space_gb(min_free_gb: int) -> None:
    """Abort the submit if the cluster filesystem's free space is below the floor.
    Reads `df` once. Prevents a queue of jobs from filling the shared
    filesystem and wedging every user's jobs."""
    out = _ssh(f"df -BG --output=avail {CLUSTER_DF_PATH} | tail -1", check=False)
    try:
        free_gb = int(out.strip().rstrip("G"))
    except (ValueError, AttributeError):
        print(f"[tailcycle-cluster] WARN could not parse df output {out!r}; "
              f"proceeding without disk check", flush=True)
        return
    if free_gb < min_free_gb:
        raise SystemExit(
            f"[tailcycle-cluster] ABORT: only {free_gb} GB free on "
            f"{CLUSTER_DF_PATH} (floor {min_free_gb} GB). Not staging a new "
            f"~25-140 GB session. Let running jobs finish (they auto-clean "
            f"their staging on completion), then RE-CLICK this session to "
            f"resubmit -- it isn't auto-requeued.")
    print(f"[tailcycle-cluster] disk ok: {free_gb} GB free "
          f"(floor {min_free_gb})", flush=True)


def _run_smoke_test(sid: str, date: str, logs_rel: str, chunks_dir: str,
                     queue: str, project: str, slots: int, tools_dir: str,
                     smoke_frames: int = 24,
                     dataset: str = DATASET_NAME,
                     model_override: str = "",
                     video_id: str | None = None,
                     detector: str = "", scorer: str = "",
                     repo: str = "", det_score: str = "") -> bool:
    """Submit ONE bsub running predict_chunk.sh in smoke mode (24 frames),
    poll until it produces smoke/points3d.pq or a smoke.FAILED, and return
    True/False. Prints the real infer.py error on failure so the operator
    sees exactly what's broken WITHOUT burning the full array."""
    print(f"[tailcycle-cluster] smoke test: {smoke_frames} frames on 1 gpu "
          f"before committing full array…", flush=True)
    _ssh(f"rm -rf {chunks_dir}/smoke {chunks_dir}/smoke.FAILED "
         f"{chunks_dir}/smoke.range")
    # TAILCYCLE_SMOKE_END env clamps chunk 1 to [0, smoke_frames).
    smoke_cmd = (
        f"{_gate_env(f'tailcycle smoke test (24 frames) for {sid} before the full array')}"
        f"bsub -J 'sm_{sid}' -q {queue} -n {slots} -gpu 'num=1' "
        f"-W 1:00 -P {project} -o {logs_rel}/smoke_%J.log "
        f"env TAILCYCLE_SMOKE_END={smoke_frames} {_site_env()}"
        f"bash -l {tools_dir}/predict_chunk.sh "
        f"{sid} 32 {date} {dataset} '{model_override}' {video_id or sid} "
        f"'{detector}' '{scorer}' '{repo}' '{det_score}'"
    )
    out = _ssh(smoke_cmd, timeout=120)
    print(f"[tailcycle-cluster] smoke-bsub: {out}", flush=True)

    smoke_pq   = f"{chunks_dir}/smoke/points3d.pq"
    smoke_fail = f"{chunks_dir}/smoke.FAILED"
    # Status via the CIFS mount (free for the login node); ssh only if the mount is
    # down, and then with backoff — a smoke can sit in PEND for hours.
    cifs_chunks = LOCAL_CIFS / _remote_rel(f"predictions/{sid}/chunks")
    waited, ssh_delay = 0, POLL_SECONDS

    def _smoke_status() -> tuple[str, bool]:
        if _cifs_available():
            try:
                if (cifs_chunks / "smoke" / "points3d.pq").exists():
                    return "OK", True
                if (cifs_chunks / "smoke.FAILED").exists():
                    return "FAIL", True
                return "WAIT", True
            except OSError:
                pass
        return _ssh(f"test -f {smoke_pq} && echo OK || "
                    f"(test -f {smoke_fail} && echo FAIL || echo WAIT)",
                    check=False), False

    while True:
        st, via_cifs = _smoke_status()
        if st == "OK":
            print(f"[tailcycle-cluster] ✓ smoke test PASSED "
                  f"({waited}s) — committing full array", flush=True)
            return True
        if st == "FAIL":
            # Dump the real infer error from the smoke log. smoke.FAILED is
            # written by the job BEFORE it exits, but LSF writes the -o log
            # only when the job ENDS, so wait (bounded) for the log to land —
            # reading it immediately printed an empty "Real error" (2026-10-06).
            body = ""
            cifs_logs = LOCAL_CIFS / _remote_rel(f"predictions/{sid}/logs")
            for _ in range(18):                       # <= ~3 min
                try:
                    logs = sorted(cifs_logs.glob("smoke_*.log"),
                                  key=lambda p: p.stat().st_mtime)
                    txt = logs[-1].read_text(errors="replace") if logs else ""
                except OSError:
                    txt = ""
                if "Sender:" in txt:                  # LSF footer = complete
                    seg = txt.split("SMOKE MODE", 1)[-1].split("Sender:", 1)[0]
                    keep = [l for l in seg.splitlines()
                            if l.strip() and not re.search(
                                r"WARN cache|redirected to /tmp|platforms|libc|"
                                r"declare|[╭╰│┤├─]", l)]
                    body = "\n".join(keep[-25:])
                    break
                time.sleep(10)
            if not body:
                body = "(smoke log not readable yet: see predictions/" \
                       f"{sid}/logs/smoke_*.log on the cluster)"
            print(f"[tailcycle-cluster] ✗ smoke test FAILED. Real error:\n"
                  f"{body}", flush=True)
            return False
        step = POLL_SECONDS if via_cifs else ssh_delay
        if not via_cifs:
            ssh_delay = min(ssh_delay * 2, 600)
        time.sleep(step)
        waited += step
        # A genuinely broken smoke writes smoke.FAILED within ~minutes and is
        # caught above regardless of this ceiling. The ceiling only bounds
        # PENDing — and under self-contention (our own 32-way arrays saturate
        # gpu_l4) a smoke can sit in PEND for a long time. 30 min was too
        # tight and spuriously failed sessions; 3 h tolerates a jammed queue
        # while still bailing on a truly hung job.
        if waited > 10800:
            print("[tailcycle-cluster] smoke test timed out (>3 h in queue)",
                  flush=True)
            return False


PRESENCE_MERGE_S = 5.0     # run straight through absences shorter than this
MIN_CHUNK_FRAMES = 2000    # below this, per-chunk model load dominates


def _presence_ranges(presence_json: Path, num_chunks: int) -> list[list[int]] | None:
    """Chunk ranges covering only the frames with the mouse in the arena, from
    tools/yolo_presence.py's presence.json; None when the session has no
    presence track (then the whole recording is split evenly, as before).

    Intervals closer than PRESENCE_MERGE_S are merged (a short absence costs
    less than another chunk's model load), then each is cut into near-equal
    pieces of about total/num_chunks frames (at least MIN_CHUNK_FRAMES), so
    the array is at most ~num_chunks wide plus one per extra interval."""
    if not presence_json.exists():
        return None
    pj = json.loads(presence_json.read_text())
    gap = int(PRESENCE_MERGE_S * pj.get("fps", 180.0))
    merged: list[list[int]] = []
    for s, e in sorted(pj["intervals"]):
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    total = sum(e - s for s, e in merged)
    if total == 0:
        return []
    target = max(-(-total // num_chunks), MIN_CHUNK_FRAMES)
    out = []
    for s, e in merged:
        k = -(-(e - s) // target)
        edges = [s + round(j * (e - s) / k) for j in range(k + 1)]
        out += [[a, b] for a, b in zip(edges, edges[1:]) if b > a]
    return out


def submit(animal: str, session: str, *, num_chunks: int,
            queue: str, project: str, wall: str, slots: int,
            video_root: Path, calib_root: Path,
            pred_root: Path = DEFAULT_PRED_ROOT,
            variant: str = "univ24",
            presence_gate: bool = True,
            calib_date: str | None = None,
            ranges_file: Path | None = None) -> dict:
    var  = VARIANTS[variant]
    vid_sid = _staged_id(animal, session)       # video staging identity
    sid  = vid_sid + var["sid_suffix"]          # predictions identity
    date = _date_of(session)
    videos = video_root / animal / session

    if not videos.is_dir() or not any(videos.glob("Cam*.mp4")):
        raise FileNotFoundError(f"no Cam*.mp4 in {videos}")
    # Resolve calib the same way the dashboard does: OpenCV red_data for the
    # date, else JARVIS-format jarvis_calibrations for the date, else the
    # nearest JARVIS date. The aniposelib-toml builder auto-detects format.
    # calib_date forces a specific day's calibration (a session with no
    # same-day calib whose nearest date fits worse than another, checked by
    # background shift) instead of the nearest-date fallback.
    calib, calib_date = _resolve_calib(calib_date or date, calib_root)
    if calib is None:
        raise FileNotFoundError(
            f"no calib for {date} (checked {calib_root}/{date}/calibration, "
            f"{JARVIS_CALIB_ROOT}/{date}, and nearest date)")

    # Preflight: catch API drift before burning cluster time.
    preflight_infer_flags(
        predict_chunk_sh=LOCAL_TOOLS_DIR / "predict_chunk.sh",
        cli_py=(Path(var["repo"].replace(str(CLUSTER_HOME).rstrip("/"),
                                          str(LOCAL_CIFS).rstrip("/")))
                if var.get("repo") else LOCAL_TAILCYCLENET)
               / "tailcyclenet/infer/cli.py",
    )

    # Disk hygiene, run every submit so a queue of many jobs stays bounded:
    #   1. Reclaim staged videos for any session already merged (a killed
    #      driver that never reached its own post-pull cleanup leaks ~25-140 GB;
    #      this sweeps them on the NEXT job's submit).
    #   2. Refuse to stage if free space is below the floor -- better to fail
    #      one job with a clear message than fill the cluster filesystem and wedge
    #      everyone's jobs (the "Disk quota exceeded" incident).
    _reclaim_completed_staging()
    _require_free_space_gb(min_free_gb=500)

    # Stage ONLY the usable cameras:
    #   1. calibrated — a stray non-rig camera (e.g. Cam710040) absent from
    #      the calibration can't be triangulated, must never reach the cluster
    #   2. not blurry / not unreadable — from the sanitize audit's blur cache
    #      (cache-only read: if the session was never blur-scanned we stage
    #      all calibrated cams, same as before).
    # Bad-but-calibrated cams are ALSO excluded from the calibration.toml
    # below, so tailcyclenet's name-matching ignores them entirely.
    calib_cams = sorted(y.stem for y in Path(calib).glob("Cam*.yaml"))
    # A calibrated camera that wasn't recorded this session (remus 07_20
    # has 13 of the 16 cams solved on 07_16) is dropped from the
    # calibration.toml too, or infer.py refuses the group with a hole.
    no_video = {c for c in calib_cams
                if not (videos / f"{c}.mp4").exists()}
    if no_video:
        print(f"[tailcycle-cluster] calibrated cams with no video: "
              f"{sorted(no_video)}", flush=True)
    bad_cams: set[str] = set(no_video)
    try:
        from tailcycle_run.vendor import blur_audit
        # compute_if_missing=True: the blur check is MANDATORY — an
        # unscanned session must not silently stage a blurry camera
        # ("i want blurry cam to not contribute to the prediction").
        # Costs ~10-20 s on a cache miss; cached afterwards.
        bl = blur_audit.session_blur(animal, session, videos,
                                      compute_if_missing=True)
        if bl.get("checked"):
            video_cams = {p.stem for p in videos.glob("Cam*.mp4")}
            bad_cams |= set(bl.get("blurry", []))
            bad_cams |= video_cams - set(bl.get("sharpness", {}))  # unreadable
            bad_cams &= set(calib_cams)   # only calibrated cams matter here
            if bl.get("blurry"):
                print(f"[tailcycle-cluster] blurry cams excluded: "
                      f"{sorted(bl['blurry'])}", flush=True)
    except Exception as e:                 # sanitize is best-effort
        print(f"[tailcycle-cluster] blur-cache read failed ({e}); "
              f"staging all calibrated cams", flush=True)
    # Reprojection-error gate: project the calibration solve's own board
    # landmarks through the EXPORTED per-cam yamls; a camera whose median
    # error exceeds the threshold is excluded like a blurry one ("i want
    # low reprojection error"). Only measurable for cams that saw the
    # world-frame board; others are reported as unverified.
    try:
        reproj = _calib_reproj_errors(Path(calib))
        if reproj:
            bad_rp = {c for c, e in reproj.items() if e > REPROJ_MAX_PX}
            msg = ", ".join(f"{c}={e:.1f}px" for c, e in sorted(reproj.items()))
            print(f"[tailcycle-cluster] calib reproj (median px): {msg}",
                  flush=True)
            unmeasured = set(calib_cams) - set(reproj)
            if unmeasured:
                print(f"[tailcycle-cluster] reproj unverified (no board "
                      f"landmarks): {sorted(unmeasured)}", flush=True)
            if (Path(calib).parent / "REPROJ_GATE_WAIVED").exists():
                # user-approved calibration despite elevated reproj errors
                # (e.g. 2026_09_09: registration recorded with the climb
                # structure installed — errors are uniform-moderate, user
                # chose to accept). Waive exclusion/refusal for this date.
                print(f"[tailcycle-cluster] reproj gate WAIVED for this "
                      f"date (marker file present)", flush=True)
                bad_rp = set()
            if len(bad_rp) > len(reproj) / 2:
                # MAJORITY of measured cams over threshold = the solve's
                # output artifacts are from a DIFFERENT run than the exported
                # yamls (stale bad-watcher-era leftovers) — the measurement
                # is meaningless, not the cameras. Do NOT exclude; demand a
                # re-solve so the date becomes verifiable again.
                raise SystemExit(
                    f"[tailcycle-cluster] calib UNVERIFIABLE for {date}: "
                    f"{len(bad_rp)}/{len(reproj)} measured cams exceed "
                    f"{REPROJ_MAX_PX}px — solve artifacts out of sync with "
                    f"exported yamls. Re-run the calibration pipeline for "
                    f"this date before predicting.")
            if bad_rp:
                print(f"[tailcycle-cluster] HIGH-REPROJ cams excluded "
                      f"(> {REPROJ_MAX_PX}px): {sorted(bad_rp)}", flush=True)
                bad_cams |= bad_rp & set(calib_cams)
    except Exception as e:
        print(f"[tailcycle-cluster] reproj check failed ({e}); skipping",
              flush=True)
    # Realigned-video overrides: fix_desync_video.py rebuilds a
    # frame-dropping camera onto the reference trigger timeline and writes it
    # to <pred>/<session>/aligned/Cam*.mp4. Where an override exists it
    # REPLACES the mount's copy for both the desync count and staging.
    aligned_dir = None
    try:
        # Use the SAME pred_root the driver writes/reads predictions under, so
        # aligned overrides are found where fix_desync_video.py wrote them.
        # (Falls back to local.fallback_pred_root for a non-default video root.)
        pr = Path(pred_root) if pred_root else Path(
            DEFAULT_PRED_ROOT if Path(video_root) == DEFAULT_VIDEO_ROOT
            else FALLBACK_PRED_ROOT)
        aligned_dir = pr / animal / session / "aligned"
    except Exception:
        aligned_dir = None
    overrides = ({p.stem: p for p in aligned_dir.glob("Cam*.mp4")}
                 if aligned_dir and aligned_dir.is_dir() else {})
    if overrides:
        print(f"[tailcycle-cluster] realigned overrides: "
              f"{sorted(overrides)}", flush=True)
    # Desync gate: a camera whose frame count deviates from the session
    # median by more than the usual dropped-trigger 1-2 frames is temporally
    # MISALIGNED under tailcycle's index-based --trim-to-shortest alignment
    # (measured: tata/2026_04_15 Cam2006055 was 685 frames short with a
    # progressive drift — its 2D was seconds off and pulled triangulation to
    # the wrong part of the arena). Exclude it like a blurry cam.
    try:
        import cv2, statistics
        counts = {}
        for c in calib_cams:
            mp4 = overrides.get(c) or (videos / f"{c}.mp4")
            if not Path(mp4).exists():
                continue
            cap = cv2.VideoCapture(str(mp4))
            counts[c] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if counts:
            med = statistics.median(counts.values())
            desync = {c for c, n in counts.items() if abs(n - med) > 2}
            if desync:
                print(f"[tailcycle-cluster] DESYNC cams excluded "
                      f"(frame count vs median {int(med)}): "
                      f"{sorted((c, counts[c]) for c in desync)}", flush=True)
                bad_cams |= desync
    except Exception as e:
        print(f"[tailcycle-cluster] desync check failed ({e}); skipping",
              flush=True)
    allowed_cams = [c for c in calib_cams if c not in bad_cams]
    if len(allowed_cams) < 8:
        raise SystemExit(
            f"[tailcycle-cluster] only {len(allowed_cams)} usable cameras "
            f"(calibrated + sharp + readable) — refusing to burn cluster "
            f"time on a degenerate triangulation. bad={sorted(bad_cams)}")
    if allowed_cams:
        cam_includes = [f"--include={c}.mp4" for c in allowed_cams]
        skipped = sorted(p.stem for p in videos.glob("Cam*.mp4")
                         if p.stem not in set(allowed_cams))
        if skipped:
            print(f"[tailcycle-cluster] excluding cameras: {skipped} "
                  f"(non-calibrated, blurry, or unreadable)", flush=True)
    else:
        cam_includes = ["--include=Cam*.mp4"]
    print(f"[tailcycle-cluster] staging videos: {videos}", flush=True)
    stage_lock = LOCAL_CIFS / _remote_rel(f"tailcycle_videos/{vid_sid}") / ".staging"
    stage_lock.parent.mkdir(parents=True, exist_ok=True)
    stage_lock.touch()                       # keeps the reclaim sweep off (see there)
    # drivers still running OLD code sweep without the lock check; an
    # existing predictions/<sid> dir without DONE also keeps them off
    pred_dir = LOCAL_CIFS / _remote_rel(f"predictions/{sid}")
    if not (pred_dir / "DONE").exists():
        pred_dir.mkdir(parents=True, exist_ok=True)
    _rsync_up(videos, f"tailcycle_videos/{vid_sid}",
               extra=[*cam_includes, "--exclude=*"])
    # realigned overrides go up AFTER the main sync so they overwrite the
    # mount's desynced copies in the staged dir
    for c, p in overrides.items():
        if c in allowed_cams:
            print(f"[tailcycle-cluster] staging realigned override {c}",
                  flush=True)
            _rsync_up(p.parent, f"tailcycle_videos/{vid_sid}",
                       extra=[f"--include={c}.mp4", "--exclude=*"])

    # NEW pipeline: tailcycle infer.py reads an aniposelib calibration.toml
    # directly via --calibration, so we build one aniposelib TOML from our
    # 16 Cam*.yaml files locally and only upload that single file.
    # When this session excludes cameras (blurry/unreadable), the toml is
    # session-specific — upload it under <date>__<sid> instead of the shared
    # <date> dir so other sessions on the same date keep the full calib.
    calib_ref = f"{date}__{sid}" if bad_cams else date
    # frame the resulting 3D will be in: canonical iff the source calib has
    # been world-aligned (its .world_aligned marker is present)
    world_frame = ("canonical" if (Path(calib) / ".world_aligned").exists()
                   else "dateframe")
    import tempfile
    print(f"[tailcycle-cluster] building calibration.toml from {calib}"
          + (f" excluding {sorted(bad_cams)}" if bad_cams else ""),
          flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        toml_local = Path(tmp) / "calibration.toml"
        cmd = [LOCAL_PYTHON,
               str(LOCAL_TOOLS_DIR / "opencv_to_anipose_calibration.py"),
               "--calib-dir", str(calib),
               "--out",       str(toml_local)]
        if bad_cams:
            cmd += ["--exclude", ",".join(sorted(bad_cams))]
        subprocess.run(cmd, check=True)
        print(f"[tailcycle-cluster] uploading calibration.toml → cluster",
              flush=True)
        _rsync_up(toml_local.parent, f"tailcycle_calibs/{calib_ref}",
                   extra=["--include=calibration.toml", "--exclude=*"])

    tools_dir = _harness_dir()
    if _ssh(f"test -f {tools_dir}/.complete && echo yes || true",
            check=False).strip() == "yes":
        print(f"[tailcycle-cluster] harness scripts already deployed: {tools_dir}",
              flush=True)
    else:
        # Fresh, content-addressed dir: nothing is running from it yet, so
        # uploading into it can't disturb a job. .complete marks it usable.
        print(f"[tailcycle-cluster] deploying harness scripts -> {tools_dir}",
              flush=True)
        _ssh(f"mkdir -p {tools_dir}")
        # keep the leading "~/": _rsync_up routes non-work-dir dests over
        # ssh; without it the scripts would land under the work dir via CIFS
        _rsync_up(LOCAL_TOOLS_DIR, tools_dir,
                   checksum=True,
                   extra=[*(f"--include={n}" for n in HARNESS_FILES),
                          "--exclude=*"])
        _ssh(f"chmod +x {tools_dir}/*.sh && touch {tools_dir}/.complete")

    # Pre-download VJEPA2 encoder weights ONCE. Without this every array
    # element races to write to the same ~/.cache/torch/hub/checkpoints/
    # file, and on the shared NFS half of them hit "Errno 116 Stale file
    # handle" mid-load (measured: 8/16 succeeded, 8/16 crashed at
    # torch.load). Doing it here serialises the download to a single
    # writer -- the array elements then see a stable file.
    print("[tailcycle-cluster] pre-downloading VJEPA2 weights (once)...",
          flush=True)
    _ssh(
        "mkdir -p ~/.cache/torch/hub/checkpoints && "
        "F=~/.cache/torch/hub/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt && "
        "if [ ! -s \"$F\" ] || [ $(stat -c%s \"$F\") -lt 4000000000 ]; then "
        "  curl -fsSL -o \"$F.tmp\" "
        "    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt "
        "  && mv -f \"$F.tmp\" \"$F\"; "
        "fi",
        timeout=900,  # 4.8 GB @ ~140 MB/s ≈ 40 s but pad for slow first hit
    )

    # Mouse-presence gate: predict only where the top-down YOLO track has the
    # mouse in the arena (presence.json from tools/yolo_presence.py). The
    # ranges go to predictions/<sid>/ranges.txt, which predict_chunk.sh reads.
    ranges = (None if not presence_gate else
              _presence_ranges(Path(pred_root) / animal / session / "presence.json",
                               num_chunks))
    # Explicit windows (e.g. mined rear/climb bouts): one "FS FE" line per chunk,
    # half-open source-frame ranges. Overrides the presence gate. Frames outside
    # the windows stay NaN in the merged data3D.csv (row = video frame).
    if ranges_file is not None:
        ranges = []
        for ln in Path(ranges_file).read_text().splitlines():
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            fs, fe = (int(x) for x in ln.split()[:2])
            if not 0 <= fs < fe:
                raise SystemExit(f"[tailcycle-cluster] bad range {ln!r} in {ranges_file}")
            ranges.append([fs, fe])
        print(f"[tailcycle-cluster] explicit windows from {ranges_file}: "
              f"{len(ranges)} chunks", flush=True)
    if ranges is not None:
        if not ranges:
            raise SystemExit("[tailcycle-cluster] presence.json has no frames with "
                             "the mouse in the arena -- nothing to predict")
        n_pres = sum(e - s for s, e in ranges)
        print(f"[tailcycle-cluster] presence gate: {n_pres:,} frames with the "
              f"mouse in the arena, {len(ranges)} chunks", flush=True)
        num_chunks = len(ranges)
    ranges_txt = "".join(f"{s} {e}\n" for s, e in ranges or [])

    chunks_dir = f"{CF_REMOTE}/predictions/{sid}/chunks"
    logs_rel   = f"{CF_REMOTE}/predictions/{sid}/logs"
    _ssh(f"mkdir -p {chunks_dir} {logs_rel}")
    _require_cifs()
    ranges_file = LOCAL_CIFS / _remote_rel(f"predictions/{sid}/ranges.txt")
    if ranges:
        ranges_file.parent.mkdir(parents=True, exist_ok=True)  # CIFS may lag the ssh mkdir
        ranges_file.write_text(ranges_txt)
    else:
        ranges_file.unlink(missing_ok=True)
    # Clear DONE/FAILED but keep the chunks/ contents.
    _ssh(f"rm -f {CF_REMOTE}/predictions/{sid}/DONE "
         f"{CF_REMOTE}/predictions/{sid}/FAILED "
         f"{chunks_dir}/*.FAILED "
         f"{CF_REMOTE}/tailcycle_videos/{vid_sid}/.staging")

    # Resume is only valid if the chunks were produced from the SAME inputs.
    # A resubmit after a realign / cam exclusion change / calib change must not
    # reuse old chunks (they'd be triangulated from a different camera set and
    # merged into one CSV with the new ones). Fingerprint the inputs; on a
    # mismatch, wipe chunks/ so the run starts fresh.
    ov_sig = ",".join(
        f"{c}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
        for c, p in sorted(overrides.items()) if c in allowed_cams)
    fingerprint = (f"cams={','.join(sorted(allowed_cams))};"
                   f"overrides={ov_sig};calib={calib_ref};"
                   f"chunks={num_chunks};variant={variant};"
                   f"model={var['model'] or 'default'};"
                   f"dataset={var['dataset']}"
                   + (f";detector={var['detector']}" if var.get("detector") else "")
                   + (f";scorer={var['scorer']}" if var.get("scorer") else "")
                   + (f";repo={var['repo']}" if var.get("repo") else "")
                   + (f";det_score={var['det_score']}" if var.get("det_score") else "")
                   + (f";ranges={hashlib.sha1(ranges_txt.encode()).hexdigest()[:12]}"
                      if ranges else ""))
    fp_remote = f"{CF_REMOTE}/predictions/{sid}/run_fingerprint"
    prev_fp = _ssh(f"cat {fp_remote} 2>/dev/null || true", check=False).strip()
    if prev_fp and prev_fp != fingerprint:
        print(f"[tailcycle-cluster] inputs changed since the existing "
              f"chunks were produced -- discarding them (no resume).\n"
              f"  was: {prev_fp}\n  now: {fingerprint}", flush=True)
        _ssh(f"rm -rf {chunks_dir} && mkdir -p {chunks_dir}")
    elif not prev_fp:
        # Run from before fingerprints existed. Those were ALL johnson24 in the
        # plain <sid> dir, or mega47 in <sid>__47. Their chunks are only safe to
        # resume under that same model — e.g. a mega24 run must not merge
        # leftover johnson24 chunks into its CSV.
        legacy_variant = "mega47" if var["sid_suffix"] else "johnson24"
        if variant != legacy_variant:
            print(f"[tailcycle-cluster] unfingerprinted chunks here came from "
                  f"{legacy_variant}; this run is {variant} -- discarding them.",
                  flush=True)
            _ssh(f"rm -rf {chunks_dir} && mkdir -p {chunks_dir}")
    if prev_fp != fingerprint:
        _ssh(f"printf '%s' {shlex.quote(fingerprint)} > {fp_remote}")

    # Which chunks are already done? (resume support — cluster preemption can
    # kill 2-3 chunks after hours; we resubmit only the missing indices.)
    # Done = the .range / .EMPTY marker predict_chunk.sh writes AFTER infer
    # exits 0. points3d.pq alone is not enough: infer writes it incrementally,
    # so a chunk that crashed 64 frames in leaves a partial one (2026-10-21:
    # five taco chunks resumed as "done" with ~1% coverage).
    listing = _ssh(f"ls {chunks_dir}/chunk_*.range {chunks_dir}/chunk_*.EMPTY "
                   f"2>/dev/null || true", check=False)
    done_idxs = set()
    for name in listing.splitlines():
        m = re.search(r"chunk_(\d+)\.(range|EMPTY)$", name)
        if m: done_idxs.add(int(m.group(1)))
    missing_idxs = [i for i in range(1, num_chunks + 1) if i not in done_idxs]

    # SMOKE TEST: on a FRESH run (nothing done yet), validate the entire real
    # path on ONE gpu over a tiny 24-frame range before committing the full
    # K-way array. Catches everything the flag-preflight can't -- cam-regex
    # mismatches, calibration parse errors, model/detector load, the actual
    # forward -- in ~5 min instead of burning K gpus × (repeatedly).
    if not done_idxs:
        smoke_ok = _run_smoke_test(sid, calib_ref, logs_rel, chunks_dir,
                                    queue, project, slots, tools_dir,
                                    dataset=var["dataset"],
                                    model_override=var["model"],
                                    video_id=vid_sid,
                                    detector=var.get("detector", ""),
                                    scorer=var.get("scorer", ""),
                                    repo=var.get("repo", ""),
                                    det_score=var.get("det_score", ""))
        if not smoke_ok:
            raise SystemExit(
                "[tailcycle-cluster] SMOKE TEST FAILED — not submitting the "
                "full array. See the error above.")

    if not missing_idxs:
        print(f"[tailcycle-cluster] all {num_chunks} chunks already produced "
              f"from a prior run -- skipping array, going straight to merge",
              flush=True)
    else:
        print(f"[tailcycle-cluster] resume: {len(done_idxs)}/{num_chunks} chunks "
              f"already done, resubmitting only {missing_idxs}", flush=True)

    arr_id = None
    if missing_idxs:
        idx_spec = ",".join(str(i) for i in missing_idxs)   # LSF: '[1,3,7]'
        array_cmd = (
            f"{_gate_env(f'tailcycle 3D pose prediction ({variant}) for {sid}; produces predictions/{sid}/data3D.csv')}"
            f"bsub -J 'pc_{sid}[{idx_spec}]' -q {queue} -n {slots} "
            f"-gpu 'num=1' -W {wall} -P {project} "
            f"-o {logs_rel}/pc_%I_%J.log "
            f"env {_site_env()}bash -l {tools_dir}/predict_chunk.sh "
            f"{sid} {num_chunks} {calib_ref} {var['dataset']} "
            f"'{var['model']}' {vid_sid} "
            f"'{var.get('detector', '')}' '{var.get('scorer', '')}' "
            f"'{var.get('repo', '')}' '{var.get('det_score', '')}'"
        )
        print(f"[tailcycle-cluster] submitting {len(missing_idxs)}/{num_chunks}"
              f"-way array…", flush=True)
        # bsub can take 1-3 min on gpu_l4 when -n 16 forces cross-node
        # scheduling. Default _ssh timeout of 60 s isn't enough — bump.
        out = _ssh(array_cmd, timeout=300)
        print(f"[tailcycle-cluster] array-bsub: {out}", flush=True)
        arr_id = _parse_bsub_id(out)

    # Merge always runs; only gate on array dep when there IS an array.
    dep = f" -w 'ended({arr_id})'" if arr_id else ""
    merge_cmd = (
        f"{_gate_env(f'merge tailcycle chunks for {sid} into data3D.csv')}"
        f"bsub -J 'mc_{sid}' -q short -n 2 -W 1:00 -P {project}"
        f"{dep} "
        f"-o {logs_rel}/mc_%J.log "
        f"env {_site_env()}bash -l {tools_dir}/merge_chunks.sh {sid} {num_chunks} "
        f"{var['kp_set']}"
    )
    out = _ssh(merge_cmd, timeout=300)
    print(f"[tailcycle-cluster] merge-bsub: {out}", flush=True)
    merge_id = _parse_bsub_id(out)

    return {"staged_id":   sid,
            "variant":     variant,
            "model_run":   var["model"],
            "kp_dataset":  var["dataset"],
            "detector_run": var.get("detector", ""),
            "scorer_run":  var.get("scorer", ""),
            "out_name":    var["out_name"],
            "num_chunks":  num_chunks,
            "array_id":    arr_id,
            "merge_id":    merge_id,
            "date":        date,
            # the date whose calibration the 3D is reconstructed in — may
            # differ from the session date (nearest-date fallback). The
            # reprojection info.yaml MUST reference this one, not `date`.
            "calib_date":  calib_date,
            "excluded_cams": sorted(bad_cams),
            "frame":       world_frame,
            "video_root":  str(video_root),
            "done_path":   f"{CF_REMOTE}/predictions/{sid}/DONE",
            "failed_path": f"{CF_REMOTE}/predictions/{sid}/FAILED"}


def wait_and_fetch(animal: str, session: str, submission: dict,
                    pred_root: Path) -> Path:
    sid = submission["staged_id"]
    done, failed = submission["done_path"], submission["failed_path"]
    # The same markers seen through the CIFS mount of the cluster home. Reading
    # them there costs the login node nothing; the old loop ran
    # `ssh <login> test -f` every 30 s for hours per driver (~14 logins/min with
    # 7 drivers), which cluster rules forbid (no unattended hours-long
    # watches on the login node; poll with backoff from 1 min, doubling).
    cifs_pred = LOCAL_CIFS / _remote_rel(f"predictions/{sid}")
    delay, delay_max = 60, 600
    unreachable_since = None

    def _check() -> str:
        if _cifs_available():
            try:
                if (cifs_pred / "DONE").exists():
                    return "DONE"
                if (cifs_pred / "FAILED").exists():
                    return "FAILED"
                return "WAIT"
            except OSError:
                pass            # stale mount: fall through to ssh this round
        return _ssh(f"test -f {done} && echo DONE || "
                    f"(test -f {failed} && echo FAILED || echo WAIT)")

    while True:
        # Runs for hours; a transient drop (ssh rc=255 / mount hiccup) must not
        # kill a driver whose array is already running (it stranded 2 completed
        # sessions on 2026-08-29). Give up only after 30 min of continuous
        # unreachability — measured in time, since the interval now grows.
        try:
            out = _check()
            unreachable_since = None
        except Exception as e:
            unreachable_since = unreachable_since or time.time()
            if time.time() - unreachable_since > 1800:
                raise RuntimeError(f"cluster markers unreachable for 30 min: {e}")
            print(f"[tailcycle-cluster] poll failed ({e.__class__.__name__}) "
                  f"— retrying in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, delay_max)
            continue
        if out == "DONE":
            break
        if out == "FAILED":
            try:
                body = (cifs_pred / "FAILED").read_text()[:4000]
            except OSError:
                body = _ssh(f"head -40 {failed}", check=False)
            raise RuntimeError(f"cluster tailcycle failed: {body}")
        time.sleep(delay)
        delay = min(delay * 2, delay_max)

    out_dir = pred_root / animal / session / submission.get("out_name",
                                                             "tailcycle")
    _rsync_down(f"predictions/{sid}/data3D.csv",
                out_dir / "data3D.csv")
    # Framewise scorer quality (only variants with a scorer). A quality.csv
    # left from a previous run with a DIFFERENT model must not sit beside this
    # data3D.csv, so drop it when this run had no scorer.
    q_local = out_dir / "quality.csv"
    if submission.get("scorer_run"):
        try:
            _rsync_down(f"predictions/{sid}/quality.csv", q_local)
            print(f"[tailcycle-cluster] pulled -> {q_local}", flush=True)
        except Exception as e:
            print(f"[tailcycle-cluster] WARN no quality.csv pulled ({e}); "
                  f"data3D.csv is fine", flush=True)
    elif q_local.exists():
        q_local.unlink()

    # dataset_name must be the calibration the 3D was RECONSTRUCTED in (the
    # calib_date, which differs from the session date on nearest-date
    # fallback) — reprojecting with any other calib puts kp in the wrong
    # place (see docs/free_dataset.md issue #7). Convert the OpenCV calib to
    # JARVIS format if that date has no jarvis_calibrations dir yet.
    cal_date = submission.get("calib_date") or submission["date"]
    jdir = JARVIS_CALIB_ROOT / cal_date
    if not jdir.is_dir():
        src = LOCAL_OPENCV_CALIB_ROOT / cal_date / "calibration"
        if src.is_dir():
            subprocess.run(
                [LOCAL_PYTHON,
                 str(LOCAL_TOOLS_DIR.parent / "vendor" / "convert_calib_generic.py"),
                 str(src), str(jdir)],
                check=False)
    (out_dir / "info.yaml").write_text(
        f"recording_path: {submission.get('video_root', DEFAULT_VIDEO_ROOT)}"
        f"/{animal}/{session}\n"
        f"dataset_name: {JARVIS_CALIB_ROOT / cal_date}\n"
        f"model: tailcyclenet"
        f"{'-' + submission['variant'] if submission.get('variant') in ('mega47', 'univ47') else ''}\n"
        f"variant: {submission.get('variant', 'johnson24')}\n"
        f"model_run: {submission.get('model_run') or 'predict_chunk.sh default (johnson 500k-repaired)'}\n"
        f"kp_dataset: {submission.get('kp_dataset', '')}\n"
        f"detector_run: {submission.get('detector_run') or 'predict_chunk.sh default (johnson v3best)'}\n"
        + (f"scorer_run: {submission['scorer_run']}\n" if submission.get('scorer_run') else "")
        + f"cluster_array_id: {submission['array_id']}\n"
        f"cluster_merge_id: {submission['merge_id']}\n"
        f"num_chunks: {submission['num_chunks']}\n"
        + (f"excluded_cams: {','.join(submission['excluded_cams'])}\n"
           if submission.get("excluded_cams") else "")
        # Stamp which world frame the 3D is in, decided at toml-build time by
        # whether the source calib was world-aligned. Lets apply_world_
        # alignment skip already-canonical predictions (race-proof).
        + (f"aligned_frame: {submission.get('frame', 'dateframe')}\n")
    )
    print(f"[tailcycle-cluster] pulled -> {out_dir / 'data3D.csv'}", flush=True)

    # Reclaim disk on cluster. Chunks are ours alone; the staged VIDEOS may
    # be shared with the other model variant (sid vs sid__47), so leave them
    # to _reclaim_completed_staging, which only removes videos once every
    # variant's data3D.csv exists.
    try:
        _ssh(f"rm -rf {CF_REMOTE}/predictions/{sid}/chunks", check=False)
        _reclaim_completed_staging()
    except Exception as e:
        print(f"[tailcycle-cluster] WARN cleanup failed: {e}", flush=True)
    return out_dir / "data3D.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animal",  required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--num-chunks", type=int, default=DEFAULT_CHUNKS)
    ap.add_argument("--queue",   default=DEFAULT_QUEUE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--wall",    default=DEFAULT_WALL)
    ap.add_argument("--slots",   type=int, default=DEFAULT_SLOTS)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    ap.add_argument("--pred-root",  type=Path, default=DEFAULT_PRED_ROOT)
    ap.add_argument("--calib-root", type=Path, default=LOCAL_OPENCV_CALIB_ROOT)
    ap.add_argument("--variant", choices=sorted(VARIANTS),
                    default="univ24",
                    help="univ24 = universal mouse pose + detector + scorer, "
                          "24-kp johnson output into <session>/tailcycle/ "
                          "(+ quality.csv) (default); "
                          "mega24 = mega-mouse model, 24-kp johnson output "
                          "into <session>/tailcycle/; "
                          "johnson24 = previous 24-kp model, same output; "
                          "mega47 = mega-mouse model, 47-kp allen output "
                          "into <session>/tailcycle47/")
    ap.add_argument("--no-presence-gate", action="store_true",
                    help="predict the whole recording even if presence.json "
                         "says the mouse is out of the arena for part of it")
    ap.add_argument("--ranges-file", type=Path, default=None,
                    help="predict only these windows: one 'FS FE' line per chunk "
                         "(half-open source frames); overrides the presence gate")
    ap.add_argument("--allow-sealed", action="store_true",
                    help="predict an animal listed in the config's sealed_animals "
                         "(held-out test animals: refused by default)")
    ap.add_argument("--calib-date", default=None,
                    help="use this YYYY_MM_DD calibration instead of the "
                         "session date / nearest date")
    args = ap.parse_args()

    from tailcycle_run.config import sealed_animals
    if args.animal in sealed_animals() and not args.allow_sealed:
        raise SystemExit(f"[tailcycle-cluster] {args.animal} is a SEALED animal "
                         f"(config sealed_animals); refusing. Pass --allow-sealed "
                         f"only for the final held-out test.")
    sub = submit(args.animal, args.session,
                  num_chunks=args.num_chunks,
                  queue=args.queue, project=args.project, wall=args.wall,
                  slots=args.slots,
                  video_root=args.video_root, calib_root=args.calib_root,
                  pred_root=args.pred_root,
                  variant=args.variant,
                  presence_gate=not args.no_presence_gate,
                  calib_date=args.calib_date,
                  ranges_file=args.ranges_file)
    print(f"[tailcycle-cluster] submitted: {sub}", flush=True)
    wait_and_fetch(args.animal, args.session, sub, args.pred_root)


if __name__ == "__main__":
    sys.exit(main() or 0)
