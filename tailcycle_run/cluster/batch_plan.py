#!/usr/bin/env python3
"""Unattended 47-kp v2 (univ2_47) cluster batch over a plan file, one session at a time.

Plan: TSV with a header, columns  tab animal session ranges proj_gpuh
(tab = a [sources.<tab>] table of the config; ranges = a cluster_run
--ranges-file; proj_gpuh = frames/5000 + 0.045 x windows, see `plan`).

Per session:
  1. skip if <pred_root>/<a>/<s>/tailcycle47_v2/data3D.csv exists
  2. stop the batch if measured-so-far + this projection > --total-cap
  3. wait until the cluster has room for the session's videos + 500 GB
  4. cluster_run.py (stage at --bwlimit, rewrap MPEG-TS on the cluster, smoke,
     array with a %--max-concurrent limit, merge, fetch, reclaim) while a
     burn_guard.py on the array + merge job enforces a per-session cap of
     max(1.5 x projection, projection + 2) GPU-h, checking every 5 min
  5. yolo_clean (--source tailcycle47_v2 --no-trial-gate) and body_span on the
     new output (tailcycle47_v2 and yolo_clean47_v2)
Stops after --max-consecutive-fail failures in a row, or when a guard kills.
The weekly-budget override reason comes from GATE_OVERRIDE_BUDGET in the env.

Keep --max-concurrent x element time well inside the guard interval (5 min):
the guard can only stop elements that have not started yet.

    tailcycle-run run --plan plan.tsv --total-cap 740 [--bwlimit 50000 --max-concurrent 12]

Per-session logs at <local.log_dir>/tailcycle_<animal>_<session>_univ2_47.log.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailcycle_run.config import get as _cfg, path as _cfgp, sources, sealed_animals  # noqa: E402
from cluster_run import VARIANTS, LOGIN, CLUSTER_DF_PATH  # noqa: E402

PY = _cfg("local.python", sys.executable)
HERE = Path(__file__).resolve().parent
POST = HERE.parent / "post"
LOG_DIR = _cfgp("local.log_dir", "/tmp/tailcycle_run")
FLOOR_GB = 500
RECHECK_S = 1800
VARIANT = "univ2_47"
OUT = VARIANTS[VARIANT]["out_name"]          # tailcycle47_v2
CLEAN_OUT = "yolo_clean47_v2"                # what yolo_clean --source tailcycle47_v2 writes


def say(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def guard_cap(proj: float) -> float:
    """Per-session burn_guard cap: 50 % headroom, at least 2 GPU-h."""
    return max(1.5 * proj, proj + 2.0)


def free_gb() -> float:
    """Free space on the cluster filesystem via one ssh df (once per session /
    per 30-min recheck)."""
    out = subprocess.run(["ssh", LOGIN, f"df -B1 --output=avail {CLUSTER_DF_PATH} | tail -1"],
                         capture_output=True, text=True, timeout=60).stdout
    return int(out.strip()) / 1e9


def session_gb(vdir: Path) -> float:
    return sum(p.stat().st_size for p in vdir.glob("Cam*.mp4")) / 1e9


def run_session(row: dict, src: dict, a, log: Path) -> tuple[int, float | None, bool]:
    """-> (cluster_run rc, guard-measured GPU-h or None, guard killed?)"""
    proj = float(row["proj_gpuh"])
    cap = guard_cap(proj)
    env = dict(os.environ, TAILCYCLE_BWLIMIT=str(a.bwlimit),
               TAILCYCLE_MAX_CONCURRENT=str(a.max_concurrent))
    env.pop("TAILCYCLE_SKIP_VIDEO_STAGE", None)
    cmd = [PY, "-u", str(HERE / "cluster_run.py"), "--animal", row["animal"],
           "--session", row["session"], "--video-root", str(src["video_root"]),
           "--pred-root", str(src["pred_root"]), "--variant", VARIANT,
           "--ranges-file", row["ranges"]] + (["--allow-sealed"] if a.allow_sealed else [])
    guard: dict = {}

    def watch() -> None:
        # the array and merge ids appear in the driver's log after the smoke
        arr = merge = None
        while proc.poll() is None and not (arr and merge):
            txt = log.read_text(errors="replace") if log.exists() else ""
            m = re.search(r"array-bsub: [\s\S]{0,2000}?Job <(\d+)>", txt)
            n = re.search(r"merge-bsub: [\s\S]{0,2000}?Job <(\d+)>", txt)
            arr, merge = (m.group(1) if m else None), (n.group(1) if n else None)
            time.sleep(20)
        if not arr:
            return
        g = subprocess.run([PY, "-u", str(HERE / "burn_guard.py"), "--array", arr]
                           + (["--merge", merge] if merge else [])
                           + ["--cap", f"{cap:.2f}", "--every-min", "5"],
                           capture_output=True, text=True)
        with open(log, "a") as f:
            f.write(g.stdout + g.stderr)
        guard["killed"] = "STOP: killing" in g.stdout
        vals = re.findall(r"measured ([\d.]+) GPU-h", g.stdout)
        guard["gpuh"] = float(vals[-1]) if vals else None

    with open(log, "w") as f:
        f.write(f"[run] {row['animal']}/{row['session']} proj {proj:.1f} GPU-h, "
                f"guard cap {cap:.1f}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        t = threading.Thread(target=watch, daemon=True)
        t.start()
        rc = proc.wait()
    t.join(timeout=3600)
    return rc, guard.get("gpuh"), guard.get("killed", False)


def post(row: dict, src: dict) -> None:
    root = src["pred_root"]
    for cmd in ([PY, str(POST / "yolo_clean.py"), "--pred-root", str(root), "--animal",
                 row["animal"], "--sessions", row["session"], "--source", OUT, "--no-trial-gate"],
                [PY, str(POST / "body_span.py"), "--pred-root", str(root), "--source", OUT],
                [PY, str(POST / "body_span.py"), "--pred-root", str(root),
                 "--source", CLEAN_OUT]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:]])[-1:]
        say(f"   post {Path(cmd[1]).name}: rc {r.returncode} {tail}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="tailcycle-run run")
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--total-cap", type=float, required=True, help="GPU-h for the whole batch")
    ap.add_argument("--bwlimit", type=int, default=50000, help="KB/s staging pace")
    ap.add_argument("--max-concurrent", type=int, default=12)
    ap.add_argument("--max-consecutive-fail", type=int, default=2)
    ap.add_argument("--allow-sealed", action="store_true",
                    help="run plan rows whose animal is in the config's sealed_animals")
    a = ap.parse_args()
    srcs = sources()
    with open(a.plan) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    bad = sorted({r["tab"] for r in rows} - set(srcs))
    if bad:
        sys.exit(f"plan {a.plan}: tab(s) {bad} not in the config's [sources.*] ({sorted(srcs)})")
    if not os.environ.get("GATE_OVERRIDE_BUDGET"):
        say("note: GATE_OVERRIDE_BUDGET not set; the spend gate may refuse jobs")
    sealed = sealed_animals()
    say(f"plan {a.plan}: {len(rows)} sessions, projected "
        f"{sum(float(r['proj_gpuh']) for r in rows):.1f} GPU-h, total cap {a.total_cap}")
    spent, streak, failed = 0.0, 0, []
    for i, row in enumerate(rows, 1):
        src = srcs[row["tab"]]
        tag = f"[{i}/{len(rows)}] {row['tab']} {row['animal']}/{row['session']}"
        if row["animal"] in sealed and not a.allow_sealed:
            say(f"{tag} SEALED animal (config sealed_animals), skipping")
            continue
        if (src["pred_root"] / row["animal"] / row["session"] / OUT / "data3D.csv").exists():
            say(f"{tag} already has {OUT}, skipping")
            continue
        proj = float(row["proj_gpuh"])
        if spent + proj > a.total_cap:
            say(f"{tag} STOP: spent {spent:.1f} + projected {proj:.1f} > total cap {a.total_cap}")
            break
        need = session_gb(src["video_root"] / row["animal"] / row["session"]) + FLOOR_GB
        while (free := free_gb()) < need:
            say(f"{tag} waiting: {free:.0f} GB free on cluster, need {need:.0f}; "
                f"recheck in {RECHECK_S // 60} min")
            time.sleep(RECHECK_S)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = LOG_DIR / f"tailcycle_{row['animal']}_{row['session']}_{VARIANT}.log"
        say(f"{tag} start: proj {proj:.1f} GPU-h, {need - FLOOR_GB:.0f} GB, log {log}")
        rc, gpuh, killed = run_session(row, src, a, log)
        spent += gpuh if gpuh is not None else (proj if rc == 0 else 0.0)
        say(f"{tag} rc {rc}, measured {gpuh} GPU-h, batch total {spent:.1f}")
        if killed:
            say(f"{tag} STOP: the burn guard killed this session's jobs")
            break
        if rc != 0:
            failed.append(f"{row['animal']}/{row['session']}")
            streak += 1
            if streak >= a.max_consecutive_fail:
                say(f"STOP: {streak} failures in a row; FAILED {failed}")
                break
            continue
        streak = 0
        post(row, src)
    say(f"BATCH END: total {spent:.1f} GPU-h" + (f"; FAILED {failed}" if failed else ""))


if __name__ == "__main__":
    main()
