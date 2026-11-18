#!/usr/bin/env python3
"""Run tailcycle cluster predictions for a list of sessions ONE AT A TIME.

For very large sessions (orma: 0.2-1 TB of rig video each) the dashboard
queue's 3-way concurrency can stage several TB on the cluster filesystem at once.
This runs cluster_run.py synchronously per session: it stages, submits,
waits, pulls data3D.csv and reclaims the staged videos before the next one
starts. Before each upload it waits until the cluster filesystem has
room for that session's videos plus FLOOR_GB. Stops at the first failure.

Mouse-presence gate: when <session>/presence.json exists (post/yolo_presence.py,
the top-down YOLO track), a session whose mouse is in the arena for less than
--min-present-s is skipped -- tailcycle there would only fit false poses. After
each run the YOLO-gated pose (<session>/yolo_clean/, tools/yolo_clean.py) is
built from the new tailcycle output (post/yolo_clean.py).

    run_sequential.py --tab orma --animal taco --sessions s1 s2 ...

Per-session log at <local.log_dir>/tailcycle_<animal>_<session>.log; overall
log printed to stdout. Tabs are the [sources.<tab>] tables of the config.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailcycle_run.config import get as _cfg, path as _cfgp, sources  # noqa: E402
from cluster_run import VARIANTS, LOGIN, CLUSTER_DF_PATH  # noqa: E402

SOURCES   = sources()
PY        = _cfg("local.python", sys.executable)
DRIVER    = Path(__file__).with_name("cluster_run.py")
YOLO_CLEAN = Path(__file__).resolve().parents[1] / "post" / "yolo_clean.py"
LOG_DIR   = _cfgp("local.log_dir", "/tmp/tailcycle_run")
FLOOR_GB  = 500
RECHECK_S = 1800


def _say(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def _session_gb(vdir: Path) -> float:
    return sum(p.stat().st_size for p in vdir.glob("Cam*.mp4")) / 1e9


def _free_gb() -> float:
    """Free space on the cluster filesystem via one ssh df (no dependence on
    the CIFS mount; called once per session / per 30-min recheck)."""
    out = subprocess.run(["ssh", LOGIN,
                          f"df -B1 --output=avail {CLUSTER_DF_PATH} | tail -1"],
                         capture_output=True, text=True, timeout=60).stdout
    return int(out.strip()) / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True, choices=list(SOURCES))
    ap.add_argument("--animal", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--variant", default="univ24")
    ap.add_argument("--min-present-s", type=float, default=30.0,
                    help="skip sessions whose presence.json has less mouse-in-arena time")
    ap.add_argument("--allow-ungated", action="store_true",
                    help="run sessions that have no presence.json (whole recording)")
    ap.add_argument("--no-presence-gate", action="store_true",
                    help="run whole recordings (passed to cluster_run); e.g. to "
                         "fetch a run submitted before the gate existed")
    ap.add_argument("--allow-sealed", action="store_true",
                    help="pass --allow-sealed to cluster_run (sealed test animals)")
    ap.add_argument("--keep-going", action="store_true",
                    help="log a failed session and continue with the next")
    ap.add_argument("--max-consecutive-fail", type=int, default=3,
                    help="with --keep-going, still stop after this many failures in a row "
                         "(a systematic error, e.g. a broken harness, fails every session)")
    a = ap.parse_args()
    out_name = VARIANTS[a.variant]["out_name"]
    failed = []
    streak = 0
    src = SOURCES[a.tab]
    for sess in a.sessions:     # the gate decisions up front, for the log
        pres = src["pred_root"] / a.animal / sess / "presence.json"
        _say(f"  {sess}: " + (f"mouse in arena {json.loads(pres.read_text())['present_s']:.0f} s"
                              if pres.exists() else "no presence.json"))

    for i, sess in enumerate(a.sessions, 1):
        tag = f"[{i}/{len(a.sessions)}] {a.animal}/{sess}"
        out = src["pred_root"] / a.animal / sess / out_name
        # a windows-only prediction (REAR-DATA clip batch) is not a full one
        if (out / "data3D.csv").exists() and "windows_only" not in (
                (out / "info.yaml").read_text() if (out / "info.yaml").exists() else ""):
            _say(f"{tag} already has {out_name}/data3D.csv, skipping")
            continue
        pres = src["pred_root"] / a.animal / sess / "presence.json"
        if a.no_presence_gate:
            pass
        elif pres.exists():
            present_s = json.loads(pres.read_text())["present_s"]
            if present_s < a.min_present_s:
                _say(f"{tag} mouse in arena only {present_s:.0f} s "
                     f"(< {a.min_present_s:.0f} s), skipping")
                continue
        elif not a.allow_ungated:
            _say(f"{tag} no presence.json (run tools/yolo_presence.py first, or "
                 f"pass --allow-ungated), skipping")
            continue
        else:
            _say(f"{tag} WARNING: no presence.json -- running ungated")
        vdir = src["video_root"] / a.animal / sess
        need = _session_gb(vdir) + FLOOR_GB
        while True:
            free = _free_gb()
            if free >= need:
                break
            _say(f"{tag} waiting: {free:.0f} GB free on cluster, "
                 f"need {need:.0f} GB; rechecking in {RECHECK_S // 60} min")
            time.sleep(RECHECK_S)
        log = LOG_DIR / (f"tailcycle_{a.animal}_{sess}.log" if out_name == "tailcycle"
                         else f"tailcycle_{a.animal}_{sess}_{a.variant}.log")
        _say(f"{tag} starting ({need - FLOOR_GB:.0f} GB, {free:.0f} GB free) "
             f"log={log}")
        with open(log, "w") as f:
            f.write(f"[run_sequential] tailcycle predict {a.animal}/{sess} "
                    f"tab={a.tab} variant={a.variant}\n")
            f.flush()
            rc = subprocess.run(
                [PY, "-u", str(DRIVER), "--animal", a.animal, "--session", sess,
                 "--video-root", str(src["video_root"]),
                 "--pred-root", str(src["pred_root"]),
                 "--variant", a.variant]
                + (["--no-presence-gate"] if a.no_presence_gate else [])
                + (["--allow-sealed"] if a.allow_sealed else []),
                stdout=f, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            if not a.keep_going:
                sys.exit(f"{tag} FAILED rc={rc}; see {log}. Stopping the batch.")
            _say(f"{tag} FAILED rc={rc}; see {log}. Continuing.")
            failed.append(sess)
            streak += 1
            if streak >= a.max_consecutive_fail:
                sys.exit(f"{streak} sessions failed in a row (last: {log}); stopping -- "
                         f"looks systematic. FAILED: {failed}")
            continue
        streak = 0
        if pres.exists() and out_name == "tailcycle":
            r = subprocess.run([PY, str(YOLO_CLEAN), "--pred-root", str(src["pred_root"]),
                                "--animal", a.animal, "--sessions", sess],
                               capture_output=True, text=True)
            _say(f"{tag} yolo_clean: " + (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])
        _say(f"{tag} done")
    _say("all sessions done" + (f"; FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
