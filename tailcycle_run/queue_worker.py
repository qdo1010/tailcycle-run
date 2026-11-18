"""Concurrent worker for the tailcycle-predict queue.

Runs up to MAX_CONCURRENT tailcycle cluster drivers at a time. When a slot
frees, pops the next job from `tailcycle_queue.json` and spawns it. Exits
when queue is empty AND no drivers are still alive.

Job entry format (same shape as the scorer/posetail queue):
    {"cmd": [...],  "log_path": "...", "animal": "...", "session": "...",
     "queued_at": "YYYY-MM-DD HH:MM:SS"}
"""
from __future__ import annotations
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailcycle_run.config import path as _cfgp  # noqa: E402

QUEUE_DIR       = _cfgp("local.log_dir", "/tmp/tailcycle_run")
QUEUE           = QUEUE_DIR / "tailcycle_queue.json"
LOCK            = QUEUE_DIR / "tailcycle_queue.lock"
WORKER_PID      = QUEUE_DIR / "tailcycle_queue_worker.pid"
CURRENT         = QUEUE_DIR / "tailcycle_queue_current.json"

MAX_CONCURRENT  = 3         # tuned by user — 3 concurrent tailcycle drivers
POLL_SECONDS    = 5


def _read(path: Path, default):
    if not path.exists(): return default
    try: return json.loads(path.read_text() or json.dumps(default))
    except Exception: return default


def _pop_next() -> dict | None:
    QUEUE_DIR.mkdir(exist_ok=True)
    with open(LOCK, "w") as fl:
        fcntl.flock(fl, fcntl.LOCK_EX)
        items = _read(QUEUE, [])
        if not items: return None
        job, rest = items[0], items[1:]
        QUEUE.write_text(json.dumps(rest, indent=2))
        return job


def _drivers_alive() -> int:
    """Count cluster_run.py processes for tailcycle."""
    try:
        r = subprocess.run(["pgrep", "-f", "cluster/cluster_run.py"],
                           capture_output=True, text=True)
        return len([l for l in r.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


def _publish_running(running: list[dict]) -> None:
    """List of currently-running (kind + session + started_at + pid) — one
    entry per active driver. Consumed by a status panel (e.g. a dashboard)."""
    CURRENT.write_text(json.dumps(running, indent=2))


def _extract_session(cmd: list[str]) -> tuple[str | None, str | None]:
    def _get(flag):
        try: return cmd[cmd.index(flag) + 1]
        except Exception: return None
    return _get("--animal"), _get("--session")


WORKER_LOCK     = QUEUE_DIR / "tailcycle_queue_worker.lock"
_worker_lock_fh = None       # held for the process lifetime; released on exit


def main():
    # Single-instance guard via an exclusive, non-blocking flock. The old
    # check-pidfile-then-write had a race: N rapid submits each saw "no
    # worker" before the first one wrote its pid, so 5 workers ran at once
    # and the 3-driver cap was blown (7 drivers, 2026-09-29). flock is atomic
    # and the kernel drops it if the worker dies, so no stale-lock cleanup.
    global _worker_lock_fh
    QUEUE_DIR.mkdir(exist_ok=True)
    _worker_lock_fh = open(WORKER_LOCK, "w")
    try:
        fcntl.flock(_worker_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[tailcycle-queue] another worker holds the lock; exit")
        return 0
    WORKER_PID.write_text(str(os.getpid()))

    active: list[dict] = []     # currently-spawned jobs I own
    try:
        while True:
            # Reap children I spawned that have exited. Using waitpid(WNOHANG)
            # -- os.kill(pid, 0) alone returns True for zombies (pid entry
            # still exists but process is defunct) which stalls the queue with
            # a "running" driver that isn't actually doing anything.
            import errno
            still_alive = []
            for j in active:
                try:
                    reaped_pid, _status = os.waitpid(j["pid"], os.WNOHANG)
                    if reaped_pid == 0:
                        still_alive.append(j)
                    else:
                        print(f"[tailcycle-queue] {j['animal']}/{j['session']} exited")
                except OSError as e:
                    if e.errno == errno.ECHILD:
                        # child was already reaped elsewhere -- treat as exited
                        print(f"[tailcycle-queue] {j['animal']}/{j['session']} "
                              f"already reaped")
                    else:
                        # unknown error -- log and drop so we don't stall
                        print(f"[tailcycle-queue] waitpid on "
                              f"{j['animal']}/{j['session']}: {e}")
            active = still_alive
            _publish_running(active)

            # Also count total drivers (mine + any orphans the user spawned
            # directly via the button without going through the queue — we
            # want the concurrency cap to be a global one).
            alive = _drivers_alive()
            slots = max(0, MAX_CONCURRENT - alive)

            if slots > 0:
                popped = 0
                while popped < slots:
                    job = _pop_next()
                    if job is None: break
                    animal, session = _extract_session(job["cmd"])
                    log = open(job["log_path"], "ab")
                    p = subprocess.Popen(job["cmd"], stdout=log,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True)
                    active.append({"pid": p.pid, "animal": animal,
                                    "session": session,
                                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "log_path": job["log_path"]})
                    print(f"[tailcycle-queue] started {animal}/{session} pid={p.pid} "
                          f"(active: {len(active)}/{MAX_CONCURRENT})")
                    popped += 1

            # Empty queue + no drivers -> exit cleanly.
            if not active and not _read(QUEUE, []):
                _publish_running([])
                print("[tailcycle-queue] queue drained, no active drivers, exit")
                return 0

            time.sleep(POLL_SECONDS)
    finally:
        try: WORKER_PID.unlink()
        except FileNotFoundError: pass


if __name__ == "__main__":
    sys.exit(main() or 0)
