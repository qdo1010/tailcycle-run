#!/usr/bin/env python3
"""GPU-h cap guard for one tailcycle LSF array (1 GPU per element).

Every --every-min minutes it makes ONE batched `bjobs -a` call over ssh to the login node for the
array (and the merge job). It sums the elements' run_time and kills the array
plus its merge job (bkill) if either:
  (a) measured GPU-h >= --frac x cap while elements are still pending/running, or
  (b) >= 20 % of the elements are DONE and the mean GPU-h per finished element
      x total elements > cap.
It exits when no element is pending or running, printing the final measured GPU-h.

    burn_guard.py --array 12345 [--merge 12346] --cap 28 [--every-min 15]
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tailcycle_run.config import get as _cfg  # noqa: E402


def bjobs(ids: list[str]) -> list[tuple[str, int]]:
    out = subprocess.run(
        ["ssh", _cfg("cluster.login"),
         "bjobs -a -noheader -o 'stat run_time delimiter=\"|\"' " + " ".join(ids)],
        capture_output=True, text=True, timeout=120).stdout
    rows = []
    for ln in out.splitlines():
        if "|" not in ln:
            continue
        st, rt = ln.split("|", 1)
        sec = int(rt.split()[0]) if rt.strip() and rt.split()[0].isdigit() else 0
        rows.append((st.strip(), sec))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", required=True)
    ap.add_argument("--merge", default=None)
    ap.add_argument("--cap", type=float, required=True, help="GPU-h")
    ap.add_argument("--frac", type=float, default=0.9)
    ap.add_argument("--every-min", type=float, default=15)
    a = ap.parse_args()
    kill = [a.array] + ([a.merge] if a.merge else [])
    while True:
        rows = bjobs([a.array])
        if not rows:
            print(f"[burn] {time.strftime('%F %T')} no rows from bjobs (yet)", flush=True)
            time.sleep(a.every_min * 60)
            continue
        n = len(rows)
        gpuh = sum(s for _, s in rows) / 3600
        done = [s for st, s in rows if st == "DONE"]
        live = sum(st in ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP") for st, _ in rows)
        proj = (sum(done) / 3600 / len(done) * n) if len(done) >= 0.2 * n and done else None
        print(f"[burn] {time.strftime('%F %T')} {n} elements, {len(done)} done, {live} live, "
              f"measured {gpuh:.2f} GPU-h" + (f", projected {proj:.1f}" if proj else "")
              + f" (cap {a.cap})", flush=True)
        if live and (gpuh >= a.frac * a.cap or (proj is not None and proj > a.cap)):
            print(f"[burn] STOP: killing {kill}", flush=True)
            subprocess.run(["ssh", _cfg("cluster.login"), "bkill " + " ".join(kill)], timeout=120)
            return
        if not live:
            print(f"[burn] FINAL measured {gpuh:.2f} GPU-h over {n} elements", flush=True)
            return
        time.sleep(a.every_min * 60)


if __name__ == "__main__":
    main()
