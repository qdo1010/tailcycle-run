#!/usr/bin/env python3
"""Split a recording into frame chunks, and merge the per-chunk predictions.

    python chunks.py split --videos REC_DIR --chunks K --out OUT_DIR [--glob '*.mp4']
        Writes OUT_DIR/chunks.txt, one "FS FE" line per chunk (half-open source
        frames), covering [0, N) where N is the SHORTEST camera's frame count.
        Refuses MPEG-TS files (see README: rewrap them to MP4 first).
        --smoke F instead writes ONE chunk of F frames from the middle of the
        recording (a smoke test; use its own OUT_DIR and never merge it).

    python chunks.py merge OUT_DIR [--table points3d] [--csv] [--no-complete-check]
        Checks that every chunk_NNN/ finished (session.toml: complete = true),
        concatenates their <table>.pq into OUT_DIR/<table>.pq (and .csv), and
        adds a `chunk` column. Exits 1 and lists the chunks to re-run otherwise.

Needs Python >= 3.11 and pyarrow (the tailcyclenet pixi env has both) and
ffprobe on the PATH for `split`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path


def _probe(video: Path) -> tuple[str, int | None]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames:format=format_name", "-of", "json", str(video)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    fmt = j.get("format", {}).get("format_name", "")
    nb = (j.get("streams") or [{}])[0].get("nb_frames")
    return fmt, (int(nb) if nb not in (None, "N/A") else None)


def cmd_split(a: argparse.Namespace) -> None:
    vids = sorted(Path(a.videos).glob(a.glob))
    if not vids:
        sys.exit(f"no {a.glob} in {a.videos}")
    counts = {}
    for v in vids:
        fmt, n = _probe(v)
        if "mpegts" in fmt:
            sys.exit(f"{v.name} is MPEG-TS (whatever its extension). Exact frame seeking does not "
                     f"work on TS; rewrap it to MP4 first (README, 'Before you start').")
        if n is None:
            sys.exit(f"{v.name}: the container has no frame count; rewrap it to MP4 first")
        counts[v.name] = n
    n = min(counts.values())
    spread = max(counts.values()) - n
    print(f"{len(vids)} cameras, shortest {n:,} frames"
          + (f" (cameras differ by up to {spread} frames)" if spread else ""))
    if a.smoke:
        fs = max(0, n // 2 - a.smoke // 2)
        lines = [f"{fs} {min(n, fs + a.smoke)}"]
        k, w = 1, min(n, fs + a.smoke) - fs
    else:
        if not a.chunks:
            sys.exit("give --chunks K (or --smoke FRAMES)")
        k = min(a.chunks, n)
        w = n // k
        lines = [f"{(i * w)} {(n if i == k - 1 else (i + 1) * w)}" for i in range(k)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chunks.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'chunks.txt'}: {k} chunk(s) of about {w:,} frames")


def _complete(chunk: Path) -> bool:
    st = chunk / "session.toml"
    if not st.exists():
        return False
    try:
        return bool(tomllib.loads(st.read_text()).get("complete", False))
    except tomllib.TOMLDecodeError:
        return False


def cmd_merge(a: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    out = Path(a.out)
    k = sum(1 for line in (out / "chunks.txt").read_text().splitlines() if line.strip())
    missing, tables = [], []
    for i in range(1, k + 1):
        c = out / f"chunk_{i:03d}"
        if not (_complete(c) if a.complete_check else (c / "session.toml").exists()):
            missing.append(i)
            continue
        f = c / f"{a.table}.pq"
        if not f.exists():          # finished, but no detections in this chunk
            print(f"note: chunk {i} finished with no {a.table} rows")
            continue
        t = pq.read_table(f)
        # dictionary columns differ per chunk; decode them so the concat lines up
        t = pa.table({n: (col.cast(col.type.value_type)
                          if pa.types.is_dictionary(col.type) else col)
                      for n, col in zip(t.column_names, t.columns)})
        tables.append(t.append_column("chunk", pa.array([i] * t.num_rows, pa.int32())))
    if missing:
        sys.exit(f"{len(missing)} of {k} chunks not complete; re-run: {' '.join(map(str, missing))}")
    if not tables:
        sys.exit("every chunk finished with no rows")
    t = pa.concat_tables(tables, promote_options="permissive")
    if "frame" in t.column_names:
        t = t.take(pc.sort_indices(t, sort_keys=[("frame", "ascending")]))
    dst = out / f"{a.table}.pq"
    pq.write_table(t, dst)
    print(f"wrote {dst}: {t.num_rows:,} rows from {len(tables)} chunks")
    if a.csv:
        pcsv.write_csv(t, dst.with_suffix(".csv"))
        print(f"wrote {dst.with_suffix('.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("split")
    w.add_argument("--videos", required=True)
    w.add_argument("--chunks", type=int)
    w.add_argument("--smoke", type=int, metavar="FRAMES")
    w.add_argument("--out", required=True)
    w.add_argument("--glob", default="*.mp4")
    m = sub.add_parser("merge")
    m.add_argument("out")
    m.add_argument("--table", default="points3d")
    m.add_argument("--csv", action="store_true")
    m.add_argument("--no-complete-check", dest="complete_check", action="store_false",
                   help="for tailcyclenet checkouts too old to write `complete` in "
                        "session.toml: accept any chunk that has one (a crashed chunk's "
                        "partial output is then NOT caught)")
    a = ap.parse_args()
    {"split": cmd_split, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
