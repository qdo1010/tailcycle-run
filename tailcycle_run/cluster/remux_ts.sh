#!/bin/bash
# Rewrap MPEG-TS (named .mp4) -> real MP4, stream copy (no re-encode), in place.
# Usage (CPU LSF job, -n 4): remux_ts.sh <staged dir> [parallel=4]
# For the v2 frame-indexed reader, which cannot index MPEG-TS (2026-11-17).
# A file is replaced only if its TS packet count equals the MP4 header frame
# count. Writes REMUX_OK in the dir when every file is MP4.
set -u
D=${1:?usage: remux_ts.sh <staged dir> [parallel]}
P=${2:-4}
cd "$D" || exit 1

one() {
  f=$1
  fmt=$(ffprobe -v error -show_entries format=format_name -of csv=p=0 "$f" | head -1 | tr -d '[:space:]')
  if [ "$fmt" != "mpegts" ]; then echo "$f: already $fmt, skip"; return 0; fi
  ns=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets \
       -of csv=p=0 "$f" | head -1 | tr -dc 0-9)
  if ! ffmpeg -nostdin -v error -fflags +genpts -i "$f" -map 0:v:0 -c copy \
       -movflags +faststart -f mp4 "$f.remux.tmp"; then
    echo "$f: ffmpeg FAILED"; rm -f "$f.remux.tmp"; return 1
  fi
  nd=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames \
       -of csv=p=0 "$f.remux.tmp" | head -1 | tr -dc 0-9)
  if [ -n "$nd" ] && [ "$ns" = "$nd" ]; then
    mv -f "$f.remux.tmp" "$f"; echo "$f: ok packets $ns -> mp4 frames $nd"
  else
    echo "$f: MISMATCH packets '$ns' vs mp4 frames '$nd', kept TS"; rm -f "$f.remux.tmp"; return 1
  fi
}
export -f one

ls Cam*.mp4 | xargs -P "$P" -I{} bash -c 'one "$1"' _ {}
bad=0
for f in Cam*.mp4; do
  [ "$(ffprobe -v error -show_entries format=format_name -of csv=p=0 "$f" | head -1 | tr -d '[:space:]')" = "mpegts" ] && bad=1
done
if [ $bad = 0 ]; then touch REMUX_OK; echo REMUX_DONE; else echo REMUX_HAD_FAILURES; fi
