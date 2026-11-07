#!/usr/bin/env bash
# Complete a partially downloaded file using N parallel HTTP range requests (resumable per chunk).
# usage: parallel_range_fetch.sh <url> <partfile> <final_out> [N]
set -euo pipefail
url="$1"; part="$2"; final="$3"; N="${4:-6}"
total=$(curl -sSIL --max-time 60 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
have=$( [ -f "$part" ] && stat -f%z "$part" || echo 0 )
[ -n "$total" ] || { echo "no content-length"; exit 2; }
echo "total=$total have=$have"
rem=$(( total - have )); chunk=$(( (rem + N - 1) / N ))
pids=()
for i in $(seq 0 $((N-1))); do
  s=$(( have + i*chunk )); e=$(( s + chunk - 1 )); [ $e -ge $total ] && e=$(( total - 1 ))
  [ $s -gt $e ] && continue
  ( f="$part.c$i"; want=$(( e - s + 1 ))
    for t in $(seq 1 20); do
      got=$( [ -f "$f" ] && stat -f%z "$f" || echo 0 )
      [ "$got" -ge "$want" ] && break
      curl -sSfL --connect-timeout 30 -r "$(( s + got ))-$e" "$url" >> "$f" || sleep 10
    done ) & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for i in $(seq 0 $((N-1))); do [ -f "$part.c$i" ] && cat "$part.c$i" >> "$part" && rm "$part.c$i"; done
size=$(stat -f%z "$part")
[ "$size" -eq "$total" ] || { echo "SIZE MISMATCH $size != $total"; exit 3; }
mv "$part" "$final"; echo "complete $final $size"
