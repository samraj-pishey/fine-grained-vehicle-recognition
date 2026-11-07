#!/usr/bin/env bash
# Acquire the ORIGINAL Stanford Cars release files.
#
# The official host (ai.stanford.edu/~jkrause/...) has been offline since ~2023
# (see torchvision docs / pytorch/vision#7545). We fetch the Internet Archive
# raw-byte captures ("id_" URLs) of the original Stanford files and record
# provenance + SHA-256 in <DATA_ROOT>/manifests/acquisition_manifest.tsv.
#
# Usage: scripts/acquire_dataset.sh <DATA_ROOT>
# Resumable: re-running continues partial downloads (curl -C -).
set -euo pipefail

DATA_ROOT="${1:?usage: acquire_dataset.sh <DATA_ROOT>}"
ARCH="$DATA_ROOT/raw_archives"
MAN="$DATA_ROOT/manifests/acquisition_manifest.tsv"
mkdir -p "$ARCH" "$DATA_ROOT/manifests"

MIN_FREE_GB=8   # safety margin: refuse to start if less than this is free
free_gb=$(df -g "$DATA_ROOT" | awk 'NR==2{print $4}')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "ABORT: only ${free_gb} GB free (< ${MIN_FREE_GB} GB safety margin)" >&2; exit 2
fi

WB="https://web.archive.org/web/20221212052731id_"
# name|original official URL
FILES=(
  "car_devkit.tgz|https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
  "cars_test_annos_withlabels.mat|http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"
  "cars_train.tgz|http://ai.stanford.edu/~jkrause/car196/cars_train.tgz"
  "cars_test.tgz|http://ai.stanford.edu/~jkrause/car196/cars_test.tgz"
)

[ -f "$MAN" ] || printf "file\toriginal_url\tretrieval_url\tretrieved_utc\tbytes\tsha256\n" > "$MAN"

fetch() {  # name url -> downloads to $ARCH/name.part (resumable)
  local name="$1" url="$2" out="$ARCH/$1"
  [ -f "$out" ] && return 0
  for attempt in 1 2 3 4 5 6 7 8; do
    if curl -sSfL --retry 5 --retry-delay 10 --connect-timeout 30 -C - -o "$out.part" "$url"; then return 0; fi
    echo "  retry $attempt for $name"; sleep 20
  done
  return 1
}

# Phase 1: parallel download of all not-yet-recorded files (each writes its own .part).
pids=()
for entry in "${FILES[@]}"; do
  name="${entry%%|*}"; orig="${entry#*|}"
  grep -q "^${name}	" "$MAN" && continue
  echo "[get ] $name  <- $WB/$orig"
  fetch "$name" "$WB/$orig" & pids+=($!)
done
for p in "${pids[@]+"${pids[@]}"}"; do wait "$p"; done

# Phase 2: sequential verification + provenance record.
for entry in "${FILES[@]}"; do
  name="${entry%%|*}"; orig="${entry#*|}"; url="$WB/$orig"
  out="$ARCH/$name"
  if grep -q "^${name}	" "$MAN"; then echo "[skip] $name already recorded"; continue; fi
  [ -f "$out" ] || mv "$out.part" "$out"
  # gzip/mat sanity before accepting
  case "$name" in
    *.tgz) gzip -t "$out" ;;
    *.mat) head -c 10 "$out" | grep -q "MATLAB" ;;
  esac
  chmod a-w "$out"   # raw archives are immutable
  sha=$(shasum -a 256 "$out" | awk '{print $1}')
  bytes=$(stat -f%z "$out")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$orig" "$url" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$bytes" "$sha" >> "$MAN"
  echo "[ ok ] $name $bytes bytes sha256=$sha"
done
echo "DONE"; cat "$MAN"
