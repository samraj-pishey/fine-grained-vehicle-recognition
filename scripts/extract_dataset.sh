#!/usr/bin/env bash
# Extract the verified raw archives into <DATA_ROOT>/raw_extracted (once), then make them read-only.
# Refuses to run if the projected extracted size plus a safety margin exceeds free disk.
set -euo pipefail
DATA_ROOT="${1:?usage: extract_dataset.sh <DATA_ROOT>}"
ARCH="$DATA_ROOT/raw_archives"; OUT="$DATA_ROOT/raw_extracted"
SAFETY_GB=5
mkdir -p "$OUT"
for name in cars_train cars_test; do
  tgz="$ARCH/$name.tgz"
  grep -q "^${name}.tgz	" "$DATA_ROOT/manifests/acquisition_manifest.tsv" || { echo "ABORT: $name.tgz not verified in manifest"; exit 2; }
  if [ -d "$OUT/$name" ]; then echo "[skip] $OUT/$name exists"; continue; fi
  # JPEGs barely compress, so the extracted size ~= archive size; use 1.1x as a conservative projection.
  need_gb=$(( $(stat -f%z "$tgz") * 11 / 10 / 1024 / 1024 / 1024 + 1 ))
  free_gb=$(df -g "$DATA_ROOT" | awk 'NR==2{print $4}')
  if [ $(( free_gb - need_gb )) -lt $SAFETY_GB ]; then
    echo "ABORT: free ${free_gb}GB - need ${need_gb}GB < safety ${SAFETY_GB}GB"; exit 3
  fi
  echo "[extract] $name (need ~${need_gb}GB, free ${free_gb}GB)"
  tar -xzf "$tgz" -C "$OUT"   # streaming extraction; archive is preserved
  chmod -R a-w "$OUT/$name"
done
du -sh "$OUT"/*
