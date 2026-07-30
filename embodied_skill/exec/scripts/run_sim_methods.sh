#!/usr/bin/env bash
# Isolated multi-pass simulation (append mode): for each method passed in, start one independent controller pass,
# with --no-gold --merge accumulating into the existing sim_results.json (keeping gold and other methods' results).
# Usage:  bash run_sim_methods.sh M1-Head M1-Head+L4
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export DISPLAY=:0
export GALLIUM_DRIVER=d3d12
PY="${PYTHON:-python}"
S=exec/code/run_sim_from_selections.py
SEL=eval/datasets/selections.json
OUT=eval/datasets/sim_results.json

for m in "$@"; do
  echo "==== PASS $m ===="
  "$PY" "$S" --selections "$SEL" --out "$OUT" --methods "$m" --no-gold --merge
done
echo ALLDONE
