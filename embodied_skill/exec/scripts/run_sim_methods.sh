#!/usr/bin/env bash
# 隔离多趟仿真（追加模式）：对传入的每个方法各起一趟独立 controller pass，
# --no-gold --merge 累加进已有 sim_results.json（保留 gold 及其它方法结果）。
# 用法：  bash run_sim_methods.sh M1-Head M1-Head+L4
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
