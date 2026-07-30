#!/usr/bin/env bash
# 分批调用 synth_composite.py 累加复合技能到目标数量。
# 每批带 OS 级 timeout 兜底：某批卡死会被杀掉，循环继续下一批。
# 用法: bash run_synth_batches.sh <mode> <per_batch_n> <num_batches> [base_limit]
set -u
source ~/.embodied_skill_env 2>/dev/null || true
PY="${PYTHON:-python}"
EMB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$EMB/skills/code/synth_composite.py"

MODE="${1:-composite}"
N="${2:-8}"
BATCHES="${3:-4}"
BASE_LIMIT="${4:-8}"
BASE_OFFSET_START="${5:-0}"   # variant 模式：起始基准偏移，每批 +BASE_LIMIT
PER_BATCH_TIMEOUT=360   # 秒；单批超时即放弃该批

echo "== 分批合成: mode=$MODE n=$N batches=$BATCHES base_limit=$BASE_LIMIT offset_start=$BASE_OFFSET_START =="
for i in $(seq 1 "$BATCHES"); do
  echo "---- 批次 $i/$BATCHES ----"
  START=$(date +%s)
  if [ "$MODE" = "composite" ]; then
    timeout "$PER_BATCH_TIMEOUT" "$PY" "$SCRIPT" --mode composite --n "$N" --run
  else
    OFFSET=$(( BASE_OFFSET_START + (i - 1) * BASE_LIMIT ))
    echo "base_offset=$OFFSET"
    timeout "$PER_BATCH_TIMEOUT" "$PY" "$SCRIPT" --mode variant --base-limit "$BASE_LIMIT" --base-offset "$OFFSET" --run
  fi
  CODE=$?
  echo "批次 $i 退出码=$CODE 用时=$(( $(date +%s) - START ))s"
  if [ "$CODE" -eq 124 ]; then
    echo "[警告] 批次 $i 超时被杀，继续下一批"
  fi
done
echo "== 全部批次结束 =="
