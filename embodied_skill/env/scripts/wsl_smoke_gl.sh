#!/usr/bin/env bash
# 用 WSLg 的 GPU OpenGL(d3d12) 跑 AI2-THOR Linux64 冒烟测试。
export DISPLAY=:0
# 关键：d3d12 是 gallium 驱动，必须用 GALLIUM_DRIVER 指定，才能让 OpenGL 走 GPU
export GALLIUM_DRIVER=d3d12
export MESA_LOADER_DRIVER_OVERRIDE=d3d12
export LIBGL_ALWAYS_SOFTWARE=0

echo "== DISPLAY=$DISPLAY  GALLIUM_DRIVER=$GALLIUM_DRIVER =="
START=$(date +%s)
EMB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
timeout 240 "${PYTHON:-python}" "$EMB/env/code/smoke_thor_gl.py"
CODE=$?
END=$(date +%s)
echo "REAL_EXIT=$CODE elapsed=$((END-START))s"
if [ "$CODE" = "124" ]; then echo "RESULT=TIMEOUT(仍卡住/太慢)"; elif [ "$CODE" = "0" ]; then echo "RESULT=OK"; else echo "RESULT=ERROR"; fi
