#!/usr/bin/env bash
# 检查 WSLg + OpenGL(d3d12) GPU 路径是否具备条件（用于跑 AI2-THOR Linux64/OpenGL 构建）。
echo "== 1) WSLg 显示是否存在 =="
echo "DISPLAY=$DISPLAY   WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
ls -d /mnt/wslg 2>/dev/null && ls /mnt/wslg/.X11-unix 2>/dev/null || echo "无 /mnt/wslg"

echo
echo "== 2) d3d12 OpenGL 驱动是否就位 =="
ls /usr/lib/x86_64-linux-gnu/dri/ 2>/dev/null | grep -iE "d3d12|zink" || echo "无 d3d12/zink dri（可能需要装 mesa）"

echo
echo "== 3) WSL GPU 库 =="
ls /usr/lib/wsl/lib 2>/dev/null | grep -iE "d3d12|dxcore" || echo "无 wsl d3d12 库"

echo
echo "== 4) glxinfo 是否可用（看真实 OpenGL renderer） =="
if which glxinfo >/dev/null 2>&1; then
  glxinfo -B 2>/dev/null | grep -iE "OpenGL renderer|OpenGL version|Device" || glxinfo 2>/dev/null | grep -iE "renderer|vendor" | head
else
  echo "无 glxinfo（需 sudo apt install mesa-utils 才能确认 renderer）"
fi
echo
echo "CHECK_DONE"
