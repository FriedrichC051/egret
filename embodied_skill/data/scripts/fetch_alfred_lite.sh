#!/usr/bin/env bash
# 下载 ALFRED Lite (json_2.1.0.7z, 35MB, 仅轨迹JSON) 到 WSL 原生盘并解压。
# 用 py7zr(pip, 免sudo) 解压，避免装 p7zip 需要 sudo。
# 用法： bash embodied_skill/data/scripts/fetch_alfred_lite.sh
# 如需代理：export HTTPS_PROXY=http://host:port
set -e

PROXY="${HTTPS_PROXY:-}"   # optional; leave unset for direct download
URL="https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
DEST="$HOME/alfred_data"
ARC="$DEST/json_2.1.0.7z"
PY="${PYTHON:-python}"
PIP="${PIP:-pip}"

mkdir -p "$DEST"

echo "== 确保 py7zr 已安装 =="
$PIP install -q py7zr 2>&1 | tail -n 1 || true

echo "== 下载 json_2.1.0.7z（35MB，走代理，断点续传） =="
curl ${PROXY:+-x "$PROXY"} -L -C - -o "$ARC" \
  --retry 5 --retry-delay 3 \
  -w "done HTTP=%{http_code} avg=%{speed_download}B/s size=%{size_download}\n" \
  "$URL"

echo "== 解压到 $DEST/json_2.1.0/ =="
$PY - "$ARC" "$DEST" <<'PY'
import sys, py7zr, os, time
arc, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
with py7zr.SevenZipFile(arc, mode="r") as z:
    z.extractall(path=dest)
print("extracted in %.1fs" % (time.time() - t0))
# 快速统计
import glob
n = len(glob.glob(os.path.join(dest, "json_2.1.0", "*", "*", "*", "traj_data.json")))
print("traj_data.json 总数:", n)
PY

echo "== 清理压缩包释放空间 =="
rm -f "$ARC"
echo "ALFRED_LITE_DONE"
