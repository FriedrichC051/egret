#!/usr/bin/env bash
# 通过 Windows 本地代理下载 AI2-THOR Linux64(OpenGL) 构建，解压到 ~/.ai2thor/releases/<BUILD>/。
set -e

PROXY="${HTTPS_PROXY:-}"   # optional; leave unset for direct download
COMMIT="f0825767cd50d69f666c7f282e54abfe58f1e917"
BUILD="thor-Linux64-${COMMIT}"
URL="http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/${BUILD}.zip"
REL="$HOME/.ai2thor/releases"
DEST="$REL/$BUILD"
ZIP="/tmp/${BUILD}.zip"

mkdir -p "$REL"

echo "== 下载（走代理，断点续传） =="
curl ${PROXY:+-x "$PROXY"} -L -C - -o "$ZIP" \
  --retry 5 --retry-delay 3 \
  -w "done HTTP=%{http_code} avg=%{speed_download}B/s size=%{size_download}\n" \
  "$URL"

echo "== 清理残留并解压到 $DEST =="
rm -rf "$DEST"
mkdir -p "$DEST"
"${PYTHON:-python}" - "$ZIP" "$DEST" <<PY
import sys, zipfile
zip_path, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zip_path) as z:
    z.extractall(dest)
    print("extracted", len(z.namelist()), "entries")
PY

echo "== 赋可执行权限 =="
BIN="$DEST/$BUILD"
chmod +x "$BIN" 2>/dev/null || true
find "$DEST" -type f -name "*.so" -exec chmod +x {} \; 2>/dev/null || true
ls -la "$DEST" | head -n 8
echo "OK executable: $BIN"

rm -f "$ZIP"
echo "LINUX64_FETCH_DONE"
