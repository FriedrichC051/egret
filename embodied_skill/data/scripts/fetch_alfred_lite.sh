#!/usr/bin/env bash
# Download ALFRED Lite (json_2.1.0.7z, 35MB, trajectory JSON only) to the WSL native disk and extract it.
# Extract with py7zr (pip, no sudo) to avoid installing p7zip which needs sudo.
# Usage: bash embodied_skill/data/scripts/fetch_alfred_lite.sh
# For a proxy: export HTTPS_PROXY=http://host:port
set -e

PROXY="${HTTPS_PROXY:-}"   # optional; leave unset for direct download
URL="https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
DEST="$HOME/alfred_data"
ARC="$DEST/json_2.1.0.7z"
PY="${PYTHON:-python}"
PIP="${PIP:-pip}"

mkdir -p "$DEST"

echo "== ensure py7zr is installed =="
$PIP install -q py7zr 2>&1 | tail -n 1 || true

echo "== download json_2.1.0.7z (35MB, via proxy, resumable) =="
curl ${PROXY:+-x "$PROXY"} -L -C - -o "$ARC" \
  --retry 5 --retry-delay 3 \
  -w "done HTTP=%{http_code} avg=%{speed_download}B/s size=%{size_download}\n" \
  "$URL"

echo "== extract to $DEST/json_2.1.0/ =="
$PY - "$ARC" "$DEST" <<'PY'
import sys, py7zr, os, time
arc, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
with py7zr.SevenZipFile(arc, mode="r") as z:
    z.extractall(path=dest)
print("extracted in %.1fs" % (time.time() - t0))
# quick count
import glob
n = len(glob.glob(os.path.join(dest, "json_2.1.0", "*", "*", "*", "traj_data.json")))
print("traj_data.json total:", n)
PY

echo "== clean up the archive to free space =="
rm -f "$ARC"
echo "ALFRED_LITE_DONE"
