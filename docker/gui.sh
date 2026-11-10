#!/usr/bin/env bash
# Start the CoTracker web GUI (gui/) at http://localhost:${PORT:-7860}
#   docker/gui.sh                          # all GPUs, port 7860
#   PORT=8000 DATA=/path/to/videos docker/gui.sh
#   MOUNTS="/data/zeyu /mnt/videos" docker/gui.sh   # folders the picker can browse
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PORT="${PORT:-7860}" GPU="${GPU:-all}"
# Host folders the video picker can open in place (read-only, same paths as on the host).
export MOUNTS="${MOUNTS:-/data/zeyu $HOME}"
# Cached rebuild: picks up Dockerfile changes (e.g. the web dependencies).
docker build -q -t cotracker3:latest -f "$REPO/docker/Dockerfile" "$REPO/docker" >/dev/null
exec "$REPO/docker/run.sh" python -m gui.server --host 0.0.0.0 --port "$PORT"
