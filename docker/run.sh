#!/usr/bin/env bash
# Usage:
#   docker/run.sh                       # interactive shell in the container
#   docker/run.sh python scripts/track_video.py --video assets/apple.mp4
#
# Env vars:
#   GPU=1          GPUs to expose: one id, a list like 0,1,2,3, or all (default: 0)
#   DATA=/path     extra host dir with your videos, mounted at /data (optional)
#   PORT=7860      publish this container port on the host's 127.0.0.1 (optional)
#   MOUNTS="/a /b" host dirs mounted read-only at the same path inside (optional)
set -euo pipefail

IMAGE=cotracker3:latest
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-0}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$REPO/docker/Dockerfile" "$REPO/docker"
fi

EXTRA=()
if [[ -n "${DATA:-}" ]]; then
    EXTRA+=(-v "$(realpath "$DATA")":/data)
fi

if [[ "$GPU" == all ]]; then
    GPU_ARG=all
else
    GPU_ARG="\"device=$GPU\""  # docker needs the inner quotes for comma lists
fi

for m in ${MOUNTS:-}; do
    [[ -d "$m" ]] && EXTRA+=(-v "$(realpath "$m")":"$(realpath "$m")":ro)
done
EXTRA+=(-e HOST_MOUNTS="${MOUNTS:-}")

if [[ -n "${PORT:-}" ]]; then
    EXTRA+=(-p "127.0.0.1:$PORT:$PORT")
fi

TTY=()
[[ -t 0 ]] && TTY=(-it)

exec docker run --rm "${TTY[@]}" \
    --gpus "$GPU_ARG" \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$REPO":/workspace \
    "${EXTRA[@]}" \
    "$IMAGE" "${@:-bash}"
