#!/usr/bin/env bash
# ssh client and a user for the owner of /workspace, to run the VLM question loop (loop.py) inside a container.
# Run it inside the container, as root. The loop uses ssh to reach the machine with the videos. The new user has the
# uid and gid of the owner of /workspace, and its home is /workspace. Files it writes keep that owner. ssh reads its
# key from /workspace/.ssh; this script makes no key. It stops when another user already has that uid.
# Tested: 2026-11-06, Ubuntu 24.04, openssh-client 1:9.6p1-3ubuntu13.20
# Usage: bash scripts/vlm_loop/setup_ssh.sh USER_NAME      (idempotent)
set -euo pipefail
NAME=${1:?usage: bash scripts/vlm_loop/setup_ssh.sh USER_NAME}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
read -r OWNER_UID OWNER_GID < <(stat -c '%u %g' /workspace)

if command -v ssh >/dev/null; then
  echo "ssh already installed"
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends openssh-client
  rm -rf /var/lib/apt/lists/*
fi

# group and user with the owner ids of /workspace; nothing is made in /workspace (idempotent)
getent group "$OWNER_GID" >/dev/null || groupadd -g "$OWNER_GID" "$NAME"
getent passwd "$OWNER_UID" >/dev/null || useradd -u "$OWNER_UID" -g "$OWNER_GID" -d /workspace -M -s /bin/bash "$NAME"
IFS=: read -r USER_NAME _ _ _ _ USER_HOME _ < <(getent passwd "$OWNER_UID")
if [ "$USER_HOME" != /workspace ]; then  # ssh would look for the key in the home of that user
  echo "uid $OWNER_UID already belongs to $USER_NAME (home $USER_HOME), not to a user with home /workspace" >&2
  exit 1
fi

ssh -V
getent passwd "$OWNER_UID"
echo "=== Setup done: docker exec -it -u $USER_NAME -w $REPO <container> bash   (on the host)"
