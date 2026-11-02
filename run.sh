#!/usr/bin/env bash
# Run a command inside the experiment container.
#   ./run.sh [--net] [--cuda] [--build] [--] <cmd...>
#   ./run.sh python scripts/check_device.py
#   ./run.sh --net python -m abr.make_data          # network only when downloading
#   ./run.sh --build                                # (re)build the image only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAVOR=xpu
NET=none
BUILD=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --net)   NET=pasta; shift ;;
        --cuda)  FLAVOR=cuda; shift ;;
        --build) BUILD=1; shift ;;
        --)      shift; break ;;
        *)       break ;;
    esac
done
IMAGE="localhost/abresearch:${FLAVOR}"

if [[ $BUILD == 1 ]] || ! podman image exists "$IMAGE"; then
    podman build -t "$IMAGE" -f "$HERE/Containerfile.${FLAVOR}" "$HERE"
fi
[[ $# -eq 0 ]] && exit 0

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

DEV_ARGS=()
if [[ $FLAVOR == xpu ]]; then
    DEV_ARGS=(--device /dev/dri --group-add keep-groups)
else
    DEV_ARGS=(--device nvidia.com/gpu=all)   # needs nvidia-container-toolkit CDI
fi

TTY=()
[[ -t 0 && -t 1 ]] && TTY=(-it)

# --userns=keep-id: the container's exp user (uid 1000) maps to the host user, so files in /work stay owned by you.
exec podman run --rm "${TTY[@]}" \
    "${DEV_ARGS[@]}" \
    --userns=keep-id \
    --network="$NET" \
    --read-only \
    --tmpfs /tmp:rw,size=4g,mode=1777 \
    --shm-size=2g \
    -v "$HERE:/work:Z" \
    -v "$HF_CACHE:/hf:Z" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HUB_OFFLINE="$([[ $NET == none ]] && echo 1 || echo 0)" \
    "$IMAGE" "$@"
