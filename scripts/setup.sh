#!/usr/bin/env bash
# One-shot environment setup + verification. Safe to re-run (idempotent).
#   scripts/setup.sh            detect environment, build/install, verify device, ensure data
#   scripts/setup.sh --no-data  skip dataset download
# Exit codes: 0 ok, 2 no accelerator (CPU only), 3 data download failed, 1 other error.
set -euo pipefail
cd "$(dirname "$0")/.."
WANT_DATA=1
[[ "${1:-}" == "--no-data" ]] && WANT_DATA=0

log() { printf '\n[setup] %s\n' "$*"; }

in_container() { [[ -f /.dockerenv || -f /run/.containerenv || -n "${CONTAINER_ID:-}" || -n "${VAST_CONTAINERLABEL:-}" ]]; }

if in_container; then
    MODE=cloud            # e.g. Vast.ai: the instance itself is the container, run python directly
elif command -v podman >/dev/null && command -v nvidia-smi >/dev/null; then
    MODE=host-cuda
elif command -v podman >/dev/null && [[ -e /dev/dri/renderD128 ]]; then
    MODE=host-xpu
else
    echo "[setup] no supported environment: need podman on the host, or to run inside a CUDA container" >&2
    exit 1
fi
log "mode=$MODE"

case $MODE in
    host-xpu)  PY=(./run.sh python);        NETPY=(./run.sh --net python);        ./run.sh --build ;;
    host-cuda) PY=(./run.sh --cuda python); NETPY=(./run.sh --cuda --net python); ./run.sh --cuda --build ;;
    cloud)
        PY=(python); NETPY=(python)
        export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
        if ! python -c "import transformers, accelerate, datasets, yaml" 2>/dev/null; then
            log "installing requirements.txt"
            pip install --no-cache-dir -r requirements.txt
        fi
        ;;
esac

log "device check"
OUT=$("${PY[@]}" scripts/check_device.py 2>&1) || { echo "$OUT"; exit 1; }
echo "$OUT"
if ! grep -qE '^(xpu|cuda) .*matmul' <<<"$OUT"; then
    echo "[setup] WARNING: no GPU visible, only CPU. Ask the user before running on CPU." >&2
    exit 2
fi

if [[ $WANT_DATA == 1 ]]; then
    if ls data/sets/full/*.jsonl >/dev/null 2>&1; then
        log "data/sets/full present, keeping it (delete it to rebuild)"
    else
        log "building datasets (network)"
        "${NETPY[@]}" -m abr.make_data || exit 3
    fi
fi
log "OK. next: ${PY[*]} -m abr.run_all [--preset smoke]   (host modes: add --net on the first run to download the model)"
