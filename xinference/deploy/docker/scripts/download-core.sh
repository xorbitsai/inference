#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download-core.sh — Download core packages (common + arch-compiled).
#
# Usage:
#   download-core.sh <arch> <output_dir>
#
#   arch       : amd64 | arm64
#   output_dir : absolute path to save downloaded wheels
#
# This script handles packages that are NOT CUDA-version-specific:
#   - common/base.txt, common/ml.txt, common/models.txt
#   - <arch>/compiled.txt
# ---------------------------------------------------------------------------
set -euo pipefail

ARCH="${1:?missing arch}"
OUTDIR="${2:?missing output dir}"

# Validate OUTDIR is an absolute path
if [[ "${OUTDIR}" != /* ]]; then
    echo "ERROR: output_dir must be an absolute path, got: ${OUTDIR}" >&2
    exit 1
fi

# Validate ARCH
case "${ARCH}" in
    amd64|arm64) ;;
    *)
        echo "ERROR: unknown arch '${ARCH}' (expected amd64 or arm64)" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTDIR}"

log() { echo "[download-core] $*"; }

BASE="/build/pypi-requirements"

# ------------------------------------------------------------------
# Common packages — pure Python, diverse sources.
# No --only-binary because some packages are sdist-only
# (e.g. transformers-stream-generator). Build tools are pre-installed
# in the Dockerfile.
# ------------------------------------------------------------------
log "Downloading common packages (base + ml + models)..."
pip download --no-cache-dir \
    -r "${BASE}/common/base.txt" \
    -r "${BASE}/common/ml.txt" \
    -r "${BASE}/common/models.txt" \
    -d "${OUTDIR}/common/"

# ------------------------------------------------------------------
# Compiled packages — arch-specific but not CUDA-specific.
# These all have prebuilt wheels (uvloop, onnxruntime).
# ------------------------------------------------------------------
COMPILED="${BASE}/${ARCH}/compiled.txt"
if [ -f "${COMPILED}" ]; then
    log "Downloading compiled packages for ${ARCH}..."
    pip download --no-cache-dir --only-binary :all: \
        -r "${COMPILED}" \
        -d "${OUTDIR}/compiled/"
else
    log "WARNING: ${COMPILED} not found, skipping"
fi

log "Done."
