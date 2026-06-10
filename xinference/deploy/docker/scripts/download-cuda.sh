#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download-cuda.sh — Download CUDA-version-specific packages.
#
# Usage:
#   download-cuda.sh <arch> <cuda_versions> <output_dir>
#
#   arch          : amd64 | arm64
#   cuda_versions : space-separated list, e.g. "cu126 cu128"
#   output_dir    : absolute path to save downloaded wheels
#
# Handles:
#   - torch ecosystem (per CUDA version, different PyTorch index URLs)
#   - torch CPU variant
#   - CUDA extensions (per CUDA version)
#   - vLLM ecosystem (amd64 only, per CUDA version)
#
# Each CUDA version loop collects failures independently — one failing
# version does not block the others.
# ---------------------------------------------------------------------------
set -euo pipefail

ARCH="${1:?missing arch}"
CUDA_VERSIONS="${2:?missing cuda versions}"
OUTDIR="${3:?missing output dir}"

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

log() { echo "[download-cuda] $*"; }

BASE="/build/pypi-requirements"

# ------------------------------------------------------------------
# PyTorch ecosystem — per CUDA version
# Each version uses a different --index-url, same torch.txt file.
# ------------------------------------------------------------------
log "Downloading PyTorch ecosystem..."
torch_failed=""

for cu in ${CUDA_VERSIONS}; do
    req="${BASE}/${ARCH}/torch.txt"
    if [ ! -f "${req}" ]; then
        log "WARNING: ${req} not found, skipping torch-${cu}"
        continue
    fi
    log "  torch ${cu} (index: https://download.pytorch.org/whl/${cu})..."
    pip download --no-cache-dir --only-binary :all: \
        --index-url "https://download.pytorch.org/whl/${cu}" \
        -r "${req}" -d "${OUTDIR}/${cu}/torch/" \
        || torch_failed="${torch_failed} torch-${cu}"
done

if [ -n "${torch_failed}" ]; then
    log "WARNING: failed to download:${torch_failed}"
fi

# ------------------------------------------------------------------
# PyTorch CPU variant — single pass
# ------------------------------------------------------------------
TORCH_CPU="${BASE}/${ARCH}/torch-cpu.txt"
if [ -f "${TORCH_CPU}" ]; then
    log "Downloading PyTorch CPU variant..."
    pip download --no-cache-dir --only-binary :all: \
        --index-url "https://download.pytorch.org/whl/cpu" \
        -r "${TORCH_CPU}" -d "${OUTDIR}/cpu/" \
        || log "WARNING: torch-cpu download failed"
fi

# ------------------------------------------------------------------
# CUDA extensions — per CUDA version
# Downloaded from standard PyPI, reusing the same cuda-ext.txt.
# ------------------------------------------------------------------
log "Downloading CUDA extensions..."
ext_failed=""

for cu in ${CUDA_VERSIONS}; do
    req="${BASE}/${ARCH}/cuda-ext.txt"
    if [ ! -f "${req}" ]; then
        log "WARNING: ${req} not found, skipping cuda-ext-${cu}"
        continue
    fi
    log "  cuda-ext ${cu}..."
    pip download --no-cache-dir --only-binary :all: \
        -r "${req}" -d "${OUTDIR}/${cu}/cuda-ext/" \
        || ext_failed="${ext_failed} cuda-ext-${cu}"
done

if [ -n "${ext_failed}" ]; then
    log "WARNING: failed to download:${ext_failed}"
fi

# ------------------------------------------------------------------
# vLLM ecosystem — amd64 only
# No prebuilt arm64 wheels for vllm or flashinfer.
# ------------------------------------------------------------------
if [ "${ARCH}" = "amd64" ]; then
    log "Downloading vLLM ecosystem..."
    vllm_failed=""

    for cu in ${CUDA_VERSIONS}; do
        req="${BASE}/${ARCH}/vllm.txt"
        if [ ! -f "${req}" ]; then
            continue
        fi
        log "  vllm ${cu}..."
        pip download --no-cache-dir --only-binary :all: \
            -r "${req}" -d "${OUTDIR}/${cu}/vllm/" \
            || vllm_failed="${vllm_failed} vllm-${cu}"
    done

    if [ -n "${vllm_failed}" ]; then
        log "WARNING: failed to download:${vllm_failed}"
    fi
else
    log "Skipping vLLM — amd64 only (no prebuilt arm64 wheels)"
fi

log "Done."
