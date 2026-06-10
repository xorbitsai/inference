#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download-wheels.sh — Download wheels from sources outside PyPI / PyTorch.
#
# Usage:
#   download-wheels.sh <arch> <cuda_versions> <output_dir>
#
#   arch          : amd64 | arm64
#   cuda_versions : space-separated list, e.g. "cu126 cu128"
#   output_dir    : absolute path to save downloaded wheels
#
# Sources:
#   - GitHub releases (flash-attn prebuilt wheels)
#   - Custom package index (xllamacpp)
#
# Packages that ARE on PyPI (triton, flashinfer) are handled by
# download-cuda.sh via cuda-ext.txt / vllm.txt — NOT duplicated here.
#
# Security note:
#   External wheel sources are trusted but not checksum-verified at download
#   time.  Consider pinning known-good hashes for production use.
# ---------------------------------------------------------------------------
set -euo pipefail

ARCH="${1:?missing arch}"
CUDA_VERSIONS="${2:?missing cuda versions}"
OUTDIR="${3:?missing output dir}"

# Validate OUTDIR is an absolute path (defense against accidental relative paths)
if [[ "${OUTDIR}" != /* ]]; then
    echo "ERROR: output_dir must be an absolute path, got: ${OUTDIR}" >&2
    exit 1
fi

# Validate ARCH against known values
case "${ARCH}" in
    amd64|arm64) ;;
    *)
        echo "ERROR: unknown arch '${ARCH}' (expected amd64 or arm64)" >&2
        exit 1
        ;;
esac

mkdir -p "${OUTDIR}"

log() { echo "[download-wheels] $*"; }

# ------------------------------------------------------------------
# flash-attention — prebuilt wheels from GitHub
# https://github.com/mjun0812/flash-attention-prebuild-wheels
#
# Wheel URLs are hardcoded per architecture and CUDA version.
# When a new CUDA version or flash-attn release comes out, update the
# wheels array below.  Verify the release exists on GitHub before
# committing.
# ------------------------------------------------------------------
download_flash_attn() {
    local base_url="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download"

    local wheels=()
    case "${ARCH}" in
        amd64)
            # CUDA 12.8 + torch 2.11, Python 3.12, x86_64
            wheels=(
                "v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
                "v0.9.3/flash_attn-2.7.4+cu126torch2.10-cp312-cp312-linux_x86_64.whl"
            )
            ;;
        arm64)
            # aarch64 prebuilt wheels (if available)
            wheels=(
                "v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_aarch64.whl"
            )
            ;;
    esac

    for whl in "${wheels[@]}"; do
        local fname="${whl##*/}"

        # Sanity check: filename must end with .whl and not contain path traversal
        if [[ "${fname}" != *.whl ]] || [[ "${fname}" == *".."* ]] || [[ "${fname}" == *"/"* ]]; then
            log "ERROR: suspicious wheel filename: ${fname}"
            continue
        fi

        local url="${base_url}/${whl}"
        log "Downloading flash-attn: ${fname}"
        wget -q --show-progress -O "${OUTDIR}/external/${fname}" "${url}" || {
            log "WARNING: failed to download ${url} (may not exist yet)"
        }
    done
}

# ------------------------------------------------------------------
# xllamacpp — custom package index (not on PyPI)
# https://xorbitsai.github.io/xllamacpp/whl/
#
# Iterates over each CUDA version to find per-CUDA wheels.
# ------------------------------------------------------------------
download_xllamacpp() {
    for cu in ${CUDA_VERSIONS}; do
        local index_url="https://xorbitsai.github.io/xllamacpp/whl/${cu}"

        log "Downloading xllamacpp for ${cu} from ${index_url}..."
        pip download \
            --no-cache-dir --only-binary :all: \
            --index-url "${index_url}" \
            xllamacpp \
            -d "${OUTDIR}/external/" \
            || log "WARNING: failed to download xllamacpp for ${cu} (index may not exist)"
    done
}

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
log "Arch: ${ARCH}"
log "CUDA versions: ${CUDA_VERSIONS}"
log "Output: ${OUTDIR}"

download_flash_attn
download_xllamacpp

log "Done."
if compgen -G "${OUTDIR}"/*.whl > /dev/null 2>&1; then
    log "Wheels in ${OUTDIR}:"
    ls -la "${OUTDIR}"/*.whl
else
    log "(no new wheels downloaded by this script)"
fi
