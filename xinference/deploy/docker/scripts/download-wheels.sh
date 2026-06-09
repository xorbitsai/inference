#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download-wheels.sh — Download wheels from non-PyPI sources.
#
# Usage:
#   download-wheels.sh <arch> <output_dir>
#
#   arch       : amd64 | arm64
#   output_dir : absolute path to save downloaded wheels
#
# Sources:
#   - GitHub releases (flash-attn, flashinfer, etc.)
#   - Custom package indexes (xllamacpp, etc.)
#
# Security note:
#   External wheel sources are trusted but not checksum-verified at download
#   time.  Consider pinning known-good hashes for production use.
# ---------------------------------------------------------------------------
set -euo pipefail

ARCH="${1:?missing arch}"
OUTDIR="${2:?missing output dir}"

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
# Wheel URLs are hardcoded per architecture.  When updating, verify the
# release exists before committing.
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
        wget -q --show-progress -O "${OUTDIR}/${fname}" "${url}" || {
            log "WARNING: failed to download ${url} (may not exist yet)"
        }
    done
}

# ------------------------------------------------------------------
# xllamacpp — custom package index
# https://xorbitsai.github.io/xllamacpp/whl/
# ------------------------------------------------------------------
download_xllamacpp() {
    local index_url="https://xorbitsai.github.io/xllamacpp/whl/cu128"

    log "Downloading xllamacpp from ${index_url}"
    pip download \
        --no-cache-dir \
        --index-url "${index_url}" \
        xllamacpp \
        -d "${OUTDIR}" \
        || log "WARNING: failed to download xllamacpp"
}

# ------------------------------------------------------------------
# triton — available for both amd64 and arm64
# ------------------------------------------------------------------
download_triton() {
    log "Downloading triton from PyPI"
    pip download \
        --no-cache-dir \
        triton \
        -d "${OUTDIR}" \
        || log "WARNING: failed to download triton"
}

# ------------------------------------------------------------------
# flashinfer — prebuilt wheels (amd64 only)
# ------------------------------------------------------------------
download_flashinfer() {
    if [ "${ARCH}" != "amd64" ]; then
        log "Skipping flashinfer — no prebuilt wheels for ${ARCH}"
        return 0
    fi

    # flashinfer is typically installed via pip from PyPI,
    # but specific CUDA versions may need custom wheels.
    log "Downloading flashinfer from PyPI"
    pip download \
        --no-cache-dir \
        flashinfer \
        -d "${OUTDIR}" \
        || log "WARNING: failed to download flashinfer (may need custom source)"
}

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
log "Arch: ${ARCH}"
log "Output: ${OUTDIR}"

download_flash_attn
download_xllamacpp
download_triton
download_flashinfer

log "Done."
if compgen -G "${OUTDIR}"/*.whl > /dev/null 2>&1; then
    log "Wheels in ${OUTDIR}:"
    ls -la "${OUTDIR}"/*.whl
else
    log "(no new wheels downloaded by this script)"
fi
