# Slim GPU image for Xinference.
#
# The image intentionally ships NO inference engine (vllm / sglang / ...).
# At model launch time the worker creates a per-model virtual environment
# (XINFERENCE_ENABLE_VIRTUAL_ENV, on by default) and installs the engine
# packages declared by the model spec — from public PyPI online, or from the
# private mirror wired up by the offline Docker Compose profile.
#
# What IS baked in, and why:
#   * Python 3.12 toolchain — runtime interpreter, matches the Python the
#     per-model venvs are created from.
#   * torch / torchvision / torchaudio / torchcodec (cu130) — model specs
#     reference the parent environment's versions via #system_torch#-style
#     placeholders, which fail if the package is absent; keeping one shared
#     CUDA torch stack also lets venvs reuse it via
#     XINFERENCE_VIRTUAL_ENV_SKIP_INSTALLED instead of downloading
#     multi-GB wheels per model.
#   * transformers + accelerate — the default Transformers engine's venv
#     requirements verbatim, so the most common engine works out of the box
#     and its venv install is a no-op.
#   * xinference itself with its core dependencies and the prebuilt Web UI.
#   * ffmpeg, libsndfile, git, curl — runtime tools the venv-installed model
#     stacks rely on (audio decoding, VCS package installs, health probes).
#
# The CUDA *devel* base keeps nvcc and the CUDA headers as a fallback for
# model venv packages that only ship sdists and compile CUDA extensions on
# the fly. Requires a host driver supporting CUDA >= 13.0.
#
# This single Dockerfile builds both linux/amd64 and linux/arm64 (the CI
# aarch64 job builds it on an arm64 runner): the CUDA base image, the
# deadsnakes Python packages, and every pinned wheel in
# requirements-runtime.txt ship both architectures. Keep any additions
# multi-arch, or gate them on TARGETARCH (declare `ARG TARGETARCH` inside
# the stage first — undeclared it expands to an empty string).

# ---------------------------------------------------------------------------
# Stage 1: build the Web UI static export
# ---------------------------------------------------------------------------
FROM node:20-slim AS web-builder

WORKDIR /opt/inference/frontend
COPY frontend ./
RUN npm ci && npm run build
# `npm run build` stages the static export at ../xinference/ui/web/dist via
# its postbuild hook (frontend/scripts/stage-export.mjs).

# ---------------------------------------------------------------------------
# Stage 2: runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive

# System packages:
#   * python3.12 (deadsnakes) — Ubuntu 22.04 ships 3.10, the project runtime
#     is 3.12; python3.12-dev provides Python.h for sdist builds in venvs.
#   * build-essential / ninja-build — host compiler and build driver for
#     nvcc runtime JIT (vLLM engine startup shells out to ninja) and for
#     sdist fallbacks.
#   * git — model venvs may install VCS requirements.
#   * ffmpeg / libsndfile1 — audio decoding for venv-installed audio stacks
#     (librosa/soundfile/torchcodec load these shared libraries).
#   * libgl1 / libglib2.0-0 — required by non-headless opencv variants that
#     model venvs may pull in.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
      | gpg --dearmor -o /usr/share/keyrings/deadsnakes.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes.gpg] https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" \
      > /etc/apt/sources.list.d/deadsnakes.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential ninja-build git procps \
      ffmpeg libsndfile1 libgl1 libglib2.0-0 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Shared CUDA torch stack (see header). requirements-runtime.txt is also
# consumed by the offline pypiserver build, so parent-version constraints can
# never refer to a version that the mirror omitted. The +cu130 local version is
# significant: the worker reads it to auto-configure the matching PyTorch
# wheel index for per-model venv installs. pandas backs the #system_pandas#
# placeholder used by several audio model specs.
COPY xinference/deploy/docker/requirements-runtime.txt /opt/requirements-runtime.txt
RUN pip install --no-cache-dir -i "$PIP_INDEX" --upgrade pip "setuptools<82" wheel && \
    pip install --no-cache-dir --constraint /opt/requirements-runtime.txt \
      torch torchvision torchaudio torchcodec --index-url "$TORCH_INDEX" && \
    pip install --no-cache-dir --constraint /opt/requirements-runtime.txt \
      -i "$PIP_INDEX" numpy pandas

COPY . /opt/inference
COPY --from=web-builder /opt/inference/xinference/ui/web/dist /opt/inference/xinference/ui/web/dist
WORKDIR /opt/inference

# NO_WEB_UI=1: the static Web UI was already staged by the builder stage and
# is packaged via MANIFEST.in; skip the npm build hook of setup.py.
# Keep only the engine core in the slim parent image. Optional model-side and
# format-specific quantization dependencies live in model virtualenv specs;
# they are installed only into matching per-model venvs and are also collected
# by the offline pypiserver generator.
# uv (a runtime dependency anyway, it powers the per-model venv installer)
# also does the image install: parallel downloads with per-request timeouts
# recover from throttled/stalled index connections that serial pip rides
# to a crawl.
RUN pip install --no-cache-dir -i "$PIP_INDEX" uv && \
    NO_WEB_UI=1 uv pip install --system --no-cache --index-url "$PIP_INDEX" \
      --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
      --constraint /opt/requirements-runtime.txt \
      ".[otel]" transformers accelerate

# Parity with previous images: no entrypoint, plain shell by default; deploy
# commands (xinference-local / supervisor / worker) are passed explicitly.
ENTRYPOINT []
CMD ["/bin/bash"]
