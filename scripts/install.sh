#!/bin/sh
# Xinference installer
#
#   curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
#
# Installs the *base* Xinference framework (CLI + server + Web UI) into a
# dedicated, isolated virtualenv managed by uv, and links its commands into
# ~/.local/bin. Because Xinference depends on PyTorch, the installer selects a
# hardware-appropriate PyTorch build (CPU / CUDA / ROCm / Intel XPU) via uv's
# --torch-backend, which requires the `uv pip` interface (and therefore a
# managed venv rather than `uv tool install`).
#
# IMPORTANT: this installs only the base framework. Model-serving backends
# (transformers, vLLM, sglang, MLX, llama.cpp, image/audio/video, ...) are NOT
# installed. To serve models, install the relevant extra afterwards, or use the
# functional Quick Start: pip install "xinference[all]".
#
# Options (environment variables):
#   XINFERENCE_BACKEND   PyTorch backend to install. Default: auto.
#                        auto  - detect GPU/driver and pick the best build
#                        cpu   - CPU-only PyTorch
#                        cu128, cu126, ... - a specific CUDA build
#                        rocm  - AMD ROCm build (Linux only)
#                        xpu   - Intel GPU build (Linux only)
#                        On macOS this is ignored (a single universal wheel).
#   XINFERENCE_EXTRAS    optional extras, e.g. XINFERENCE_EXTRAS=all or
#                        XINFERENCE_EXTRAS=vllm,transformers (default: none).
#                        Some extras (e.g. vllm) require Linux + CUDA.
#   XINFERENCE_VERSION   pin a version, e.g. XINFERENCE_VERSION=1.8.1
#   XINFERENCE_PYTHON    Python version for the venv (default: 3.12)
#   XINFERENCE_HOME_DIR  install location (default: ~/.xinference/venv)
#
# Prefer to manage the environment yourself? The equivalent manual install is:
#   uv venv --python 3.12 .venv && . .venv/bin/activate
#   uv pip install --torch-backend auto xinference
set -eu

ORIG_PATH="$PATH"

APP="xinference"
SERVE_CMD="xinference-local"
PORT="9997"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="${XINFERENCE_HOME_DIR:-$HOME/.xinference/venv}"
PY_VERSION="${XINFERENCE_PYTHON:-3.12}"
# Commands exposed by the package; symlinked from the venv into BIN_DIR.
CMDS="xinference xinference-local xinference-supervisor xinference-worker"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }
err() {
  printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Linux) platform="linux" ;;
  Darwin)
    platform="macos"
    case "$arch" in
      arm64) ;;  # Apple silicon: fully supported
      x86_64)
        warn "Intel-based macOS detected. Recent PyTorch releases no longer ship"
        warn "Intel-mac wheels, so installation may fail or pin an old torch."
        ;;
    esac
    ;;
  *) err "Unsupported OS '$os'. On Windows, install with: pip install \"${APP}[all]\" (in a virtualenv)." ;;
esac

# ---------------------------------------------------------------------------
# Preflight: warn about an existing installation / environment conflicts
# ---------------------------------------------------------------------------
# shellcheck disable=SC2030  # the PATH override is deliberately subshell-local
existing="$( ( PATH="$ORIG_PATH"; command -v "$SERVE_CMD" ) 2>/dev/null || true )"
if [ -n "$existing" ]; then
  warn "An existing '$SERVE_CMD' was found at: $existing"
  warn "This installer creates an isolated environment and will shadow it via $BIN_DIR."
fi
if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
  warn "You are inside an active virtualenv/conda environment."
  warn "uv installs into a separate isolated env; packages from the active"
  warn "environment are NOT reused. To install into the current environment"
  warn "instead, run: pip install \"${APP}[all]\""
fi

# ---------------------------------------------------------------------------
# Ensure uv is available (isolates the install; avoids system-Python/PEP 668).
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (Python package/tool manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    # shellcheck disable=SC2031  # this runs in the main shell, not a subshell
    [ -d "$d" ] && PATH="$d:$PATH"
  done
  export PATH
fi
command -v uv >/dev/null 2>&1 || err "uv not found on PATH after install; open a new shell and re-run."

# ---------------------------------------------------------------------------
# Resolve the PyTorch backend
# ---------------------------------------------------------------------------
backend="${XINFERENCE_BACKEND:-auto}"
if [ "$platform" = "macos" ]; then
  # macOS ships a single PyTorch wheel; --torch-backend does not apply.
  torch_arg=""
  [ "$backend" != "auto" ] && warn "XINFERENCE_BACKEND='$backend' ignored on macOS."
else
  torch_arg="--torch-backend=$backend"
fi

# Build the package spec: name[extras]==version, with each part optional.
spec="$APP"
if [ -n "${XINFERENCE_EXTRAS:-}" ]; then
  spec="${spec}[${XINFERENCE_EXTRAS}]"
fi
if [ -n "${XINFERENCE_VERSION:-}" ]; then
  version="${XINFERENCE_VERSION#v}"  # strip a leading 'v' (v1.8.1 -> 1.8.1)
  [ -n "$version" ] || err "XINFERENCE_VERSION='$XINFERENCE_VERSION' is not a valid version."
  spec="$spec==$version"
fi

# ---------------------------------------------------------------------------
# Create the venv and install
# ---------------------------------------------------------------------------
info "Creating environment at $VENV_DIR (Python $PY_VERSION)..."
uv venv --python "$PY_VERSION" "$VENV_DIR"

info "Installing $spec (torch backend: ${backend})..."
# shellcheck disable=SC2086  # torch_arg is intentionally unquoted (may be empty)
VIRTUAL_ENV="$VENV_DIR" uv pip install --python "$VENV_DIR/bin/python" $torch_arg "$spec"

# ---------------------------------------------------------------------------
# Link commands into ~/.local/bin
# ---------------------------------------------------------------------------
mkdir -p "$BIN_DIR"
for c in $CMDS; do
  if [ -x "$VENV_DIR/bin/$c" ]; then
    ln -sf "$VENV_DIR/bin/$c" "$BIN_DIR/$c"
  fi
done

printf '\n'
info "Installed the base Xinference framework. Next steps:"
printf '\n'
printf '  Start the server:   %s\n' "$SERVE_CMD"
printf '  Open the Web UI:    http://127.0.0.1:%s\n' "$PORT"
printf '\n'
if [ -z "${XINFERENCE_EXTRAS:-}" ]; then
  printf '  No model backend was installed. Add one when you need it, e.g.:\n'
  printf '    uv pip install --python "%s/bin/python" "%s[transformers]"\n' "$VENV_DIR" "$APP"
  printf '  (transformers / vllm [Linux+CUDA] / mlx [Apple silicon] / llama_cpp / all)\n'
  printf '\n'
fi

# Warn if BIN_DIR is not on the parent shell's PATH. Use a subshell to scope the
# PATH override: a leading `PATH=... command -v` assignment on a shell built-in
# is not clearly specified by POSIX, so the subshell form is unambiguous.
if ! ( PATH="$ORIG_PATH"; command -v "$SERVE_CMD" ) >/dev/null 2>&1; then
  warn "'$BIN_DIR' does not appear to be on your PATH in this shell."
  warn "Open a new terminal, or run: export PATH=\"$BIN_DIR:\$PATH\""
fi
