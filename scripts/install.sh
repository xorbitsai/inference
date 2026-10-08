#!/bin/sh
# Install and run Xinference in a persistent, isolated uv tool environment.
# curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
set -eu

fail() { printf 'error: %s\n' "$1" >&2; exit 1; }
info() { printf '==> %s\n' "$1"; }
os="$(uname -s)"
case "$os" in
  Linux|Darwin) ;;
  *) fail 'Use scripts/install.ps1 on Windows.' ;;
esac
if [ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]; then
  info 'Creating a separate uv tool environment; the active Python environment is not reused.'
fi
if ! command -v uv >/dev/null 2>&1 || ! uv tool install --help | grep -q -- '--torch-backend'; then
  info 'Installing a compatible uv...'
  uv_script="$(mktemp)"
  trap 'rm -f "$uv_script"' 0
  curl -fsSL https://astral.sh/uv/install.sh -o "$uv_script"
  sh "$uv_script"
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH
fi
command -v uv >/dev/null 2>&1 || fail 'uv was not found after installation. Add its binary directory to PATH.'
if ! uv tool install --help | grep -q -- '--torch-backend'; then
  fail 'This installer requires a uv version supporting uv tool install --torch-backend (tested with 0.11.26). Upgrade uv first.'
fi
# The same Python transaction handles installation, upgrade, and rollback.
installer="$(dirname "$0")/manage_install.py"
if [ ! -f "$installer" ]; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' 0
  curl -fsSL "${XINFERENCE_INSTALLER_URL:-https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/manage_install.py}" -o "$installer"
fi
uv run --no-project --no-config --python 3.12 python "$installer"
