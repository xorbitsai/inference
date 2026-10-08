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
python_version="${XINFERENCE_PYTHON:-3.12}"
backend="${XINFERENCE_BACKEND:-auto}"
start="${XINFERENCE_START:-1}"
service="${XINFERENCE_SERVICE:-none}"
host="${XINFERENCE_HOST:-127.0.0.1}"
port="${XINFERENCE_PORT:-9997}"
case "$start" in 0|1) ;; *) fail 'XINFERENCE_START must be 0 or 1.' ;; esac
case "$service" in none|user|system) ;; *) fail 'XINFERENCE_SERVICE must be none, user, or system.' ;; esac
case "$port" in ''|*[!0-9]*) fail 'XINFERENCE_PORT must be an integer.' ;; esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
  fail 'XINFERENCE_PORT must be between 1 and 65535.'
fi
if [ "$service" = system ] && { [ "$(id -u)" -ne 0 ] || [ -n "${SUDO_USER:-}" ]; }; then
  fail 'System service installation requires root. Install normally first, then run xinference service --system install --start with sudo and an absolute command path.'
fi
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
# An override controls the uv tool store, not the persistent model data directory.
if [ -n "${XINFERENCE_HOME_DIR:-}" ]; then
  UV_TOOL_DIR="$XINFERENCE_HOME_DIR"
  export UV_TOOL_DIR
fi
spec="${XINFERENCE_PACKAGE:-xinference}"
if [ -z "${XINFERENCE_PACKAGE:-}" ]; then
  [ -z "${XINFERENCE_EXTRAS:-}" ] || spec="${spec}[$XINFERENCE_EXTRAS]"
  if [ -n "${XINFERENCE_VERSION:-}" ]; then
    version="${XINFERENCE_VERSION#v}"
    [ -n "$version" ] || fail 'XINFERENCE_VERSION must contain a version.'
    spec="$spec==$version"
  fi
fi
set -- --python "$python_version"
if [ "$os" != Darwin ]; then
  set -- "$@" --torch-backend "$backend"
elif [ "$(uname -m)" = x86_64 ]; then
  info 'Intel macOS detected: current PyTorch releases do not provide Intel Mac wheels.'
fi
info "Installing $spec..."
uv tool install "$@" "$spec"
tool_dir="$(uv tool dir)"
cli="$tool_dir/xinference/bin/xinference"
server="$tool_dir/xinference/bin/xinference-local"
[ -x "$server" ] || fail "Xinference was not installed at $tool_dir/xinference."
info "Installed Xinference. Commands are linked in $(uv tool dir --bin)."
info 'Model engines use Xinference model environments; extras can also be preinstalled with XINFERENCE_EXTRAS.'
if [ "$service" != none ]; then
  set -- service
  [ "$service" != system ] || set -- "$@" --system
  set -- "$@" install --host "$host" --port "$port"
  [ -z "${XINFERENCE_HOME:-}" ] || set -- "$@" --home "$XINFERENCE_HOME"
  [ "$start" != 1 ] || set -- "$@" --start
  "$cli" "$@"
elif [ "$start" = 1 ]; then
  info "Starting Xinference on $host:$port (Ctrl+C to stop)..."
  exec "$server" --host "$host" --port "$port"
else
  info "Start the server: $server --host $host --port $port"
fi
