#!/bin/sh
# Xinference installer
#
#   curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
#
# Installs the `xinference` package as an isolated uv tool, so nothing touches
# your system Python and PEP 668 (externally-managed-environment) never bites.
# On success the `xinference` / `xinference-local` commands are available; start
# the server and open the Web UI.
#
# Options (environment variables):
#   XINFERENCE_VERSION   pin a specific version, e.g. XINFERENCE_VERSION=1.8.1
#   XINFERENCE_EXTRAS    install optional backends, e.g. XINFERENCE_EXTRAS=all
#                        or XINFERENCE_EXTRAS=vllm,transformers (default: none).
#                        Note: some extras (e.g. vllm) require Linux + CUDA and
#                        may fail to install on macOS or CPU-only machines.
#
# Prefer not to pipe curl into sh? The equivalent manual install is:
#   uv tool install xinference          # or, in a venv: pip install xinference
set -eu

# The user's PATH before this script mutates it (below, when bootstrapping uv).
# Used at the end to warn correctly about whether the parent shell will find the
# installed command.
ORIG_PATH="$PATH"

APP="xinference"
# Command used to start the server locally (the `xinference` command itself is
# the client CLI; `xinference-local` boots a standalone supervisor + worker).
SERVE_CMD="xinference-local"
PORT="9997"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$1" >&2; }
err() {
  printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

# uv supports Linux and macOS. Windows users should use pip in a venv.
os="$(uname -s)"
case "$os" in
  Linux | Darwin) ;;
  *) err "Unsupported OS '$os'. On Windows, install with: pip install \"$APP[all]\" (in a virtualenv)." ;;
esac

# Ensure uv is available (isolates the install; avoids system-Python/PEP 668).
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv (Python tool manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs into ~/.local/bin (or ~/.cargo/bin on older installers); make it
  # visible to the rest of this script without requiring a new shell.
  for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
    [ -d "$d" ] && PATH="$d:$PATH"
  done
  export PATH
fi
command -v uv >/dev/null 2>&1 || err "uv not found on PATH after install; open a new shell and re-run."

# Build the package spec: name[extras]==version, with each part optional.
spec="$APP"
if [ -n "${XINFERENCE_EXTRAS:-}" ]; then
  spec="$spec[$XINFERENCE_EXTRAS]"
fi
if [ -n "${XINFERENCE_VERSION:-}" ]; then
  # Strip a leading 'v' (e.g. v1.8.1 -> 1.8.1) so a git-tag-style value works.
  version="${XINFERENCE_VERSION#v}"
  [ -n "$version" ] || err "XINFERENCE_VERSION='$XINFERENCE_VERSION' is not a valid version."
  spec="$spec==$version"
fi

info "Installing $spec ..."
uv tool install --upgrade "$spec"

printf '\n'
info "Installed. Next steps:"
printf '\n'
printf '  Start the server:   %s\n' "$SERVE_CMD"
printf '  Open the Web UI:    http://127.0.0.1:%s\n' "$PORT"
if [ -z "${XINFERENCE_EXTRAS:-}" ]; then
  printf '\n'
  printf '  This installed the base package. To add an inference backend later, e.g.:\n'
  printf '    uv tool install --upgrade "%s[transformers]"   # PyTorch / transformers\n' "$APP"
  printf '    uv tool install --upgrade "%s[vllm]"           # vLLM (Linux + CUDA)\n' "$APP"
  printf '    uv tool install --upgrade "%s[mlx]"            # MLX (Apple silicon)\n' "$APP"
  printf '  Or reinstall everything with XINFERENCE_EXTRAS=all.\n'
fi
printf '\n'

# Check against the parent shell's original PATH. Scope the PATH override with a
# subshell rather than a leading `PATH=... command -v` assignment: the latter's
# effect on a shell built-in is not clearly specified by POSIX, so the subshell
# form is unambiguous across shells.
if ! ( PATH="$ORIG_PATH"; command -v "$SERVE_CMD" ) >/dev/null 2>&1; then
  warn "'$SERVE_CMD' is not on your PATH in this shell yet."
  if ( PATH="$ORIG_PATH"; command -v uv ) >/dev/null 2>&1; then
    warn "Run 'uv tool update-shell' and open a new terminal, then run '$SERVE_CMD'."
  else
    # uv was just installed by this script and isn't on the parent shell's PATH.
    warn "Open a new terminal, or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
  fi
fi
