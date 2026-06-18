#!/bin/bash
# Extract all pip-installable Python packages for the given platform.
# Usage: ./list-packages.sh --platform=amd64
#        ./list-packages.sh --platform=arm64

set -euo pipefail

PLATFORM="${1:?Usage: $0 --platform=amd64|arm64}"

case "$PLATFORM" in
  --platform=amd64)  EXCLUDE_ARCH='aarch64' ;;
  --platform=arm64)  EXCLUDE_ARCH='x86_64'  ;;
  *) echo "Usage: $0 --platform=amd64|arm64"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/xinference/model"

# Step 1: find all model JSON files
FILES=$(grep -rl '"virtualenv"' "$MODEL_DIR" --include="*.json" | sort || true)

if [ -z "$FILES" ]; then
  echo "Error: No model JSON files found in $MODEL_DIR" >&2
  exit 1
fi

# Step 2: extract all virtualenv.packages[] entries, dedup
ALL=$(jq -r '
  [ ..
    | select(.virtualenv? | type == "object")
    | .virtualenv.packages[]?
  ] | unique[]
' $FILES)

# Step 3: filter by platform, exclude opposite arch entries
# Step 4: strip conditional markers
PACKAGES=$(echo "$ALL" | grep -v "platform_machine == \"$EXCLUDE_ARCH\"" | sed 's/ ; .*//')

# Step 5: expand placeholders to real pip-installable specs
echo "$PACKAGES" | while IFS= read -r pkg; do
  case "$pkg" in
    '#system_numpy#')     printf 'numpy\n' ;;
    '#system_torch#')     printf 'torch\n' ;;
    '#system_torchaudio#') printf 'torchaudio\n' ;;
    '#system_torchvision#') printf 'torchvision\n' ;;
    '#system_pandas#')    printf 'pandas\n' ;;

    '#transformers_dependencies#'|transformers_dependencies)
      printf 'transformers>=4.53.3\naccelerate>=0.28.0\n' ;;
    '#vllm_dependencies#'|vllm_dependencies)
      printf 'vllm>=0.11.2\n' ;;
    '#sglang_dependencies#'|sglang_dependencies)
      printf 'pybase64\nzmq\npartial_json_parser\nsentencepiece\ndill\nninja\nnumpy>=2.4.1\nsglang>=0.5.6\nsgl_kernel\n' ;;
    '#sentence_transformers_dependencies#'|sentence_transformers_dependencies)
      printf 'sentence_transformers\neinops\ntransformers>=4.53.3\naccelerate>=0.28.0\n' ;;
    '#diffusers_dependencies#'|diffusers_dependencies)
      printf 'diffusers>=0.32.0\nhuggingface-hub<1.0\n' ;;
    '#mlx_dependencies#'|mlx_dependencies)
      printf 'mlx-lm>=0.24.0\n' ;;
    '#llama_cpp_dependencies#'|llama_cpp_dependencies)
      printf 'xllamacpp>=0.2.6\n' ;;

    *) printf '%s\n' "$pkg" ;;
  esac
done | sort -u
