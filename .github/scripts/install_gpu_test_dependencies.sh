#!/usr/bin/env bash
# Install only the dependencies needed by the selected GPU test families.
set -euo pipefail

groups=${GPU_TEST_GROUPS:?GPU_TEST_GROUPS must name at least one GPU test group}
selected=()
for group in $groups; do
  case "$group" in
    llm|embedding|image|audio) selected+=("$group") ;;
    *) echo "Unknown GPU test group: $group" >&2; exit 1 ;;
  esac
done
if [ ${#selected[@]} -eq 0 ]; then
  echo 'At least one GPU test group is required' >&2
  exit 1
fi
groups="${selected[*]}"

has_group() {
  [[ " $groups " == *" $1 "* ]]
}

export PIP_CONSTRAINT="$PWD/.github/requirements/model-ci-constraints.txt"
extras=dev
packages=(
  'torch==2.7.0' 'torchvision==0.22.0' 'numpy>=2'
  transformers accelerate hf_transfer gguf 'openai>1' 'passlib[bcrypt]' pynvml
)

if has_group llm; then
  packages+=(tensorizer sentencepiece tiktoken einops protobuf transformers_stream_generator)
fi
if has_group embedding; then
  packages+=('sentence-transformers==5.1.1' 'FlagEmbedding==1.3.5' onnx onnxconverter_common)
fi
if has_group image; then
  packages+=('diffusers==0.35.1' sentencepiece protobuf)
fi
if has_group audio; then
  # Preserve the existing audio extra and its native dependencies. Splitting
  # audio into individual engines needs separate model-runtime validation.
  extras=audio,dev
  conda install -y -c conda-forge 'pynini==2.1.5' 'ffmpeg<7'
  packages+=(
    'torchaudio==2.7.0' 'funasr==1.2.7' 'nemo_text_processing<1.1.0'
    'WeTextProcessing<1.0.4' 'omegaconf~=2.3.0' librosa xxhash 'ChatTTS==0.2.4'
    HyperPyYAML 'onnxruntime-gpu==1.22.0; sys_platform == "linux"' openai-whisper
    loguru natsort loralib ormsgpack verovio cachetools silero-vad pydantic
    'diffusers==0.35.1' onnx onnxconverter_common torchdiffeq
    'x_transformers>=1.31.14' pypinyin tomli vocos jieba soundfile
  )
fi

# Resolve the selected environment in one transaction. The constraint file
# also applies to transitive dependencies, without installing unused families.
python -m pip install -U -e ".[${extras}]" "${packages[@]}"

if has_group embedding; then
  python -m pip install 'xllamacpp>=0.2.0' \
    --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124 \
    --extra-index-url https://pypi.org/simple
fi
if has_group audio; then
  python -m pip uninstall -y matcha-tts opencc faster_whisper
  sudo apt-get update -qq
  sudo apt-get install -y -qq espeak-ng
fi

# Fail during setup if an installed family's import stack is incompatible.
python -c 'from transformers import AutoModel; from peft import PeftModel'
if has_group llm; then
  python -c 'import tensorizer'
fi
if has_group embedding; then
  python -c 'from datasets import Dataset; from sentence_transformers import SentenceTransformer, CrossEncoder; from FlagEmbedding import BGEM3FlagModel; import xllamacpp'
fi
if has_group image || has_group audio; then
  python -c 'from diffusers import DiffusionPipeline'
fi
