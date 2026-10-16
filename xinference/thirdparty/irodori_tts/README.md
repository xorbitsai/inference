# Irodori-TTS

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Space-blue)](https://huggingface.co/spaces/Aratako/Irodori-TTS-v4.1-Small-Demo)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)

Training and inference code for **Irodori-TTS**, a Flow Matching-based Text-to-Speech model. The architecture and training design largely follow [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/), using [DACVAE](https://github.com/facebookresearch/dacvae) continuous latents as the generation target.

For an OpenAI-compatible inference API server, see [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server).

> [!IMPORTANT]
> `main` tracks the **v4** codebase and is intended for use with the unified **Irodori-TTS-v4.1-Small** release.
> The current code remains backward-compatible with the released v2/v3 base and VoiceDesign checkpoints.
> Previous codebase states are available through the `v3`, `v2`, and `v1` tags.
> v1 checkpoints / preprocessing are not compatible with v2/v3/v4.

For model weights and audio samples, please refer to the [Irodori-TTS-v4.1-Small model card](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small).

## Features

- **Flow Matching TTS**: Rectified Flow Diffusion Transformer (RF-DiT) over continuous DACVAE latents
- **Voice Cloning**: Zero-shot voice cloning from reference audio
- **Multi-modal Voice Design**: v4-Small combines text, reference speech, and caption text for voice identity plus style/emotion control
- **Long Reference Audio**: One or more reference clips can be concatenated up to the checkpoint's 120-second limit
- **Emoji-based Style Control**: Emoji annotations in input text can influence delivery and non-verbal vocal expressions in supported checkpoints
- **Automatic Duration Prediction**: v4-Small estimates output length without manual `--seconds`
- **Automatic Watermarking**: Generated audio is watermarked with [SilentCipher](https://github.com/sony/silentcipher) when available
- **Multi-GPU Training**: Distributed training via `uv run --no-sync torchrun` with gradient accumulation, mixed precision (bf16), and W&B logging
- **PEFT LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT/LoRA for released checkpoints
- **Speaker Inversion**: Learn reusable speaker embedding tokens for a target voice while freezing the base model
- **Flexible Inference**: CLI, Gradio Web UI, and HuggingFace Hub checkpoint support

## Architecture

The current release, **`Aratako/Irodori-TTS-v4.1-Small`**, unifies the previous base and
VoiceDesign families in one checkpoint. It supports 3-branch conditioning from text,
reference speech, and caption text. Released v2/v3 checkpoints remain supported for inference.

Shared building blocks:

1. **Shared Text/Caption Encoder**: A fine-tuned ModernBERT backbone processes both reading text and caption text
2. **Reference Latent Encoder**: Encodes patched reference audio latents for speaker identity conditioning, with up to 120 seconds of combined reference audio in v4-Small
3. **Condition Projectors**: Separate text and caption projectors map the shared encoder states into their conditioning spaces
4. **Diffusion Transformer**: Joint-attention DiT blocks with Low-Rank AdaLN (timestep-conditioned adaptive layer normalization), half-RoPE, and SwiGLU MLPs
5. **Duration Predictor**: Integrated predictor for automatic output length estimation

Audio is represented as continuous latent sequences via the codec configured by the checkpoint. The released v2/v3/v4 checkpoints use the 32-dim [Semantic-DACVAE-Japanese-32dim](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) codec for 48kHz waveform reconstruction.

## Installation

```bash
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync --extra cu128  # NVIDIA CUDA 12.8 (Linux/Windows)
```

If you want to explicitly select a PyTorch backend, use one of the backend
extras below:

```bash
# NVIDIA CUDA 12.8 on Linux/Windows
uv sync --extra cu128

# AMD ROCm on Linux/WSL
uv sync --extra rocm

# Intel XPU on Linux/Windows
uv sync --extra xpu

# CPU-only, or macOS CPU/MPS via PyPI
uv sync --extra cpu
```

The PyTorch backend extras are mutually exclusive. The `cu128` extra uses the
PyTorch CUDA 12.8 index, the `rocm` extra uses the PyTorch ROCm index on
Linux, and the `xpu` extra uses the PyTorch XPU index on Linux/Windows.
The `cpu` extra uses the CPU PyTorch index on Linux/Windows and falls
back to the standard PyPI PyTorch wheels on macOS.

After syncing with a backend extra, use `uv run --no-sync ...` for the commands
below to avoid re-syncing the environment without the selected PyTorch backend
extra.

The `rocm` extra includes `pytorch-triton-rocm` because `triton-rocm` alone does
not provide `triton.language` for the `transformers` to `torch._dynamo` import
path. This was validated with AMD GPU inference.

## Quick Start

### Simple Inference

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

### Inference without Reference Audio

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --no-ref \
  --output-wav outputs/sample.wav
```

### VoiceDesign Inference

Pure VoiceDesign from text + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

Style-controlled voice cloning with text + reference speech + caption:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --ref-wav path/to/reference.wav \
  --caption "深く傷つき、今にも泣き出しそうな様子。声が震えており、悲痛なトーンで弱々しく話す。" \
  --output-wav outputs/sample_voice_design_clone.wav
```

Long-reference checkpoints can concatenate multiple reference clips in the specified order:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "複数の参照音声を使って合成します。" \
  --caption "落ち着いた自然な声" \
  --ref-wavs ref_01.wav ref_02.wav ref_03.wav \
  --output-wav outputs/sample_long_reference.wav
```

Each waveform is encoded independently before its latent is concatenated. The combined
reference is trimmed to the checkpoint's maximum reference duration. Use `--ref-latents`
in the same way for precomputed latent files.

For v4-Small, prefer multiple clean, shorter clips from the same speaker when using a long
reference. The model was trained with randomly concatenated short utterances, and the measured
speaker-similarity benefit used the same construction. A combined duration of approximately
30 seconds already captured most of the measured gain. A single uninterrupted long recording
is accepted by inference, but that input format has not been evaluated and may behave differently.

### Speaker Inversion Inference

Use a learned Speaker Inversion embedding instead of reference audio:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed path/to/my.speaker.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Gradio Web UI

```bash
uv run --no-sync python gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

Then access the UI at `http://localhost:7860`.
The hosted v4-Small demo is available at [Aratako/Irodori-TTS-v4.1-Small-Demo](https://huggingface.co/spaces/Aratako/Irodori-TTS-v4.1-Small-Demo).
The reference input area accepts one or more audio files, which can be reordered before
generation and are concatenated in the displayed order. For long-reference cloning, upload
multiple clean, shorter clips from the same speaker; this matches v4-Small training. A single
uninterrupted long recording is accepted but has not been evaluated. The standard UI also
supports a Speaker Inversion embedding through the adjacent tab.

For VoiceDesign checkpoints, use the dedicated UI:

```bash
uv run --no-sync python gradio_app_voicedesign.py --server-name 0.0.0.0 --server-port 7861
```

The same hosted v4-Small demo supports VoiceDesign and reference-audio conditioning.

Both UIs default to `Aratako/Irodori-TTS-v4.1-Small`. `gradio_app_voicedesign.py` exposes
caption conditioning, while `gradio_app.py` includes the Speaker Inversion input.

## Inference

### CLI

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

Local checkpoints (`.pt` or `.safetensors`) are also supported:

```bash
uv run --no-sync python infer.py \
  --checkpoint outputs/checkpoint_final.safetensors \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample.wav
```

v4-Small supports caption conditioning. It can run with
caption only by passing `--no-ref`, or with both reference speech and caption by passing
`--ref-wav`, `--ref-wavs`, `--ref-latent`, `--ref-latents`, or `--ref-embed`.

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --caption "落ち着いた、近い距離感の女性話者" \
  --no-ref \
  --output-wav outputs/sample_voice_design.wav
```

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "あははっ🤭、それ本当に言ってるの？…😮‍💨まぁ、君らしいけどね。" \
  --caption "余裕のある大人の男性。親しい相手に対して、くだけた雰囲気で呆れながらも楽しそうに話している。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_voice_design_ref_caption.wav
```

The older `Aratako/Irodori-TTS-500M-v2-VoiceDesign` checkpoint is still supported, but it is caption-only and intentionally ignores speaker/reference conditioning.

LoRA adapter directories can be loaded dynamically at inference time without
exporting a merged checkpoint:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/base_model.safetensors \
  --lora-adapter outputs/irodori_tts_lora/checkpoint_final \
  --text "こんにちは、私はAIです。これはLoRA推論のテストです。" \
  --ref-wav path/to/reference.wav \
  --output-wav outputs/sample_lora.wav
```

Speaker Inversion embedding checkpoints can be used with the same base model that
was used for inversion training. Pass the embedding with `--ref-embed`;
it is mutually exclusive with `--ref-wav`, `--ref-latent`, and `--no-ref`.

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、私はAIです。これはSpeaker Inversion推論のテストです。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

### Output Duration

v4-Small integrates duration prediction into inference.
When `--seconds` is omitted, the runtime estimates the output length from the input
text and enabled conditions, then generates audio for that estimated duration. Use
`--duration-scale` to multiply the predicted length (`>1` longer, `<1` shorter). For
exact control, pass `--seconds` manually.

Older v2 checkpoints were trained with fixed-length 30-second targets. They remain
supported by the current codebase and still accept manual `--seconds`, but forcing a
non-default duration can reduce audio quality; prefer v4-Small for automatic
or scaled duration control.

### Sway Sampling

For faster experimental inference, Sway Sampling can be combined with fewer Euler
steps:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small \
  --text "こんにちは、私はAIです。これは音声合成のテストです。" \
  --ref-wav path/to/reference.wav \
  --num-steps 6 \
  --t-schedule-mode sway \
  --sway-coeff -1.0 \
  --output-wav outputs/sample_sway.wav
```

### Additional Inference Notes

For tuning guidance and detailed explanations of inference options, see the
[Parameter Guide](docs/parameters.md).

Generated audio is passed through [SilentCipher](https://github.com/sony/silentcipher) watermarking automatically when the dependency and model files are available.

## Training

This section describes how to train **Irodori-TTS-v4.1-Small**. For training instructions
for previous models, refer to the documentation in the corresponding version tags.

### 1. Prepare the Training Manifest

Encodes audio from a Hugging Face dataset into DACVAE latents and produces a JSONL manifest for training.

```bash
uv run --no-sync python prepare_manifest.py \
  --dataset myorg/my_dataset \
  --split train \
  --audio-column audio \
  --text-column text \
  --caption-column caption \
  --speaker-column speaker \
  --output-manifest data/train_manifest.jsonl \
  --latent-dir data/latents \
  --device cuda
```

v4-Small learns from text, speaker/reference audio, and captions. Include `speaker_id` and
`caption` where available so all three conditioning paths can be trained. A Speaker
Inversion manifest does not require `speaker_id`, because the run learns one shared speaker
embedding from the target-speaker samples.

The manifest `caption` value may also be a list of strings; training randomly selects one
non-empty caption each time that row is loaded.

This produces a JSONL manifest with entries like:

```json
{"text": "こんにちは", "caption": "落ち着いた、近い距離感の女性話者", "latent_path": "data/latents/00001.pt", "speaker_id": "myorg/my_dataset:speaker_001", "num_frames": 750}
```

### 2. Train v4-Small

Single-GPU training:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors
```

The v4-Small config trains the RF body, duration predictor, and shared pretrained text/caption
backbone jointly. The duration predictor regresses `log1p(num_frames)` with Huber loss and
uses the token-sum architecture selected from ablations. See the parameter guide for its
architecture details.

Multi-GPU DDP training:

```bash
uv run --no-sync torchrun --nproc_per_node 4 train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --device cuda
```

Training supports YAML config files with `model` and `train` sections. CLI arguments take precedence over YAML values. See `uv run --no-sync python train.py --help` for all available options.
For a more detailed explanation of model and training config fields, see [Parameter Guide](docs/parameters.md).

### 3. LoRA Fine-Tuning

Start a new training run from released inference weights (`.safetensors`). This initializes only the model weights; optimizer / scheduler state starts fresh. The duration predictor is kept as part of the saved adapter by default.

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors
```

The v4-Small LoRA config targets diffusion attention by default and saves the duration
predictor with the adapter. To adapt the shared ModernBERT backbone, select the
`pretrained_backbone_attn` or `pretrained_backbone_attn_mlp` target preset.

LoRA target presets, adapter saving behavior, and resume details are covered in the
[Parameter Guide](docs/parameters.md).

### 4. Speaker Inversion

Speaker Inversion trains only a small set of speaker embedding tokens while keeping the
base Irodori-TTS model frozen. It is useful when you want a reusable speaker identity
checkpoint instead of providing reference audio at every inference call.

Prepare a manifest from the target speaker's audio, then initialize from v4-Small:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_speaker_inversion.yaml \
  --manifest data/target_speaker_manifest.jsonl \
  --init-checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --output-dir outputs/speaker_inversion/name
```

The saved checkpoints are embedding-only `.speaker.safetensors` files, for example
`outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors`. Use that file
with the base model during inference:

```bash
uv run --no-sync python infer.py \
  --checkpoint path/to/Irodori-TTS-v4.1-Small/model.safetensors \
  --ref-embed outputs/speaker_inversion/name/checkpoint_final.speaker.safetensors \
  --text "こんにちは、これは学習した話者埋め込みを使った推論です。" \
  --output-wav outputs/sample_speaker_inversion.wav
```

To continue from a saved embedding, set `speaker_inversion_init_embedding` in the
config or pass `--speaker-inversion-init-embedding path/to/checkpoint.speaker.safetensors`.
Full trainer `--resume` is intentionally not used for Speaker Inversion checkpoints.
Enable `gradient_checkpointing: true` or pass `--gradient-checkpointing` if GPU memory is tight.

### 5. Resume Interrupted Training

Resume an existing training run from a training checkpoint. Full-model runs use `.pt`; LoRA runs use checkpoint directories. Both restore optimizer, scheduler, and step state.

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts \
  --resume outputs/irodori_tts/checkpoint_0010000.pt
```

LoRA resume example:

```bash
uv run --no-sync python train.py \
  --config configs/train_v4_small_lora.yaml \
  --manifest data/train_manifest.jsonl \
  --output-dir outputs/irodori_tts_lora \
  --resume outputs/irodori_tts_lora/checkpoint_0010000
```

If you move a LoRA checkpoint to another environment and the original base-checkpoint path is no longer valid, pass `--init-checkpoint path/to/base_model.safetensors` together with `--resume` to override the saved base-model path.

### 6. Convert a Training Checkpoint

Convert a training checkpoint to inference-only safetensors format:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/checkpoint_final.pt
```

LoRA adapter checkpoints can also be converted directly:

```bash
uv run --no-sync python convert_checkpoint_to_safetensors.py outputs/irodori_tts_lora/checkpoint_final
```

LoRA adapter checkpoints are merged into the base model automatically during conversion, so the exported `.safetensors` file is directly usable for inference. If you do not want to merge the adapter, pass the adapter directory directly to `infer.py --lora-adapter` or the matching Gradio field.

For checkpoints with a pretrained text encoder, conversion also writes a `tokenizer/`
directory beside the safetensors file and embeds the encoder architecture config in the file.
Keep the safetensors file and `tokenizer/` directory together when publishing or moving the model.

## Quantization

Quantized variants of Irodori-TTS reduce the memory required by the TTS model during
inference. Pre-quantized v4-Small checkpoints are available from
[Aratako/Irodori-TTS-v4.1-Small-Quantized](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-Quantized).
Select a variant by appending its subdirectory name to the Hugging Face repository ID:

```bash
uv run --no-sync python infer.py \
  --hf-checkpoint Aratako/Irodori-TTS-v4.1-Small-Quantized/int8-weight-only \
  --model-precision bf16 \
  --text "こんにちは、私はAIです。" \
  --no-ref \
  --output-wav outputs/sample_int8.wav
```

Available schemes are `int8-weight-only` (W8A16), `int8-dynamic` (W8A8),
`int4-weight-only` (W4A16, group size 128 by default), `float8-weight-only` (FP8 weights
and BF16 activations), and `float8-dynamic` (FP8 weights and activations).
INT4 weight-only uses the CUDA tinygemm kernel and requires compute capability 8.0 or newer.
Only the selected model variant and its tokenizer assets are downloaded.

`--model-precision bf16` controls the unquantized layers and floating-point activations;
quantized weights retain their stored quantization format.

To quantize another compatible inference checkpoint locally, use
`quantize_checkpoint.py`. INT8 weight-only is the default:

```bash
uv run --no-sync python quantize_checkpoint.py path/to/model.safetensors \
  --quantization int8-weight-only \
  --output path/to/quantized/model.safetensors
```

The default `core` profile quantizes the attention and MLP weights in the text,
speaker, and diffusion Transformer blocks. Projectors, AdaLN, duration prediction,
and the codec remain unquantized. `--profile all-linear` is available for more aggressive
experimentation.

Dynamic `--lora-adapter` inference is supported with quantized base checkpoints.
Train the adapter against the matching full-precision base model.

## Project Structure

```text
Irodori-TTS/
├── train.py                    # Training entry point (DDP support)
├── infer.py                    # CLI inference
├── gradio_app.py               # Gradio web UI
├── gradio_app_voicedesign.py   # Gradio web UI for VoiceDesign checkpoints
├── prepare_manifest.py         # Dataset -> DACVAE latent preprocessing
├── convert_checkpoint_to_safetensors.py  # Checkpoint converter
├── quantize_checkpoint.py      # torchao checkpoint quantization
│
├── docs/
│   └── parameters.md         # Detailed parameter guide
│
├── irodori_tts/                # Core library
│   ├── model.py                # TextToLatentRFDiT architecture
│   ├── rf.py                   # Rectified Flow utilities & Euler CFG sampling
│   ├── codec.py                # DACVAE codec wrapper
│   ├── dataset.py              # Dataset and collator
│   ├── tokenizer.py            # Pretrained LLM tokenizer wrapper
│   ├── config.py               # Model and training config dataclasses
│   ├── inference_runtime.py    # Cached, thread-safe inference runtime
│   ├── lora.py                 # PEFT LoRA integration helpers
│   ├── quantization.py         # torchao checkpoint serialization/load helpers
│   ├── speaker_inversion.py    # Speaker Inversion embedding save/load helpers
│   ├── text_normalization.py   # Japanese text normalization
│   ├── optim.py                # Muon + AdamW optimizer
│   └── progress.py             # Training progress tracker
│
└── configs/
    ├── train_v4_small.yaml                    # Irodori-TTS-v4-Small training config
    ├── train_v4_small_lora.yaml               # v4-Small LoRA fine-tuning config
    ├── train_v4_small_speaker_inversion.yaml  # v4-Small Speaker Inversion config
    ├── train_500m_v3_phase1_body.yaml        # 500M v3 body training config
    ├── train_500m_v3_phase2_duration.yaml    # 500M v3 duration-predictor training config
    ├── train_500m_v3_voice_design_phase1_body.yaml     # 600M v3 VoiceDesign body config
    ├── train_500m_v3_voice_design_phase2_duration.yaml # 600M v3 VoiceDesign duration config
    ├── train_500m_v3_voice_design_lora.yaml            # 600M v3 VoiceDesign RF+duration LoRA config
    ├── train_500m_v3_lora.yaml               # 500M v3 LoRA fine-tuning config
    ├── train_500m_v3_speaker_inversion.yaml  # 500M v3 Speaker Inversion config
    ├── train_500m_v2.yaml                    # 500M v2 backward-compatible model config
    ├── train_500m_v2_lora.yaml               # 500M v2 LoRA fine-tuning config
    ├── train_500m_v2_voice_design.yaml       # 500M v2 VoiceDesign full fine-tuning config
    ├── train_500m_v2_voice_design_lora.yaml  # 500M v2 VoiceDesign LoRA fine-tuning config
    ├── train_500m.yaml                       # 500M v1 model config
    └── train_2.5b.yaml                       # 2.5B parameter model config
```

## License

- **Code**: [MIT License](LICENSE)
- **Model Weights**: Please refer to the [Irodori-TTS-v4.1-Small model card](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) for licensing details

## Acknowledgments

This project builds upon the following works:

- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/) — Architecture and training design reference
- [DACVAE](https://github.com/facebookresearch/dacvae) — Audio VAE
- [SilentCipher](https://github.com/sony/silentcipher) — Audio watermarking

## Citation

```bibtex
@misc{irodori-tts,
  author = {Chihiro Arata},
  title = {Irodori-TTS: A Flow Matching-based Text-to-Speech Model with Emoji-driven Style Control},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Aratako/Irodori-TTS}}
}
```
