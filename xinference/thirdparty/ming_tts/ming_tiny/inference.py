"""Reusable single-GPU Ming-3.0-tts-tiny inference engine."""

from __future__ import annotations

import os
import random
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchaudio
from packaging.version import Version
import transformers

from .configuration_lingv3_tts import LingV3TTSConfig
from .prompt_contract import (
    BASE_PROTOCOL,
    ensure_base_tokenizer,
    prompt_format,
    protocol_version,
    decode_alignment,
    sampling_contract,
    task_decode_alignments,
    tokenizer_contract,
    with_rml,
)
from .spkemb_extractor import SpkembExtractor


DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

PROMPT_LATENT_CONDITIONING_MODES = {
    "prompt_latent",
    "prompt_latent_no_rml",
}
NO_RML_CONDITIONING_MODES = {"no_rml", "prompt_latent_no_rml"}
ZERO_SPEAKER_CONDITIONING_MODE = "zero_speaker"
MASKED_SPEAKER_CONDITIONING_MODE = "masked_speaker"
MULTI_SPEAKER_CONDITIONING_MODE = "multi_speaker"
CONDITIONING_MODES = {
    "speaker_only",
    MULTI_SPEAKER_CONDITIONING_MODE,
    ZERO_SPEAKER_CONDITIONING_MODE,
    MASKED_SPEAKER_CONDITIONING_MODE,
    *PROMPT_LATENT_CONDITIONING_MODES,
    *NO_RML_CONDITIONING_MODES,
}
INVALID_BUNDLE_MARKER = "INVALID_INFERENCE_BUNDLE.json"

PARITY_RUNTIME = {
    "torch": "2.4.0+cu124",
    "transformers": "4.46.3",
    "fla-core": "0.4.1",
    "flash-attn": "2.3.6+pack.glm.mask",
    "atorch": "1.6.0",
    "megablocks": "0.5.1",
    "grouped_gemm": "0.1.4",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt_waveform(path: str, target_sample_rate: int = 16000):
    # Match infer_lingv3_tts.py exactly.  The dispatcher default and
    # functional resampler can differ by a few float32 ulps from this path;
    # that difference can survive CampPlus and perturb an autoregressive run.
    waveform, sample_rate = torchaudio.load(path, backend="soundfile")
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected [channels, samples] waveform, got {waveform.shape}"
        )
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )(waveform)
    return waveform


def prompt_conditioning_geometry(
    config: LingV3TTSConfig,
) -> dict[str, int]:
    """Validate the AudioVAE/LLM/Flow units used by zero-shot prompting."""
    tts_config = config.tts_config
    audio_config = config.audio_config
    encoder_config = dict(audio_config.enc_kwargs)
    decoder_config = dict(audio_config.dec_kwargs)
    aggregator_config = dict(config.aggregator_config)
    flow_config = dict(config.flow_config)

    sample_rate = int(tts_config["sample_rate"])
    audio_sample_rate = int(audio_config.sample_rate)
    input_samples = int(encoder_config["input_dim"])
    hop_samples = int(encoder_config.get("hop_size", 320))
    audio_patch_size = int(audio_config.patch_size)
    flow_patch_size = int(tts_config["patch_size"])
    latent_dim = int(tts_config["latent_dim"])
    latent_dims = {
        "tts": latent_dim,
        "audio_encoder": int(encoder_config["latent_dim"]),
        "audio_decoder": int(decoder_config["latent_dim"]),
        "aggregator": int(
            aggregator_config.get("in_channels", latent_dim)
        ),
        "flow": int(flow_config.get("z_channels", latent_dim)),
    }
    if len(set(latent_dims.values())) != 1:
        raise RuntimeError(
            "Prompt-conditioning latent dimensions disagree: "
            f"{latent_dims}"
        )
    if sample_rate != audio_sample_rate:
        raise RuntimeError(
            "TTS and AudioVAE sample rates disagree: "
            f"{sample_rate} != {audio_sample_rate}"
        )
    if input_samples != hop_samples:
        raise RuntimeError(
            "LingV2-compatible prompt alignment requires AudioVAE "
            f"input_dim == hop_size, got {input_samples} != {hop_samples}"
        )
    if audio_patch_size < 1 or flow_patch_size < 1:
        raise RuntimeError(
            "AudioVAE and Flow patch sizes must be positive, got "
            f"{audio_patch_size} and {flow_patch_size}"
        )
    samples_per_audio_latent = hop_samples * audio_patch_size
    native_12_5hz_samples = int(sample_rate / 12.5)
    if samples_per_audio_latent != native_12_5hz_samples:
        raise RuntimeError(
            "AudioVAE geometry does not match the native 12.5 Hz prompt "
            f"contract: {samples_per_audio_latent} != "
            f"{native_12_5hz_samples} samples"
        )
    return {
        "sample_rate": sample_rate,
        "latent_dim": latent_dim,
        "audio_patch_size": audio_patch_size,
        "flow_patch_size": flow_patch_size,
        "samples_per_audio_latent": samples_per_audio_latent,
        "pad_align_samples": (
            samples_per_audio_latent * flow_patch_size
        ),
    }


def _require_parity_runtime() -> None:
    actual = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    for package in (
        "fla-core",
        "flash-attn",
        "atorch",
        "megablocks",
        "grouped_gemm",
    ):
        try:
            actual[package] = version(package)
        except PackageNotFoundError:
            actual[package] = "not installed"
    mismatches = [
        f"{package}: expected {expected}, found {actual[package]}"
        for package, expected in PARITY_RUNTIME.items()
        if actual[package] != expected
    ]
    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            f"{BASE_PROTOCOL} parity mode requires the native reference "
            "runtime. Version mismatches:\n  - "
            f"{details}\nInstall the optional pins from requirements.txt, including "
            "Torch/CUDA-compatible atorch, MegaBlocks, and grouped_gemm."
        )


def _require_parity_model_contract(config: LingV3TTSConfig) -> None:
    expected = {
        "routed_scaling_factor": 1.0,
        "max_position_embeddings": 8192,
        "num_nextn_predict_layers": 0,
        "mtp_loss_scaling_factor": 0,
    }
    actual = {
        key: getattr(config.llm_config, key, None) for key in expected
    }
    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {wanted!r}, found {found!r}"
            for key, (wanted, found) in mismatches.items()
        )
        raise RuntimeError(
            f"{BASE_PROTOCOL} model config is not the reference contract. "
            "Reconvert with the reference config; "
            f"{details}"
        )


class MingTinyEngine:
    def __init__(
        self,
        model_dir: str,
        *,
        tokenizer_dir: Optional[str] = None,
        device: str = "cuda:0",
        dtype: str = "bf16",
        campplus_path: Optional[str] = None,
        moe_backend: str = "auto",
        runtime_state_dtype: str = "auto",
    ) -> None:
        transformers_version = Version(transformers.__version__)
        if transformers_version != Version("4.46.3"):
            raise RuntimeError(
                "Ming-3.0-tts-tiny requires transformers==4.46.3; "
                f"found {transformers.__version__}."
            )
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        from .modeling_lingv3_tts import (
            LingV3TTSForConditionalGeneration,
        )

        if dtype not in DTYPES:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; choose one of {sorted(DTYPES)}"
            )
        if moe_backend not in {"auto", "loop", "atorch"}:
            raise ValueError(
                "moe_backend must be 'auto', 'loop', or 'atorch', got "
                f"{moe_backend!r}"
            )
        if runtime_state_dtype not in {
            "auto",
            "stable_fp32",
            "native_bf16",
        }:
            raise ValueError(
                "runtime_state_dtype must be 'auto', 'stable_fp32', or "
                f"'native_bf16', got {runtime_state_dtype!r}"
            )
        self.model_dir = os.path.abspath(model_dir)
        self.tokenizer_dir = os.path.abspath(tokenizer_dir or model_dir)
        invalid_marker = Path(self.model_dir) / INVALID_BUNDLE_MARKER
        if invalid_marker.is_file():
            try:
                invalid = json.loads(invalid_marker.read_text(encoding="utf-8"))
                reason = invalid.get("reason", "invalidated inference bundle")
                replacement = invalid.get("replacement")
            except (OSError, json.JSONDecodeError):
                reason = "invalidated inference bundle"
                replacement = None
            hint = f"; use {replacement}" if replacement else ""
            raise RuntimeError(f"{self.model_dir}: {reason}{hint}")
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]

        print(
            f"[Ming-3.0-tts-tiny] load config from {self.model_dir}; "
            f"tokenizer from {self.tokenizer_dir}",
            flush=True,
        )
        self.config = LingV3TTSConfig.from_pretrained(self.model_dir)
        self.with_rml = with_rml(self.config.tts_config)
        self.prompt_geometry = prompt_conditioning_geometry(self.config)
        self.protocol_version = protocol_version(self.config.tts_config)
        self.prompt_format = prompt_format(self.config.tts_config)
        self.tokenizer_contract = tokenizer_contract(
            self.config.tts_config
        )
        self.sampling_contract = sampling_contract(
            self.config.tts_config
        )
        self.enable_tts_target_language = self.config.tts_config.get(
            "enable_tts_target_language", False
        )
        self.infer_tts_language_from_text = self.config.tts_config.get(
            "infer_tts_language_from_text", False
        )
        self.decode_alignment = decode_alignment(self.config.tts_config)
        self.task_decode_alignment = task_decode_alignments(
            self.config.tts_config
        )
        self.task_contract = self.config.tts_config.get("task_contract")
        is_base_protocol = (
            self.protocol_version == BASE_PROTOCOL
        )
        if moe_backend == "auto":
            from .modeling_bailing_moe_v3 import _ATORCH_MOE_AVAILABLE

            # The fused path is the reference trajectory, but it is not
            # installable everywhere; fall back rather than refuse to run.
            self.moe_backend = (
                "atorch"
                if is_base_protocol and _ATORCH_MOE_AVAILABLE
                else "loop"
            )
            if is_base_protocol and not _ATORCH_MOE_AVAILABLE:
                print(
                    "[Ming-3.0-tts-tiny] WARNING: atorch is not installed, "
                    "falling back to the portable per-expert MoE loop. It is "
                    "slower and not numerically identical to the trajectory "
                    "this checkpoint was validated on.",
                    flush=True,
                )
        else:
            self.moe_backend = moe_backend
        self.runtime_state_dtype = (
            "native_bf16" if is_base_protocol else "stable_fp32"
        ) if runtime_state_dtype == "auto" else runtime_state_dtype
        if not is_base_protocol and (
            self.moe_backend == "atorch"
            or self.runtime_state_dtype == "native_bf16"
        ):
            print(
                "[Ming-3.0-tts-tiny] WARNING: thinking_v1/Megatron is using the "
                "legacy-oriented atorch/native_bf16 numerical path. This "
                "changes the autoregressive trajectory; use "
                "--moe_backend auto --runtime_state_dtype auto for the "
                "training-aligned loop/stable_fp32 path.",
                flush=True,
            )
        if is_base_protocol:
            _require_parity_model_contract(self.config)
        if is_base_protocol and self.moe_backend == "atorch":
            _require_parity_runtime()
            audio_vae_attention = os.environ.get(
                "MING_TINY_AUDIO_VAE_ATTN", "config"
            ).strip().lower()
            if audio_vae_attention not in {
                "config",
                "flash_attention_2",
            }:
                raise RuntimeError(
                    f"{BASE_PROTOCOL} parity requires the AudioVAE checkpoint's "
                    "flash_attention_2 backend; found "
                    "MING_TINY_AUDIO_VAE_ATTN="
                    f"{audio_vae_attention!r}."
                )
        elif is_base_protocol:
            print(
                f"[Ming-3.0-tts-tiny] WARNING: {BASE_PROTOCOL} is using the portable "
                "MoE loop. This diagnostic path is not numerically equivalent "
                "to the validated atorch trajectory.",
                flush=True,
            )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_dir,
                trust_remote_code=True,
                use_fast=True,
            )
        except ValueError as error:
            # The training-side config_vl export names BailingTokenizer but
            # does not ship tokenization_bailing.py/auto_map.  Its backend is
            # a complete tokenizer.json, so the generic fast wrapper is the
            # exact portable implementation.
            if "BailingTokenizer" not in str(error):
                raise
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.tokenizer_dir
            )
            print(
                "[Ming-3.0-tts-tiny] tokenizer export has no BailingTokenizer "
                "Python class; loaded its tokenizer.json with the generic "
                "fast wrapper",
                flush=True,
            )
        if self.tokenizer_contract == BASE_PROTOCOL:
            added = ensure_base_tokenizer(self.tokenizer)
            if added:
                print(
                    "[Ming-3.0-tts-tiny] reconstructed native multimodal tokenizer: "
                    f"added={added}, size={len(self.tokenizer)}",
                    flush=True,
                )
        print(
            f"[Ming-3.0-tts-tiny] load 9B weights as {self.dtype} on CPU",
            flush=True,
        )
        self.model = LingV3TTSForConditionalGeneration.from_pretrained(
            self.model_dir,
            config=self.config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        if self.moe_backend == "atorch":
            packed_layers = self.model.enable_native_atorch_moe()
            if packed_layers != 23:
                raise RuntimeError(
                    f"Expected 23 native MoE layers, packed {packed_layers}"
                )
        if self.runtime_state_dtype == "native_bf16":
            self.model.eval().to(self.device, dtype=self.dtype)
        else:
            self.model.eval().to(self.device)
        self.model.validate_tts_tokenizer(self.tokenizer)
        dtype_state = self.model.validate_runtime_dtypes(
            native_bf16_state=(
                self.runtime_state_dtype == "native_bf16"
            )
        )
        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        print(
            "[Ming-3.0-tts-tiny] model ready: "
            f"device={self.device}, parameters={parameter_count:,}, "
            f"protocol={self.protocol_version}, "
            f"prompt_format={self.prompt_format}, "
            f"tokenizer_contract={self.tokenizer_contract}, "
            f"sampling_contract={self.sampling_contract}, "
            f"decode_alignment={self.decode_alignment}, "
            f"with_rml={self.with_rml}, "
            f"moe_backend={self.moe_backend}, "
            f"runtime_state_dtype={self.runtime_state_dtype}, "
            f"prompt_geometry={self.prompt_geometry}, "
            f"dtype_state={dtype_state}",
            flush=True,
        )

        if campplus_path is None:
            candidate = Path(self.model_dir) / "campplus.onnx"
            campplus_path = str(candidate) if candidate.is_file() else None
        # A no-RML bundle must never instantiate CampPlus, even if a stale
        # campplus.onnx remains in the converted directory.
        self.campplus_path = campplus_path if self.with_rml else None
        self.speaker_extractor = (
            SpkembExtractor(self.campplus_path)
            if self.campplus_path is not None
            else None
        )

    @property
    def sample_rate(self) -> int:
        return int(self.config.tts_config["sample_rate"])

    def extract_speaker_embedding(
        self, prompt_wav: Optional[str]
    ) -> Optional[torch.Tensor]:
        if prompt_wav is None:
            return None
        if self.speaker_extractor is None:
            raise FileNotFoundError(
                "A prompt wav was provided, but campplus.onnx was not found. "
                "Pass --campplus explicitly or put campplus.onnx in model_dir."
            )
        waveform = load_prompt_waveform(prompt_wav, target_sample_rate=16000)
        return self.speaker_extractor(waveform)

    def extract_speaker_embeddings(
        self, prompt_wavs: Sequence[str]
    ) -> torch.Tensor:
        """Extract ordered CampPlus rows for a multi-speaker Podcast."""

        if isinstance(prompt_wavs, (str, bytes)) or len(prompt_wavs) < 2:
            raise ValueError(
                "Podcast requires an ordered list of at least two prompt wavs"
            )
        rows = [self.extract_speaker_embedding(path) for path in prompt_wavs]
        if any(row is None for row in rows):
            raise RuntimeError("Podcast speaker extraction returned no row")
        return torch.cat(rows, dim=0)

    def extract_prompt_waveform(
        self, prompt_wav: Optional[str | Sequence[str]]
    ) -> Optional[torch.Tensor]:
        """Load, concatenate, then align AudioVAE references."""
        if prompt_wav is None:
            return None
        if isinstance(prompt_wav, (str, bytes)):
            waveform = load_prompt_waveform(
                str(prompt_wav), target_sample_rate=self.sample_rate
            )
        else:
            if not isinstance(prompt_wav, Sequence) or not prompt_wav:
                raise ValueError("prompt_wav sequence must not be empty")
            waveforms = [
                load_prompt_waveform(
                    str(path), target_sample_rate=self.sample_rate
                )
                for path in prompt_wav
            ]
            waveform = torch.cat(waveforms, dim=-1)
        pad_align = int(self.prompt_geometry["pad_align_samples"])
        if pad_align < 1:
            raise ValueError(
                f"Invalid prompt waveform alignment: {pad_align}"
            )
        original_length = waveform.shape[1]
        padded_length = (
            (original_length + pad_align - 1) // pad_align * pad_align
        )
        if padded_length != original_length:
            padded = torch.zeros(
                1, padded_length, dtype=waveform.dtype
            )
            padded[:, :original_length] = waveform.clone()
            waveform = padded
        return waveform

    def synthesize(
        self,
        text: str,
        *,
        instruction: Optional[str] = None,
        prompt_wav: Optional[str | Sequence[str]],
        prompt_text: Optional[str],
        target_language: Optional[str] = None,
        include_target_language: Optional[bool] = None,
        conditioning: Optional[str] = None,
        assistant_instruction: bool = False,
        allow_mixed_rml: bool = False,
        decode_alignment_override: Optional[str] = None,
        seed: int = 1234,
        cfg_scale: float = 2.0,
        flow_steps: int = 10,
        sigma: float = 0.25,
        temperature: Optional[float] = 0.0,
        max_decode_steps: Optional[int] = None,
        min_decode_steps: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        log_every_n_steps: int = 0,
    ):
        if conditioning is None:
            conditioning = str(
                self.config.tts_config.get(
                    "conditioning_mode", "prompt_latent"
                )
            )
        if temperature is None:
            temperature = 0.0
        if max_decode_steps is None:
            max_decode_steps = int(
                self.config.tts_config.get("maximum_decode_steps", 800)
            )
        if stop_threshold is None:
            stop_threshold = float(
                self.config.tts_config.get("stop_threshold", 0.5)
            )
        if conditioning not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning must be one of {sorted(CONDITIONING_MODES)}, "
                f"got {conditioning!r}"
            )
        if not allow_mixed_rml:
            if conditioning in NO_RML_CONDITIONING_MODES:
                if self.with_rml:
                    raise ValueError(
                        f"{conditioning} conditioning requires a model "
                        "bundle converted with with_rml=false"
                    )
            elif not self.with_rml:
                raise ValueError(
                    f"{conditioning} conditioning cannot be used with a "
                    "with_rml=false model bundle"
                )
        if conditioning == "no_rml":
            if prompt_wav is not None or prompt_text is not None:
                raise ValueError(
                    "no_rml conditioning must not receive prompt_wav or "
                    "prompt_text"
                )
        elif conditioning in {
            ZERO_SPEAKER_CONDITIONING_MODE,
            MASKED_SPEAKER_CONDITIONING_MODE,
        }:
            if prompt_wav is not None or prompt_text is not None:
                raise ValueError(
                    f"{conditioning} conditioning must not receive prompt_wav "
                    "or prompt_text"
                )
        elif conditioning in PROMPT_LATENT_CONDITIONING_MODES:
            if prompt_wav is None:
                raise ValueError(
                    f"{conditioning} conditioning requires prompt_wav"
                )
            if prompt_text is None or prompt_text == "":
                raise ValueError(
                    f"{conditioning} conditioning requires non-empty "
                    "prompt_text"
                )
        elif conditioning == MULTI_SPEAKER_CONDITIONING_MODE:
            if isinstance(prompt_wav, (str, bytes)) or not isinstance(
                prompt_wav, Sequence
            ) or len(prompt_wav) < 2:
                raise ValueError(
                    "multi_speaker conditioning requires an ordered list of "
                    "at least two prompt wavs"
                )
            if not isinstance(prompt_text, str) or not prompt_text:
                raise ValueError(
                    "multi_speaker conditioning requires the ordered prompt "
                    "transcript for AudioVAE continuation"
                )
        elif conditioning == "speaker_only":
            if prompt_wav is None:
                raise ValueError(
                    "speaker_only conditioning requires prompt_wav for "
                    "CampPlus speaker extraction"
                )
            if prompt_text is not None:
                print(
                    "[Ming-3.0-tts-tiny] speaker_only conditioning ignores "
                    "prompt_text and does not prepend AudioVAE prompt latents",
                    flush=True,
                )
        seed_everything(seed)
        if conditioning == MULTI_SPEAKER_CONDITIONING_MODE:
            speaker_embedding = self.extract_speaker_embeddings(prompt_wav)
            speaker_count = len(prompt_wav)
            campplus_speaker_rows = speaker_count
        else:
            speaker_embedding = (
                None
                if conditioning in NO_RML_CONDITIONING_MODES
                or conditioning in {
                    ZERO_SPEAKER_CONDITIONING_MODE,
                    MASKED_SPEAKER_CONDITIONING_MODE,
                }
                else self.extract_speaker_embedding(prompt_wav)
            )
            speaker_count = None
            uses_real_speaker = (
                conditioning not in NO_RML_CONDITIONING_MODES
                and conditioning not in {
                    ZERO_SPEAKER_CONDITIONING_MODE,
                    MASKED_SPEAKER_CONDITIONING_MODE,
                }
            )
            if uses_real_speaker and speaker_embedding is None:
                raise RuntimeError(
                    f"{conditioning} CampPlus extraction returned no "
                    "speaker embedding"
                )
            campplus_speaker_rows = 1 if uses_real_speaker else 0
        speaker_slot = conditioning not in NO_RML_CONDITIONING_MODES
        uses_prompt_latent = (
            conditioning in PROMPT_LATENT_CONDITIONING_MODES
            or conditioning == MULTI_SPEAKER_CONDITIONING_MODE
        )
        prompt_waveform = (
            self.extract_prompt_waveform(prompt_wav)
            if uses_prompt_latent
            else None
        )
        model_prompt_text = (
            prompt_text
            if uses_prompt_latent
            else None
        )
        output = self.model.generate_speech(
            self.tokenizer,
            text,
            speaker_embedding,
            instruction=instruction,
            prompt_waveform=prompt_waveform,
            prompt_text=model_prompt_text,
            target_language=target_language,
            include_target_language=include_target_language,
            speaker_slot=speaker_slot,
            speaker_count=speaker_count,
            assistant_instruction=assistant_instruction,
            mute_speaker=(
                conditioning == MASKED_SPEAKER_CONDITIONING_MODE
            ),
            decode_alignment_override=decode_alignment_override,
            seed=seed,
            cfg_scale=cfg_scale,
            flow_steps=flow_steps,
            sigma=sigma,
            temperature=temperature,
            max_decode_steps=max_decode_steps,
            min_decode_steps=min_decode_steps,
            stop_threshold=stop_threshold,
            log_every_n_steps=log_every_n_steps,
        )
        # Lets MingTinyTTS reject an output whose RML/CampPlus conditioning was
        # silently lost.
        output.speaker_slot_count = (
            speaker_count if speaker_count is not None else int(speaker_slot)
        )
        output.campplus_speaker_rows = campplus_speaker_rows
        if not torch.isfinite(output.waveform).all():
            raise FloatingPointError(
                "Generated waveform contains NaN or Inf. Verify the patched "
                "chunk-KDA implementation and converted weights."
            )
        return output


def save_waveform(
    path: str, waveform: torch.Tensor, sample_rate: int
) -> None:
    waveform = waveform.detach().float().cpu()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(
            f"Cannot save waveform with shape {tuple(waveform.shape)}"
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torchaudio.save(path, waveform, sample_rate)
