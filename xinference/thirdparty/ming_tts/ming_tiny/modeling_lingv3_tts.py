"""Hugging Face model wrapper and autoregressive inference for Ming-3.0-tts-tiny."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .modeling_audio_vae import AudioVAE
from .configuration_lingv3_tts import LingV3TTSConfig
from .aggregator import Aggregator
from .flowloss import FlowLoss
from .modeling_bailing_moe_v3 import BailingMoeV3ForCausalLM
from .prompt_contract import (
    NATIVE_FP32_SAMPLING,
    BASE_PROTOCOL,
    OLD_SHIFT_DECODE_ALIGNMENT,
    UNSHIFTED_DECODE_ALIGNMENT,
    build_multispeaker_tts_prompt_ids,
    build_tts_prompt_ids as build_prompt_ids,
    decode_alignment,
    protocol_version,
    sampling_contract,
    validate_tts_tokenizer as validate_prompt_tokenizer,
    with_rml,
)


def inject_multispeaker_embeddings(
    inputs_embeds: torch.Tensor,
    speaker_embedding: torch.Tensor,
    speaker_positions: List[int],
    spk_head: nn.Module,
    *,
    speaker_count: int,
    speaker_dim: int,
) -> None:
    """Project and inject N ordered Podcast speaker rows in place."""

    if speaker_embedding.ndim != 2 or tuple(speaker_embedding.shape) != (
        speaker_count,
        speaker_dim,
    ):
        raise ValueError(
            "Podcast speaker_embedding must have shape "
            f"[{speaker_count}, {speaker_dim}], got "
            f"{tuple(speaker_embedding.shape)}"
        )
    if len(speaker_positions) != speaker_count:
        raise RuntimeError(
            "Podcast prompt speaker-slot count disagrees with speaker "
            "embeddings"
        )
    speaker_hidden = spk_head(speaker_embedding)
    expected_hidden_shape = (speaker_count, inputs_embeds.shape[-1])
    if tuple(speaker_hidden.shape) != expected_hidden_shape:
        raise RuntimeError(
            "spk_head returned an unexpected Podcast shape: "
            f"{tuple(speaker_hidden.shape)} != {expected_hidden_shape}"
        )
    inputs_embeds[:, speaker_positions, :] = speaker_hidden.unsqueeze(0)

# Keep the full remote-code dependency set as direct relative imports.
# Transformers copies only direct imports into its dynamic-module cache;
# without these anchors AutoConfig followed by AutoModel can leave the
# recursive AudioVAE/Flow files missing from that cache.
from .CFM import CFM as _CFMDependency  # noqa: F401
from .configuration_audio_vae import (  # noqa: F401
    AudioVAEconfig as _AudioVAEConfigDependency,
)
from .configuration_bailing_moe_v3 import (  # noqa: F401
    BailingMoeV3Config as _BailingConfigDependency,
)
from .dit import DiT as _DiTDependency  # noqa: F401
from .fm_utils import is_package_available as _package_dependency  # noqa: F401
from .istft import ISTFTHead as _ISTFTDependency  # noqa: F401
from .mlp import SimpleMLPAdaLN as _MLPDependency  # noqa: F401
from .modules import DiTBlock as _DiTBlockDependency  # noqa: F401
from .vae_modules import Encoder as _AudioVAEEncoderDependency  # noqa: F401


@dataclass
class LingV3TTSGenerationOutput:
    waveform: torch.Tensor
    latents: torch.Tensor
    stop_probabilities: List[float]
    stopped: bool
    prompt_token_count: int
    prompt_latent_steps: int
    # Populated by MingTinyEngine after model generation, so callers can check
    # that the speaker conditioning a task requires was actually applied.
    speaker_slot_count: int = 0
    campplus_speaker_rows: int = 0


class LingV3TTSForConditionalGeneration(PreTrainedModel):
    """Ming-3.0-tts-tiny backbone plus the AudioVAE, Flow/DiT and TTS heads."""

    config_class = LingV3TTSConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _no_split_modules = [
        "BailingMoeV3DecoderLayer",
        "BailingMoeV3MTPLayer",
        "DiTBlock",
    ]

    def __init__(self, config: LingV3TTSConfig):
        super().__init__(config)
        self.model = BailingMoeV3ForCausalLM(config.llm_config)
        self.audio = AudioVAE(config.audio_config)

        aggregator_config = dict(config.aggregator_config)
        aggregator_config["in_channels"] = int(
            aggregator_config.get(
                "in_channels", config.tts_config["latent_dim"]
            )
        )
        aggregator_config["llm_input_dim"] = config.llm_config.hidden_size
        self.linear_proj_audio = Aggregator(**aggregator_config)

        flow_config = dict(config.flow_config)
        z_channels = int(
            flow_config.pop(
                "z_channels", config.tts_config["latent_dim"]
            )
        )
        flow_config.pop("llm_cond_dim", None)
        flow_model_type = flow_config.pop("flow_model_type", None)
        self.flowloss = FlowLoss(
            z_channels=z_channels,
            llm_cond_dim=config.llm_config.hidden_size,
            flow_model_type=flow_model_type,
            **flow_config,
        )

        self.stop_head = nn.Linear(
            config.llm_config.hidden_size, 2, bias=True
        )
        self.spk_head = nn.Linear(
            int(config.tts_config["speaker_embedding_dim"]),
            config.llm_config.hidden_size,
            bias=True,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def forward(self, *args, **kwargs):
        """Delegate text/backbone forwards to the Ming-3.0-tts-tiny causal LM."""
        return self.model(*args, **kwargs)

    def enable_native_atorch_moe(self) -> int:
        """Pack every sparse layer after HF shard loading."""

        sparse_layers = [
            module
            for module in self.modules()
            if hasattr(module, "enable_native_atorch_moe")
            and module is not self
        ]
        for layer_index, sparse_layer in enumerate(sparse_layers, 1):
            sparse_layer.enable_native_atorch_moe()
            gc.collect()
            print(
                "[Ming-3.0-tts-tiny] packed native atorch MoE "
                f"{layer_index}/{len(sparse_layers)}",
                flush=True,
            )
        return len(sparse_layers)

    def validate_tts_tokenizer(self, tokenizer) -> Dict[str, int]:
        """Fail fast if the tokenizer does not use the training-time IDs."""
        return validate_prompt_tokenizer(
            tokenizer,
            self.config.tts_config,
            self.config.llm_config.eos_token_id,
        )

    def validate_runtime_dtypes(
        self, *, native_bf16_state: bool = False
    ) -> Dict[str, int]:
        """Protect routing/KDA numeric state from silent HF dtype casts."""
        input_embeddings = self.get_input_embeddings()
        output_embeddings = self.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise RuntimeError(
                "Ming-3.0-tts-tiny must expose both input embeddings and an LM head"
            )
        if (
            input_embeddings.weight is output_embeddings.weight
            or input_embeddings.weight.data_ptr()
            == output_embeddings.weight.data_ptr()
        ):
            raise RuntimeError(
                "Ming-3.0-tts-tiny input embeddings and lm_head were silently tied. "
                "They are independent checkpoint tensors; ensure the outer "
                "LingV3TTSConfig returns llm_config from get_text_config()."
            )

        expert_bias = {
            name: buffer
            for name, buffer in self.named_buffers()
            if name.endswith(".mlp.gate.expert_bias")
        }
        if len(expert_bias) != 23:
            raise RuntimeError(
                f"Expected 23 expert_bias buffers, got {len(expert_bias)}"
            )
        state_dtype = (
            input_embeddings.weight.dtype
            if native_bf16_state
            else torch.float32
        )
        wrong_expert = {
            name: str(buffer.dtype)
            for name, buffer in expert_bias.items()
            if buffer.dtype != state_dtype
        }
        if wrong_expert:
            raise TypeError(
                f"MoE expert_bias must be {state_dtype}: "
                + str(wrong_expert)
            )

        kda_state = {
            name: parameter
            for name, parameter in self.named_parameters()
            if name.endswith(".attention.A_log")
            or name.endswith(".attention.dt_bias")
        }
        if len(kda_state) != 36:
            raise RuntimeError(
                f"Expected 36 KDA A_log/dt_bias tensors, got {len(kda_state)}"
            )
        wrong_kda = {
            name: str(parameter.dtype)
            for name, parameter in kda_state.items()
            if parameter.dtype != state_dtype
        }
        if wrong_kda:
            raise TypeError(
                f"KDA A_log/dt_bias must be {state_dtype}: "
                + str(wrong_kda)
            )

        rotary_names = {
            "flowloss.cfm.model.rotary_embed.inv_freq",
            "linear_proj_audio.rotary_embed.inv_freq",
        }
        rotary_state = {
            name: buffer
            for name, buffer in self.named_buffers()
            if name in rotary_names
        }
        if set(rotary_state) != rotary_names:
            raise RuntimeError(
                "Missing Flow/Aggregator rotary buffers: "
                f"{sorted(rotary_state)}"
            )
        wrong_rotary = {
            name: str(buffer.dtype)
            for name, buffer in rotary_state.items()
            if buffer.dtype != state_dtype
        }
        if wrong_rotary:
            raise TypeError(
                f"Flow/Aggregator rotary inv_freq must be {state_dtype}: "
                + str(wrong_rotary)
            )
        return {
            "untied_lm_head": 1,
            "native_bf16_state": int(native_bf16_state),
            "expert_bias_checked": len(expert_bias),
            "kda_checked": len(kda_state),
            "rotary_checked": len(rotary_state),
        }

    def build_tts_prompt_ids(
        self,
        tokenizer,
        target_text: str,
        *,
        instruction: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_latent_tokens: int = 0,
        speaker_slot: Optional[bool] = None,
        assistant_instruction: bool = False,
        target_language: Optional[str] = None,
        include_target_language: Optional[bool] = None,
    ) -> tuple[torch.LongTensor, Optional[int]]:
        """Reproduce the current training SampleBuilder token sequence."""
        ids, speaker_embedding_position = build_prompt_ids(
            tokenizer,
            target_text,
            self.config.tts_config,
            instruction=instruction,
            prompt_text=prompt_text,
            prompt_latent_tokens=prompt_latent_tokens,
            speaker_slot=speaker_slot,
            assistant_instruction=assistant_instruction,
            target_language=target_language,
            include_target_language=include_target_language,
        )
        return (
            torch.tensor(ids, dtype=torch.long).unsqueeze(0),
            speaker_embedding_position,
        )

    @torch.inference_mode()
    def generate_speech(
        self,
        tokenizer,
        target_text: str,
        speaker_embedding: Optional[torch.Tensor],
        *,
        instruction: Optional[str] = None,
        prompt_waveform: Optional[torch.Tensor] = None,
        prompt_text: Optional[str] = None,
        target_language: Optional[str] = None,
        include_target_language: Optional[bool] = None,
        speaker_slot: Optional[bool] = None,
        speaker_count: Optional[int] = None,
        assistant_instruction: bool = False,
        mute_speaker: bool = False,
        decode_alignment_override: Optional[str] = None,
        seed: int = 1234,
        cfg_scale: float = 2.0,
        flow_steps: int = 10,
        sigma: float = 0.25,
        temperature: Optional[float] = 0.0,
        max_decode_steps: Optional[int] = None,
        min_decode_steps: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        return_waveform: bool = True,
        log_every_n_steps: int = 0,
    ) -> LingV3TTSGenerationOutput:
        """Generate one waveform using the exact training-time TTS protocol."""
        if not target_text:
            raise ValueError("target_text must not be empty")
        if cfg_scale < 0:
            raise ValueError("cfg_scale must be non-negative")
        if temperature is None:
            temperature = 0.0
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if max_decode_steps is None:
            max_decode_steps = int(
                self.config.tts_config.get("maximum_decode_steps", 800)
            )
        if stop_threshold is None:
            stop_threshold = float(
                self.config.tts_config.get("stop_threshold", 0.5)
            )
        if max_decode_steps < 1:
            raise ValueError("max_decode_steps must be positive")
        if (prompt_waveform is None) != (prompt_text is None):
            raise ValueError(
                "prompt_waveform and prompt_text must be provided together"
            )
        if prompt_text == "":
            raise ValueError("prompt_text must not be empty")
        if mute_speaker and speaker_slot is False:
            raise ValueError("mute_speaker requires an existing speaker slot")
        multispeaker = speaker_count is not None
        if multispeaker:
            if instruction is not None:
                raise ValueError("Podcast must not receive an instruction")
            if prompt_waveform is None or prompt_text is None:
                raise ValueError(
                    "Podcast requires its ordered speaker references as an "
                    "AudioVAE prompt continuation"
                )
            if speaker_slot is False:
                raise ValueError("Podcast requires speaker RML slots")
            if mute_speaker:
                raise ValueError("Podcast speakers must not be muted")

        self.validate_tts_tokenizer(tokenizer)
        parameter = next(self.parameters())
        device = parameter.device
        dtype = parameter.dtype

        patch_size = int(self.config.tts_config["patch_size"])
        history_size = int(self.config.tts_config["history_size"])
        latent_dim = int(self.config.tts_config["latent_dim"])
        if history_size < patch_size:
            raise ValueError(
                "history_size must be greater than or equal to patch_size"
            )

        is_base_protocol = (
            protocol_version(self.config.tts_config)
            == BASE_PROTOCOL
        )
        selected_decode_alignment = (
            str(decode_alignment_override)
            if decode_alignment_override is not None
            else decode_alignment(self.config.tts_config)
        )
        if selected_decode_alignment not in {
            OLD_SHIFT_DECODE_ALIGNMENT,
            UNSHIFTED_DECODE_ALIGNMENT,
        }:
            raise ValueError(
                "Unsupported decode_alignment_override="
                f"{selected_decode_alignment!r}"
            )
        old_shift_decode = (
            selected_decode_alignment == OLD_SHIFT_DECODE_ALIGNMENT
        )
        native_fp32_sampling = (
            sampling_contract(self.config.tts_config)
            == NATIVE_FP32_SAMPLING
        )
        if native_fp32_sampling:
            # rc1.6 uses the global CPU stream for initial Flow noise and the
            # global device stream for Euler noise.  It does not construct a
            # dedicated CUDA generator.
            torch.manual_seed(int(seed))
            generator = None
        else:
            # Preserve that shared-stream ordering without mutating unrelated
            # CUDA generators in the modern Megatron runtime.
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        use_amp = (
            device.type in {"cuda", "xpu", "npu"}
            and dtype in {torch.float16, torch.bfloat16}
        )

        def autocast_context():
            if use_amp:
                return torch.autocast(device_type=device.type, dtype=dtype)
            return nullcontext()

        prompt_latent: Optional[torch.Tensor] = None
        prompt_hidden: Optional[torch.Tensor] = None
        prompt_latent_steps = 0
        prompt_latent_tokens = 0
        if prompt_waveform is not None:
            if prompt_waveform.ndim == 1:
                prompt_waveform = prompt_waveform.unsqueeze(0)
            if prompt_waveform.ndim != 2 or prompt_waveform.shape[0] != 1:
                raise ValueError(
                    "prompt_waveform must have shape [samples] or "
                    f"[1, samples], got {tuple(prompt_waveform.shape)}"
                )
            if prompt_waveform.shape[1] < 1:
                raise ValueError("prompt_waveform must not be empty")
            prompt_waveform = prompt_waveform.to(device=device)
            prompt_waveform_length = torch.tensor(
                [prompt_waveform.shape[1]],
                dtype=torch.long,
                device=device,
            )
            with autocast_context():
                prompt_latent, prompt_frame_num = self.audio.encode_latent(
                    prompt_waveform,
                    prompt_waveform_length,
                    generator=generator,
                )
            if (
                prompt_latent.ndim != 3
                or prompt_latent.shape[0] != 1
                or prompt_latent.shape[2] != latent_dim
            ):
                raise RuntimeError(
                    "AudioVAE returned an unexpected prompt latent shape: "
                    f"{tuple(prompt_latent.shape)}"
                )
            prompt_latent_steps = int(prompt_latent.shape[1])
            reported_steps = int(prompt_frame_num.reshape(-1)[0].item())
            if reported_steps != prompt_latent_steps:
                raise RuntimeError(
                    "AudioVAE prompt length disagrees with its latent: "
                    f"reported={reported_steps}, actual={prompt_latent_steps}"
                )
            if prompt_latent_steps % patch_size:
                raise ValueError(
                    "Prompt AudioVAE latent length must be divisible by the "
                    f"Flow patch size: {prompt_latent_steps} % {patch_size}"
                )
            if not torch.isfinite(prompt_latent).all():
                raise FloatingPointError(
                    "AudioVAE prompt latent contains NaN or Inf"
                )
            prompt_latent_tokens = prompt_latent_steps // patch_size
            grouped_prompt_latent = prompt_latent.reshape(
                -1, patch_size, latent_dim
            )
            with autocast_context():
                prompt_hidden = self.linear_proj_audio(
                    grouped_prompt_latent
                )
            prompt_hidden = prompt_hidden.reshape(
                1, prompt_latent_tokens, -1
            )

        if multispeaker:
            prompt_ids, speaker_positions = build_multispeaker_tts_prompt_ids(
                tokenizer,
                target_text,
                self.config.tts_config,
                speaker_count=speaker_count,
                prompt_text=prompt_text,
                prompt_latent_tokens=prompt_latent_tokens,
                target_language=target_language,
                include_target_language=include_target_language,
            )
            input_ids = torch.tensor(
                prompt_ids, dtype=torch.long
            ).unsqueeze(0)
            speaker_position = None
        else:
            input_ids, speaker_position = self.build_tts_prompt_ids(
                tokenizer,
                target_text,
                instruction=instruction,
                prompt_text=prompt_text,
                prompt_latent_tokens=prompt_latent_tokens,
                speaker_slot=speaker_slot,
                assistant_instruction=assistant_instruction,
                target_language=target_language,
                include_target_language=include_target_language,
            )
            speaker_positions = (
                [] if speaker_position is None else [speaker_position]
            )
        input_ids = input_ids.to(device)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if prompt_hidden is not None:
            prompt_start = input_ids.shape[1] - prompt_latent_tokens
            placeholder_ids = input_ids[0, prompt_start:]
            expected_patch_id = int(
                self.config.tts_config["audio_patch_token_id"]
            )
            if not torch.all(placeholder_ids == expected_patch_id):
                raise RuntimeError(
                    "Prompt latent placeholders are not the final audio "
                    "patch tokens"
                )
            inputs_embeds[:, prompt_start:, :] = prompt_hidden.to(
                dtype=inputs_embeds.dtype
            )

        rml_enabled = (
            with_rml(self.config.tts_config)
            if speaker_slot is None
            else bool(speaker_slot)
        )
        if rml_enabled:
            if not speaker_positions:
                raise RuntimeError(
                    "speaker-slot prompt is missing its speaker placeholder"
                )
            if multispeaker:
                speaker_dim = int(
                    self.config.tts_config["speaker_embedding_dim"]
                )
                if speaker_embedding is None:
                    raise ValueError(
                        "Podcast requires one speaker embedding per RML slot"
                    )
                speaker_embedding = speaker_embedding.to(
                    device=device, dtype=dtype
                )
                inject_multispeaker_embeddings(
                    inputs_embeds,
                    speaker_embedding,
                    speaker_positions,
                    self.spk_head,
                    speaker_count=speaker_count,
                    speaker_dim=speaker_dim,
                )
            elif mute_speaker:
                if speaker_embedding is not None:
                    raise ValueError(
                        "mute_speaker inference must not receive a speaker "
                        "embedding"
                    )
                # Zero-shot cloning (as in the 0.5b model) and Voice Design/IP
                # zero the projected speaker row *after*
                # spk_head. Applying spk_head to an all-zero input is not
                # equivalent when the projection has bias.
                inputs_embeds[:, speaker_position, :].zero_()
            else:
                speaker_dim = int(
                    self.config.tts_config["speaker_embedding_dim"]
                )
                if speaker_embedding is None:
                    # Preserve the historical zero-speaker behavior used by
                    # older diagnostics and zero-shot control tasks.
                    speaker_embedding = torch.zeros(
                        1, speaker_dim, device=device, dtype=dtype
                    )
                else:
                    speaker_embedding = speaker_embedding.to(
                        device=device, dtype=dtype
                    )
                    if speaker_embedding.ndim == 1:
                        speaker_embedding = speaker_embedding.unsqueeze(0)
                    speaker_embedding = speaker_embedding.reshape(
                        speaker_embedding.shape[0], -1
                    )
                    if speaker_embedding.shape != (1, speaker_dim):
                        raise ValueError(
                            "speaker_embedding must have shape "
                            f"[{speaker_dim}] or [1, {speaker_dim}], got "
                            f"{tuple(speaker_embedding.shape)}"
                        )

                speaker_hidden = self.spk_head(speaker_embedding)
                inputs_embeds[:, speaker_position, :] = speaker_hidden
        else:
            if speaker_positions:
                raise RuntimeError(
                    "no-speaker-slot prompt unexpectedly has a speaker "
                    "placeholder"
                )
            if speaker_embedding is not None:
                raise ValueError(
                    "no-speaker-slot inference must not receive or inject a "
                    "speaker embedding"
                )

        if min_decode_steps is None:
            min_decode_steps = int(
                self.config.tts_config["minimum_decode_steps"]
            )
        if not 0 <= min_decode_steps <= max_decode_steps:
            raise ValueError(
                "min_decode_steps must be between 0 and "
                f"max_decode_steps={max_decode_steps}, got "
                f"{min_decode_steps}"
            )

        latent_history = torch.zeros(
            1,
            history_size,
            latent_dim,
            device=device,
            dtype=torch.float32 if native_fp32_sampling else dtype,
        )
        if prompt_latent is not None:
            history_prompt = prompt_latent.to(
                device=device, dtype=latent_history.dtype
            )
            if prompt_latent_steps >= history_size:
                latent_history.copy_(history_prompt[:, -history_size:, :])
            else:
                latent_history[:, -prompt_latent_steps:, :] = history_prompt
        generated: List[torch.Tensor] = []
        stop_probabilities: List[float] = []
        past_key_values = None
        offset = 0
        stopped = False
        if is_base_protocol:
            attention_mask = torch.ones(
                input_ids.shape, device=device
            )
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_(
                attention_mask == 0, 1
            )

        with autocast_context():
            current_embeds = inputs_embeds
            decode_iterations = (
                max_decode_steps
                if old_shift_decode
                else max_decode_steps + 1
            )
            for _ in range(decode_iterations):
                current_length = current_embeds.shape[1]
                if is_base_protocol:
                    # Match BailingMM.sample_instruct_tts exactly: explicit
                    # attention/position tensors and no cache_position.
                    outputs = self.model(
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        inputs_embeds=current_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    cache_position = torch.arange(
                        offset,
                        offset + current_length,
                        device=device,
                        dtype=torch.long,
                    )
                    position_ids = cache_position.unsqueeze(0)
                    outputs = self.model(
                        inputs_embeds=current_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden = outputs.hidden_states[-1][:, -1:, :]
                generated_steps = len(generated)
                if not old_shift_decode:
                    stop_probability = (
                        self.stop_head(hidden)
                        .softmax(dim=-1)[0, 0, 1]
                        .float()
                        .item()
                    )
                    stop_probabilities.append(stop_probability)
                if (
                    not old_shift_decode
                    and
                    generated_steps >= min_decode_steps
                    and stop_probability > stop_threshold
                ):
                    stopped = True
                    if log_every_n_steps > 0:
                        print(
                            "[Ming-3.0-tts-tiny decode] "
                            f"stop after {generated_steps} latent steps, "
                            f"stop_p={stop_probability:.6f}",
                            flush=True,
                        )
                    break
                if generated_steps >= max_decode_steps:
                    break

                sampled = self.flowloss.sample(
                    hidden,
                    latent_history,
                    cfg=cfg_scale,
                    patch_size=patch_size,
                    sigma=sigma,
                    temperature=temperature,
                    steps=flow_steps,
                    generator=generator,
                    noise_dtype=(
                        torch.float32 if native_fp32_sampling else dtype
                    ),
                    initial_noise_on_cpu=native_fp32_sampling,
                )
                if isinstance(sampled, tuple):
                    sampled = sampled[0]
                if sampled.shape != (1, patch_size, latent_dim):
                    raise RuntimeError(
                        "Flow returned an unexpected shape: "
                        f"{tuple(sampled.shape)}, expected "
                        f"(1, {patch_size}, {latent_dim})"
                    )
                generated.append(sampled)

                past_key_values = outputs.past_key_values
                offset += current_length
                current_embeds = self.linear_proj_audio(sampled)

                if history_size == patch_size:
                    latent_history.copy_(sampled)
                else:
                    latent_history[:, :-patch_size, :] = (
                        latent_history[:, patch_size:, :].clone()
                    )
                    latent_history[:, -patch_size:, :] = sampled

                generated_steps = len(generated)
                if is_base_protocol:
                    position_ids = (
                        position_ids[0][-1] + 1
                    ).view(1, -1)
                    attention_mask = torch.ones(
                        current_embeds.shape[0],
                        1,
                        device=current_embeds.device,
                    )
                if old_shift_decode:
                    # With shift_tts_labels_like_old=true, the hidden state
                    # predicts a Flow patch before its stop label is applied.
                    stop_probability = (
                        self.stop_head(hidden)
                        .softmax(dim=-1)[0, 0, 1]
                        .float()
                        .item()
                    )
                    stop_probabilities.append(stop_probability)
                    if (
                        generated_steps >= min_decode_steps
                        and stop_probability > stop_threshold
                    ):
                        stopped = True
                        if log_every_n_steps > 0:
                            print(
                                "[Ming-3.0-tts-tiny decode] "
                                f"stop after {generated_steps} latent steps, "
                                f"stop_p={stop_probability:.6f}",
                                flush=True,
                            )
                        break
                if log_every_n_steps > 0 and (
                    generated_steps == 1
                    or generated_steps % log_every_n_steps == 0
                ):
                    latent_rms = (
                        sampled.float().square().mean().sqrt().item()
                    )
                    print(
                        "[Ming-3.0-tts-tiny decode] "
                        f"step={generated_steps}/{max_decode_steps} "
                        f"stop_p={stop_probability:.6f} "
                        f"latent_rms={latent_rms:.6f}",
                        flush=True,
                    )

        if not generated:
            raise RuntimeError("Ming-3.0-tts-tiny generated no audio latents")
        latents = torch.cat(generated, dim=1)

        if return_waveform:
            with autocast_context():
                waveform = self.audio.infer_from_latent(latents)
            waveform = waveform.float()
        else:
            waveform = torch.empty(0, device=device)

        return LingV3TTSGenerationOutput(
            waveform=waveform,
            latents=latents,
            stop_probabilities=stop_probabilities,
            stopped=stopped,
            prompt_token_count=input_ids.shape[1],
            prompt_latent_steps=prompt_latent_steps,
        )
