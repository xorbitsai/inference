from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file

from ...models.dots_tts.core import DotsTtsCore
from ...models.dots_tts.model import DotsTtsModel
from ...utils.logging import categorized_log as logc
from ...utils.profiling import measure_inference
from ...utils.util import get_dtype


@dataclass(frozen=True)
class _AudioFill:
    patches: torch.Tensor | None = None
    latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None
    span_count: int = 0
    fill_llm: bool = False
    fill_fm_history: bool = False
    drop_tail_patch_count: int = 0


@dataclass
class _EditPromptFeatureCacheEntry:
    speaker_embedding: torch.Tensor | None = None
    prompt_latent_distribution: torch.Tensor | None = None


class DotsTtsEditModel(DotsTtsModel):
    """Edit-capable inference wrapper around the public TTS model."""

    _INFERENCE_UNUSED_CORE_KEYS = frozenset({"input_mask_embedding"})

    @classmethod
    def _load_artifact_module(cls, module, path: Path):
        state_dict = load_file(path, device="cpu")
        restored_state_dict = cls._restore_artifact_state_dict(state_dict, module)
        if isinstance(module, DotsTtsCore):
            for key in cls._INFERENCE_UNUSED_CORE_KEYS:
                if key in restored_state_dict and key not in module.state_dict():
                    restored_state_dict.pop(key)
                    logger.info(
                        "Ignoring training-only checkpoint key during inference load: {}",
                        key,
                    )
        mismatch = module.load_state_dict(restored_state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module

    @torch.no_grad()
    def _prepare_audio_fill(
        self,
        audio: torch.Tensor | None,
        *,
        span_count: int = 0,
        fill_llm: bool,
        fill_fm_history: bool = False,
        use_xvector: bool = True,
        speaker_scale: float = 1.5,
        drop_tail_patch_count: int = 0,
    ) -> _AudioFill:
        if drop_tail_patch_count < 0:
            raise ValueError("drop_tail_patch_count must be non-negative.")
        if drop_tail_patch_count > 0 and not fill_llm:
            raise ValueError("drop_tail_patch_count requires fill_llm=True.")
        if fill_fm_history and not fill_llm:
            raise ValueError(
                "Audio cannot fill FM history without also filling the LLM."
            )
        if audio is None:
            if fill_llm:
                raise ValueError("Audio is required when an audio fill enters the LLM.")
            return _AudioFill()

        device = next(self.core.parameters()).device
        audio, cache_key = self._prepare_prompt_audio_for_conditioning(
            audio
        )
        sample_count = int(audio.shape[-1])
        cache_entry = self._get_prompt_feature_cache_entry(cache_key)
        if cache_entry is None:
            cache_entry = _EditPromptFeatureCacheEntry()
        audio = audio.to(device=device)

        g_cond = None
        if use_xvector:
            self.xvector_extractor.eval()
            can_cache_speaker = self._can_cache_speaker_embedding(sample_count)
            speaker_embedding = (
                cache_entry.speaker_embedding if can_cache_speaker else None
            )
            if speaker_embedding is None:
                with measure_inference(
                    "speaker_encoder", phase="audio_fill_conditioning"
                ):
                    speaker_embedding = self.xvector_extractor(audio[None, :])
                if can_cache_speaker:
                    cache_entry.speaker_embedding = speaker_embedding.detach()
            else:
                logger.debug(
                    logc(
                        "conditioning",
                        "Audio-fill speaker cache hit: key={} samples={}",
                    ),
                    cache_key[:12],
                    sample_count,
                )
            g_cond = self.core.xvec_proj(
                speaker_embedding * float(speaker_scale)
            )

        if not fill_llm:
            self._store_prompt_feature_cache_entry(cache_key, cache_entry)
            logger.debug(
                logc(
                    "conditioning",
                    "Reference-only audio fill prepared: samples={} "
                    "speaker_scale={} use_xvector={} device={}",
                ),
                sample_count,
                speaker_scale,
                use_xvector,
                device,
            )
            return _AudioFill(g_cond=g_cond)

        self.vocoder.eval()
        latent_distribution = cache_entry.prompt_latent_distribution
        if latent_distribution is None:
            with measure_inference("latent_encoder", phase="audio_fill_conditioning"):
                latent_distribution = self._get_vocoder_inference().extract_latents(
                    audio[None, :]
                )
            cache_entry.prompt_latent_distribution = latent_distribution.detach()
        else:
            logger.debug(
                logc(
                    "conditioning",
                    "Audio-fill latent cache hit: key={} samples={}",
                ),
                cache_key[:12],
                sample_count,
            )
        self._store_prompt_feature_cache_entry(cache_key, cache_entry)
        latents = self.core.io_helper.sample_from_latent(latent_distribution)
        if drop_tail_patch_count > 0:
            drop_tail_latent_count = drop_tail_patch_count * self.config.patch_size
            if drop_tail_latent_count >= latents.size(1):
                raise ValueError(
                    "drop_tail_patch_count removes all encoded audio latents: "
                    f"drop_tail_patch_count={drop_tail_patch_count} "
                    f"latent_count={latents.size(1)}."
                )
            latents = latents[:, :-drop_tail_latent_count]
        patches = rearrange(
            self.core.io_helper.normalize(latents),
            "b (s p) d -> b s p d",
            p=self.config.patch_size,
        )
        encoded_span_count = int(patches.size(1))
        effective_span_count = (
            encoded_span_count
            if span_count <= 0
            else int(span_count) - drop_tail_patch_count
        )
        if effective_span_count <= 0:
            raise ValueError(
                "audio_fills effective span count must be positive: "
                f"span_count={span_count} "
                f"drop_tail_patch_count={drop_tail_patch_count}."
            )
        if effective_span_count != encoded_span_count:
            raise ValueError(
                "audio_fills effective span count does not match encoded audio patches: "
                f"span_count={span_count} "
                f"drop_tail_patch_count={drop_tail_patch_count} "
                f"effective={effective_span_count} encoded={encoded_span_count}."
            )
        logger.debug(
            logc(
                "conditioning",
                "Audio fill prepared: samples={} patch_count={} "
                "speaker_scale={} use_xvector={} device={}",
            ),
            sample_count,
            encoded_span_count,
            speaker_scale,
            use_xvector,
            device,
        )
        return _AudioFill(
            patches=patches,
            latents=latents,
            g_cond=g_cond,
            span_count=effective_span_count,
            fill_llm=True,
            fill_fm_history=fill_fm_history,
            drop_tail_patch_count=drop_tail_patch_count,
        )

    def _build_audio_fills(
        self,
        data: dict[str, Any],
        *,
        speaker_scale: float,
    ) -> tuple[list[_AudioFill], torch.Tensor | None]:
        if "audio_fills" not in data:
            raise KeyError("generate data must include audio_fills.")
        audio_fills: list[_AudioFill] = []
        g_cond: torch.Tensor | None = None
        for spec in data["audio_fills"]:
            audio_fill = self._prepare_audio_fill(
                spec["audio"],
                span_count=int(spec["span_count"]),
                fill_llm=bool(spec["fill_llm"]),
                fill_fm_history=bool(spec["fill_fm_history"]),
                use_xvector=bool(spec.get("use_xvector", True)),
                speaker_scale=speaker_scale,
                drop_tail_patch_count=int(
                    spec.get("drop_tail_patch_count", 0)
                ),
            )
            if g_cond is None and audio_fill.g_cond is not None:
                g_cond = audio_fill.g_cond
            audio_fills.append(audio_fill)
        return audio_fills, g_cond

    def _prefill_audio_latents(
        self,
        latents: torch.Tensor | None,
        *,
        state: Any,
    ) -> torch.Tensor | None:
        if latents is None:
            return None
        if latents.size(1) == 0:
            return latents.new_zeros(
                (latents.size(0), 0, self.core.llm_hidden_size)
            )
        patch_encoder_input = self._prepare_patch_encoder_input(latents)
        state_dtype = (
            state.fm_sequence.dtype
            if state.fm_sequence is not None
            else patch_encoder_input.dtype
        )
        with measure_inference("patch_encoder", phase="prompt_prefill"):
            patch_embeddings, state.patch_encoder_state = (
                self._get_patch_encoder_inference().prefill_with_state(
                    patch_encoder_input,
                    state.patch_encoder_state,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    dtype=state_dtype,
                )
            )
        return patch_embeddings

    def _locate_edit_prefill_boundary(
        self,
        *,
        span_positions: torch.Tensor,
        llm_fill_patch_count: int,
    ) -> tuple[int, torch.Tensor]:
        if span_positions.numel() > llm_fill_patch_count:
            return int(span_positions[llm_fill_patch_count].item()), span_positions[
                :llm_fill_patch_count
            ]
        raise RuntimeError(
            "Prefill boundary discovery failed despite prior schedule validation."
        )

    def _build_edit_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        audio_fills: list[_AudioFill],
        fill_patch_embeddings: list[torch.Tensor | None],
        fill_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.core.llm.get_input_embeddings()(
            generation_schedule
        ).clone()
        cursor = 0
        for audio_fill, patch_embeddings in zip(
            audio_fills,
            fill_patch_embeddings,
            strict=True,
        ):
            if not audio_fill.fill_llm:
                continue
            span_count = int(audio_fill.span_count)
            audio_fill_positions = fill_span_positions[cursor : cursor + span_count]
            cursor += span_count
            if span_count <= 0:
                continue
            if patch_embeddings is None:
                raise RuntimeError(
                    "Patch embeddings are required when an audio fill enters the LLM."
                )
            segment_embeddings = patch_embeddings[:, :span_count].to(
                inputs_embeds.dtype
            )
            if segment_embeddings.size(1) != span_count:
                raise RuntimeError(
                    f"Audio fill patch embeddings ({segment_embeddings.size(1)}) "
                    f"do not match fill span count ({span_count})."
                )
            inputs_embeds[:, audio_fill_positions, :] = segment_embeddings
        return inputs_embeds

    def _prefill_edit_audio_fills(
        self,
        generation_schedule: torch.Tensor,
        *,
        state: Any,
        span_positions: torch.Tensor,
        audio_fills: list[_AudioFill],
        fill_patch_embeddings: list[torch.Tensor | None],
        audio_placeholder_ids: set[int],
    ) -> int:
        llm_fill_patch_count = sum(
            int(audio_fill.span_count)
            for audio_fill in audio_fills
            if audio_fill.fill_llm
        )
        if span_positions.numel() == llm_fill_patch_count:
            prefill_end = generation_schedule.size(1)
            fill_span_positions = span_positions
        else:
            prefill_end, fill_span_positions = self._locate_edit_prefill_boundary(
                span_positions=span_positions,
                llm_fill_patch_count=llm_fill_patch_count,
            )
        if prefill_end == 0:
            return 0
        inputs_embeds = self._build_edit_prefill_inputs_embeds(
            generation_schedule[:, :prefill_end],
            audio_fills=audio_fills,
            fill_patch_embeddings=fill_patch_embeddings,
            fill_span_positions=fill_span_positions,
        )
        with measure_inference("LLM", phase="prefill"):
            _, llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                inputs_embeds=inputs_embeds,
                request_logits=False,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )
        state.llm_hiddens = llm_hiddens[:, -1:, :]

        cursor = 0
        span_cursor = 0
        for audio_fill in audio_fills:
            if not audio_fill.fill_llm:
                continue
            for audio_fill_patch_index in range(int(audio_fill.span_count)):
                span_position = int(fill_span_positions[span_cursor].item())
                span_cursor += 1
                if not audio_fill.fill_fm_history:
                    continue
                if span_position > cursor:
                    self._append_hidden_chunk(
                        state, llm_hiddens[:, span_position - 1 : span_position, :]
                    )
                patches = audio_fill.patches
                if patches is None:
                    raise RuntimeError(
                        "Audio fill patches are required when filling FM history."
                    )
                self._append_history_chunk(
                    state,
                    patches[:, audio_fill_patch_index],
                )
                if self._next_token_is_audio_span(
                    generation_schedule,
                    position=span_position,
                    audio_placeholder_ids=audio_placeholder_ids,
                ):
                    self._append_hidden_chunk(
                        state,
                        llm_hiddens[:, span_position : span_position + 1, :],
                    )
                cursor = span_position + 1
        if prefill_end > cursor:
            self._append_hidden_chunk(
                state, llm_hiddens[:, prefill_end - 1 : prefill_end, :]
            )
        return prefill_end

    def _decode_edit(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        state: Any,
        audio_placeholder_ids: set[int],
        span_positions: torch.Tensor,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        eos_threshold: float,
        suppress_eos_check_count: int = 0,
    ) -> Iterator[torch.Tensor]:
        span_cursor = torch.searchsorted(
            span_positions,
            torch.tensor(
                position,
                device=span_positions.device,
                dtype=span_positions.dtype,
            ),
        ).item()
        decoded_audio_count = 0
        while position < generation_schedule.size(1):
            token_id = int(generation_schedule[0, position].item())
            if token_id in audio_placeholder_ids:
                profile_step = decoded_audio_count + 1
                should_check_eos = decoded_audio_count >= suppress_eos_check_count
                stop_after_current_audio = (
                    self._should_stop_after_current_audio(
                        state,
                        eos_threshold=eos_threshold,
                    )
                    if should_check_eos
                    else False
                )
                audio_patch = self._decode_next_audio(
                    state,
                    g_cond=g_cond,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    profile_step=profile_step,
                )
                self._consume_audio_patch(
                    state,
                    audio_patch=audio_patch,
                    profile_step=profile_step,
                )
                decoded_audio_count += 1
                if self._next_token_is_audio_span(
                    generation_schedule,
                    position=position,
                    audio_placeholder_ids=audio_placeholder_ids,
                ):
                    self._append_hidden_chunk(state, state.llm_hiddens)
                position += 1
                span_cursor += 1
                yield audio_patch
                if stop_after_current_audio:
                    state.end_flag = True
                    return
                continue
            next_audio_position = (
                int(span_positions[span_cursor].item())
                if span_cursor < span_positions.numel()
                else generation_schedule.size(1)
            )
            position = self._consume_text_schedule(
                generation_schedule,
                position=position,
                next_audio_position=next_audio_position,
                state=state,
                profile_step=decoded_audio_count + 1,
            )

    @torch.no_grad()
    def _generate_latents_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
    ) -> Iterator[torch.Tensor]:
        if "audio_fills" not in data:
            yield from super()._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                eos_threshold=eos_threshold,
            )
            return

        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            generation_schedule: torch.Tensor = data["generation_schedule"]
            if generation_schedule.size(0) != 1:
                raise ValueError(
                    "DotsTtsModel.generate expects batch size 1 for generation_schedule."
                )
            if self._optimize_enabled:
                max_sequence_length = int(self._llm_max_sequence_length)
                schedule_length = int(generation_schedule.size(1))
                if schedule_length > max_sequence_length:
                    raise ValueError(
                        "generation_schedule length exceeds max_sequence_length for "
                        "optimized LLM StaticCache inference: "
                        f"schedule_length={schedule_length} "
                        f"max_sequence_length={max_sequence_length}."
                    )

            audio_fills, g_cond = self._build_audio_fills(
                data,
                speaker_scale=speaker_scale,
            )
            scheduled_fill_patch_count = sum(
                int(audio_fill.span_count)
                for audio_fill in audio_fills
                if audio_fill.fill_llm
            )
            audio_placeholder_ids = set(self.core.audio_span_token_ids)
            span_positions = self._find_audio_span_positions(
                generation_schedule,
                audio_placeholder_ids=audio_placeholder_ids,
            )
            span_count = int(span_positions.numel())
            minimum_required_spans = scheduled_fill_patch_count + 1
            if span_count < minimum_required_spans:
                raise ValueError(
                    f"generation_schedule provides {span_count} audio spans, but audio fills require "
                    f"{scheduled_fill_patch_count} spans and generation requires at least one additional decode span."
                )
            logger.debug(
                logc(
                    "decode",
                    "Latent generation prepared: schedule_audio_spans={} fill_patch_count={} "
                    "minimum_required_spans={}",
                ),
                span_count,
                scheduled_fill_patch_count,
                minimum_required_spans,
            )

            state = self._allocate_generate_state(
                max_audio_patch_count=span_count,
                device=device,
                dtype=dtype,
            )
            fill_patch_embeddings = [
                self._prefill_audio_latents(audio_fill.latents, state=state)
                if audio_fill.fill_llm
                else None
                for audio_fill in audio_fills
            ]
            position = self._prefill_edit_audio_fills(
                generation_schedule,
                state=state,
                span_positions=span_positions,
                audio_fills=audio_fills,
                fill_patch_embeddings=fill_patch_embeddings,
                audio_placeholder_ids=audio_placeholder_ids,
            )

            drop_num_gen_head_patch = int(data.get("drop_num_gen_head_patch", 0))
            if drop_num_gen_head_patch < 0:
                raise ValueError("drop_num_gen_head_patch must be non-negative.")
            payload_patch_count = 0
            for audio_patch in self._decode_edit(
                generation_schedule,
                position=position,
                state=state,
                audio_placeholder_ids=audio_placeholder_ids,
                span_positions=span_positions,
                g_cond=g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                eos_threshold=eos_threshold,
                suppress_eos_check_count=drop_num_gen_head_patch,
            ):
                if drop_num_gen_head_patch > 0:
                    drop_num_gen_head_patch -= 1
                    continue
                payload_patch_count += 1
                if payload_patch_count == 1 or payload_patch_count % 10 == 0:
                    logger.debug(
                        logc(
                            "decode",
                            "Latent generation progress: payload_audio_patches={}",
                        ),
                        payload_patch_count,
                    )
                yield self.core.io_helper.denormalize(audio_patch)

            if payload_patch_count == 0:
                if int(data.get("drop_num_gen_head_patch", 0)) > 0:
                    raise RuntimeError(
                        "Generation produced no payload latents after dropping generated head patches. "
                        "This usually means EOS triggered immediately after the dropped generation prefix "
                        "or the generation schedule did not provide an effective decode span."
                    )
                raise RuntimeError(
                    "Generation produced no decodable latents. "
                    "This usually means EOS triggered before the first decode patch "
                    "or the generation schedule did not provide an effective decode span."
                )
            logger.debug(
                logc("decode", "Latent generation completed: payload_audio_patches={}"),
                payload_patch_count,
            )


__all__ = ["DotsTtsEditModel"]
