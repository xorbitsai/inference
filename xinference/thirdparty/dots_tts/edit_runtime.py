from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, TypedDict

import torch
from loguru import logger

from .data.edit_instruction import (
    EditXVectorMode,
    render_source_text,
    render_target_text,
    resolve_edit_use_xvector,
)
from .data.pipelines.tokenizing import build_edit_generation_schedule
from .models.dots_tts.edit_model import DotsTtsEditModel
from .runtime import DEFAULT_MAX_SEQUENCE_LENGTH, DotsTtsRuntime
from .utils.audio import prepare_edit_source_audio
from .utils.logging import categorized_log as logc
from .utils.profiling import (
    InferenceProfiler,
    activate_inference_profiler,
    inference_profiling,
    log_inference_profile,
)


EDIT_SOURCE_TEXT_PREFIX = "[原文本]"
EDIT_SOURCE_AUDIO_PREFIX = "[原语音]"
EDIT_INSTRUCTION_PREFIX = "[编辑指令]"
EDIT_TARGET_TEXT_PREFIX = "[编辑文本]"
EDIT_TARGET_AUDIO_PREFIX = "[编辑后语音]"


class EditRuntimeInputs(TypedDict, total=False):
    """Prepared Edit request passed to :class:`DotsTtsEditModel`.

    ``generation_schedule`` is int64 with shape ``[1, sequence_length]``.
    ``audio_fills`` follows the audio-span order in that schedule. Each fill
    owns mono float audio ``[1, samples]``, its scheduled patch count, and the
    explicit LLM/FM/speaker/tail-drop behavior used by model prefill.
    """

    fid: str
    language: str
    text: str
    prompt_text: str
    template_name: str
    generation_schedule: torch.Tensor
    audio_fills: list[dict[str, Any]]
    drop_num_gen_head_patch: int
    source_text: str
    instruction: str
    source_text_source: str
    target_text_source: str


class DotsTtsEditRuntime(DotsTtsRuntime):
    """Edit-capable runtime that leaves the public TTS runtime unchanged."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        vocoder_merge_steps: int = 4,
        warmup_on_optimize: bool = True,
    ) -> DotsTtsEditRuntime:
        logger.info(
            logc(
                "runtime",
                "Runtime load started: model={} revision={} cache_dir={} precision={}",
            ),
            model_name_or_path,
            revision,
            cache_dir,
            precision,
        )
        pretrained_path = cls._resolve_pretrained_path(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        loaded_model = DotsTtsEditModel.from_pretrained(pretrained_path)
        logger.info(
            logc("runtime", "Runtime load completed: pretrained_path={}"),
            pretrained_path,
        )
        return cls(
            model=loaded_model,
            pretrained_path=pretrained_path,
            precision=precision,
            optimize=optimize,
            max_generate_length=max_generate_length,
            max_sequence_length=max_sequence_length,
            vocoder_merge_steps=vocoder_merge_steps,
            warmup_on_optimize=warmup_on_optimize,
        )

    def run_warmup(self) -> None:
        """Warm the official TTS path, then exercise the Edit prefill path."""

        super().run_warmup()
        if self.max_generate_length <= 1:
            logger.warning(
                logc(
                    "runtime",
                    "Edit warmup skipped: max_generate_length={} leaves no "
                    "source-plus-target audio budget.",
                ),
                self.max_generate_length,
            )
            return

        source_patch_count = min(2, self.max_generate_length - 1)
        target_patch_count = self.max_generate_length - source_patch_count
        source_text = "预热"
        target_text = "热"
        instruction = '<del>预</del>热'
        schedule_spec = build_edit_generation_schedule(
            source_text=source_text,
            target_text=target_text,
            instruction=instruction,
            tokenizer=self.model.tokenizer,
            source_text_prefix=EDIT_SOURCE_TEXT_PREFIX,
            source_audio_prefix=EDIT_SOURCE_AUDIO_PREFIX,
            instruction_prefix=EDIT_INSTRUCTION_PREFIX,
            target_text_prefix=EDIT_TARGET_TEXT_PREFIX,
            target_audio_prefix=EDIT_TARGET_AUDIO_PREFIX,
            source_num_audio_tokens=source_patch_count,
            target_max_audio_tokens=target_patch_count,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        self._validate_generation_schedule_length(schedule.numel())
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        source_audio = torch.zeros(
            (1, source_patch_count * samples_per_patch),
            dtype=torch.float32,
            device=self.device,
        )
        inputs: EditRuntimeInputs = {
            "fid": "edit-warmup",
            "language": "",
            "text": target_text,
            "source_text": source_text,
            "prompt_text": source_text,
            "instruction": instruction,
            "template_name": "edit",
            "audio_fills": [
                {
                    "audio": source_audio,
                    "span_count": source_patch_count,
                    "fill_llm": True,
                    "fill_fm_history": False,
                    "use_xvector": False,
                    "drop_tail_patch_count": 0,
                }
            ],
            "drop_num_gen_head_patch": 0,
            "generation_schedule": schedule.unsqueeze(0),
        }
        if self.model.config.sampling is not None:
            ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
                ode_method=None,
                num_steps=None,
                guidance_scale=None,
            )
        else:
            ode_method = "euler"
            num_steps = self.WARMUP_NUM_STEPS
            guidance_scale = 1.2
        stream = self.model.generate_audio_stream(
            inputs,
            precision=self.precision,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=1.5,
            eos_threshold=self.WARMUP_EOS_THRESHOLD,
            vocoder_merge_steps=self.vocoder_merge_steps,
        )
        try:
            next(stream)
        finally:
            stream.close()
        logger.info(logc("runtime", "Edit prefill warmup completed."))

    def _load_edit_source_audio(self, source_audio_path: str) -> torch.Tensor:
        """Load and token-align an Edit source waveform."""

        logger.debug(
            logc("io", "Loading edit source audio: path={}"), source_audio_path
        )
        source_audio = prepare_edit_source_audio(
            source_audio_path,
            target_sample_rate=self.sample_rate,
            samples_per_llm_token=int(
                self.model.config.patch_size * self.model.hop_size
            ),
        )
        logger.debug(
            logc(
                "io",
                "Edit source audio loaded: path={} sample_rate={} samples={}",
            ),
            source_audio_path,
            self.sample_rate,
            source_audio.shape[-1],
        )
        return source_audio

    @staticmethod
    def _resolve_edit_transcripts(
        *,
        instruction: str,
        source_text: str | None,
        target_text: str | None,
    ) -> tuple[str, str, str, str]:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty.")

        # Parse both surfaces even when explicit transcripts are supplied, so
        # malformed tagged instructions never reach the model.
        rendered_source = render_source_text(instruction)
        rendered_target = render_target_text(instruction)
        explicit_source = source_text if source_text and source_text.strip() else None
        explicit_target = target_text if target_text and target_text.strip() else None
        resolved_source = (
            explicit_source.strip()
            if explicit_source is not None
            else rendered_source.strip()
        )
        resolved_target = (
            explicit_target.strip()
            if explicit_target is not None
            else rendered_target.strip()
        )
        if not resolved_source:
            raise ValueError(
                "source_text is missing and instruction renders an empty source transcript."
            )
        if not resolved_target:
            raise ValueError(
                "target_text is missing and instruction renders an empty target transcript."
            )
        return (
            resolved_source,
            resolved_target,
            "request" if explicit_source is not None else "instruction",
            "request" if explicit_target is not None else "instruction",
        )

    def _prepare_edit_inputs(
        self,
        *,
        source_audio_path: str,
        instruction: str,
        source_text: str | None = None,
        target_text: str | None = None,
        use_xvector: EditXVectorMode = "auto",
    ) -> EditRuntimeInputs:
        (
            resolved_source,
            resolved_target,
            source_text_source,
            target_text_source,
        ) = self._resolve_edit_transcripts(
            instruction=instruction,
            source_text=source_text,
            target_text=target_text,
        )
        source_audio = self._load_edit_source_audio(source_audio_path)
        source_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=source_audio,
            prompt_text=resolved_source,
        )
        if self.max_generate_length <= source_patch_count:
            raise ValueError(
                "max_generate_length must exceed edit source audio patch count: "
                f"max_generate_length={self.max_generate_length} "
                f"source_audio_patch_count={source_patch_count}."
            )
        schedule_spec = build_edit_generation_schedule(
            source_text=resolved_source,
            target_text=resolved_target,
            instruction=instruction,
            tokenizer=self.model.tokenizer,
            source_text_prefix=EDIT_SOURCE_TEXT_PREFIX,
            source_audio_prefix=EDIT_SOURCE_AUDIO_PREFIX,
            instruction_prefix=EDIT_INSTRUCTION_PREFIX,
            target_text_prefix=EDIT_TARGET_TEXT_PREFIX,
            target_audio_prefix=EDIT_TARGET_AUDIO_PREFIX,
            source_num_audio_tokens=source_patch_count,
            target_max_audio_tokens=self.max_generate_length - source_patch_count,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        self._validate_generation_schedule_length(schedule.numel())
        request_payload = {
            "source_audio_path": source_audio_path,
            "source_text": resolved_source,
            "target_text": resolved_target,
            "instruction": instruction,
        }
        request_id = hashlib.sha1(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "fid": request_id,
            "language": "",
            "text": resolved_target,
            "source_text": resolved_source,
            "prompt_text": resolved_source,
            "instruction": instruction,
            "template_name": "edit",
            "audio_fills": [
                {
                    "audio": source_audio,
                    "span_count": source_patch_count,
                    "fill_llm": True,
                    "fill_fm_history": False,
                    "use_xvector": resolve_edit_use_xvector(
                        use_xvector,
                        instruction,
                    ),
                    "drop_tail_patch_count": 0,
                }
            ],
            "drop_num_gen_head_patch": 0,
            "source_text_source": source_text_source,
            "target_text_source": target_text_source,
            "generation_schedule": schedule.unsqueeze(0),
        }

    def _prepare_audio_fill_tts_inputs(
        self,
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str | None,
        language: str | None = None,
        normalize_text: bool = False,
        use_xvector: bool = True,
    ) -> EditRuntimeInputs:
        normalized_template_name = self._normalize_template_name(template_name)
        if prompt_text and not prompt_audio_path:
            raise ValueError("prompt_text requires prompt_audio_path.")

        normalized_text, normalized_language = self._process_text(
            text,
            language=language,
            normalize=normalize_text,
        )
        normalized_prompt_text = self._process_prompt_text(
            prompt_text,
            language=normalized_language,
        )
        if normalized_language is not None and not normalized_prompt_text:
            normalized_text = attach_language_tag(normalized_text, normalized_language)
        inputs: RuntimeInputs = {
            "fid": self._build_request_id(
                text=normalized_text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text,
                template_name=normalized_template_name,
                language=normalized_language,
            ),
            "language": normalized_language or "",
            "text": normalized_text,
            "prompt_text": normalized_prompt_text,
            "template_name": normalized_template_name,
            "audio_fills": [],
            "drop_num_gen_head_patch": 0,
        }

        prompt_audio = (
            self._load_prompt_audio(prompt_audio_path) if prompt_audio_path else None
        )
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=prompt_audio,
            prompt_text=normalized_prompt_text,
        )
        if prompt_audio is not None:
            has_prompt_text = bool(normalized_prompt_text)
            inputs["audio_fills"].append(
                {
                    "audio": prompt_audio,
                    "span_count": prompt_audio_patch_count,
                    "fill_llm": has_prompt_text,
                    "fill_fm_history": has_prompt_text,
                    "use_xvector": bool(use_xvector),
                    "drop_tail_patch_count": 1 if has_prompt_text else 0,
                }
            )
            if has_prompt_text:
                inputs["drop_num_gen_head_patch"] = 1
        if (
            prompt_audio_patch_count > 0
            and self.max_generate_length <= prompt_audio_patch_count
        ):
            raise ValueError(
                "max_generate_length must exceed prompt audio patch count when prompt_text is provided: "
                f"max_generate_length={self.max_generate_length} "
                f"prompt_audio_patch_count={prompt_audio_patch_count}."
            )

        schedule_spec = self._build_runtime_generation_schedule(
            template_name=normalized_template_name,
            text=normalized_text,
            prompt_text=normalized_prompt_text,
            prompt_audio_patch_count=prompt_audio_patch_count,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        self._validate_generation_schedule_length(schedule.numel())
        inputs["generation_schedule"] = schedule.unsqueeze(0)
        logger.debug(
            logc(
                "request",
                "Inputs prepared: request_id={} template_name={} "
                "language={} text_len={} prompt_text_len={} schedule_length={} "
                "prompt_audio_patch_count={} max_audio_patch_count={} "
                "max_sequence_length={} has_prompt_audio={}",
            ),
            inputs["fid"],
            normalized_template_name,
            normalized_language,
            len(normalized_text),
            len(normalized_prompt_text),
            schedule.numel(),
            prompt_audio_patch_count,
            self.max_generate_length,
            self.max_sequence_length,
            bool(prompt_audio_path),
        )
        return inputs

    def generate_stream(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        normalize_text: bool = False,
        use_xvector: bool = True,
        profile_inference: bool = False,
        log_profile_calls: bool = False,
    ) -> Iterator[torch.Tensor]:
        if use_xvector or not prompt_audio_path:
            yield from super().generate_stream(
                text=text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=prompt_text,
                template_name=template_name,
                language=language,
                speaker_scale=speaker_scale,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                normalize_text=normalize_text,
                profile_inference=profile_inference,
                log_profile_calls=log_profile_calls,
            )
            return

        ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        inputs = self._prepare_audio_fill_tts_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
            use_xvector=use_xvector,
        )
        profile_inference = bool(profile_inference or log_profile_calls)
        logger.debug(
            logc(
                "request",
                "Streaming generation started: request_id={} text_len={} has_prompt_audio={} "
                "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
                "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={} "
                "vocoder_merge_steps={}",
            ),
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
            self.vocoder_merge_steps,
        )
        start_time = time.time()
        emitted_samples = 0
        chunk_count = 0
        profiler: InferenceProfiler | None = None
        try:
            profiler = (
                InferenceProfiler(
                    self.device,
                    log_calls=True if log_profile_calls else None,
                    request_id=inputs["fid"],
                )
                if profile_inference
                else None
            )
            stream = self.model.generate_audio_stream(
                inputs,
                precision=self.precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                vocoder_merge_steps=self.vocoder_merge_steps,
            )
            while True:
                try:
                    with activate_inference_profiler(profiler):
                        chunk = next(stream)
                except StopIteration:
                    break
                emitted_samples += int(chunk.shape[-1])
                chunk_count += 1
                yield chunk
        except Exception:
            logger.exception(
                logc("request", "Streaming generation failed: request_id={}"),
                inputs["fid"],
            )
            raise
        time_used = time.time() - start_time
        duration_seconds = emitted_samples / self.sample_rate
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profile_inference and profiler is not None:
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiler.summary(duration_seconds=duration_seconds),
                duration_seconds=duration_seconds,
            )
        logger.info(
            logc(
                "request",
                "Streaming generation finished: request_id={} chunk_count={} elapsed_seconds={:.3f} "
                "audio_seconds={:.3f} rtf={:.4f} sample_rate={}",
            ),
            inputs["fid"],
            chunk_count,
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )

    def generate(
        self,
        *,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str | None = None,
        language: str | None = None,
        speaker_scale: float = 1.5,
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        normalize_text: bool = False,
        use_xvector: bool = True,
        profile_inference: bool = False,
        log_profile_calls: bool = False,
    ) -> dict[str, Any]:
        if use_xvector or not prompt_audio_path:
            return super().generate(
                text=text,
                prompt_audio_path=prompt_audio_path,
                prompt_text=prompt_text,
                template_name=template_name,
                language=language,
                speaker_scale=speaker_scale,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                normalize_text=normalize_text,
                profile_inference=profile_inference,
                log_profile_calls=log_profile_calls,
            )

        ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        inputs = self._prepare_audio_fill_tts_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
            use_xvector=use_xvector,
        )
        profile_inference = bool(profile_inference or log_profile_calls)
        logger.debug(
            logc(
                "request",
                "Generation started: request_id={} text_len={} has_prompt_audio={} "
                "has_prompt_text={} template_name={} language={} precision={} ode_method={} num_steps={} "
                "guidance_scale={} speaker_scale={} max_audio_patch_count={} normalize_text={}",
            ),
            inputs["fid"],
            len(inputs["text"]),
            bool(prompt_audio_path),
            bool(inputs["prompt_text"]),
            inputs["template_name"],
            inputs["language"] or None,
            self.precision,
            ode_method,
            num_steps,
            guidance_scale,
            speaker_scale,
            self.max_generate_length,
            normalize_text,
        )
        start_time = time.time()
        profiling = None
        try:
            with inference_profiling(
                enabled=profile_inference,
                device=self.device,
                log_calls=True if log_profile_calls else None,
                request_id=inputs["fid"],
            ) as profiler:
                audio = self.model.generate_audio(
                    inputs,
                    precision=self.precision,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=speaker_scale,
                )
        except Exception:
            logger.exception(
                logc("request", "Generation failed: request_id={}"),
                inputs["fid"],
            )
            raise
        time_used = time.time() - start_time
        duration_seconds = audio.shape[-1] / self.sample_rate
        rtf = time_used / duration_seconds if duration_seconds > 0 else float("inf")
        if profiler is not None:
            profiling = profiler.summary(duration_seconds=duration_seconds)
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiling,
                duration_seconds=duration_seconds,
            )
        logger.info(
            logc(
                "request",
                "Generation completed: request_id={} elapsed_seconds={:.3f} audio_seconds={:.3f} "
                "rtf={:.4f} sample_rate={}",
            ),
            inputs["fid"],
            time_used,
            duration_seconds,
            rtf,
            self.sample_rate,
        )
        return {
            "fid": inputs["fid"],
            "audio": audio,
            "sample_rate": self.sample_rate,
            "time_used": time_used,
            "rtf": rtf,
            "profiling": profiling,
        }

    def generate_edit(
        self,
        *,
        source_audio_path: str,
        instruction: str,
        source_text: str | None = None,
        target_text: str | None = None,
        use_xvector: EditXVectorMode = "auto",
        speaker_scale: float = 1.5,
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        profile_inference: bool = False,
        log_profile_calls: bool = False,
    ) -> dict[str, Any]:
        ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        inputs = self._prepare_edit_inputs(
            source_audio_path=source_audio_path,
            instruction=instruction,
            source_text=source_text,
            target_text=target_text,
            use_xvector=use_xvector,
        )
        profile_inference = bool(profile_inference or log_profile_calls)
        start_time = time.time()
        profiling = None
        try:
            with inference_profiling(
                enabled=profile_inference,
                device=self.device,
                log_calls=True if log_profile_calls else None,
                request_id=inputs["fid"],
            ) as profiler:
                audio = self.model.generate_audio(
                    inputs,
                    precision=self.precision,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    speaker_scale=speaker_scale,
                )
        except Exception:
            logger.exception(
                logc("request", "Edit generation failed: request_id={}"),
                inputs["fid"],
            )
            raise
        time_used = time.time() - start_time
        duration_seconds = audio.shape[-1] / self.sample_rate
        if profiler is not None:
            profiling = profiler.summary(duration_seconds=duration_seconds)
            log_inference_profile(
                request_id=inputs["fid"],
                profiling=profiling,
                duration_seconds=duration_seconds,
            )
        return {
            "fid": inputs["fid"],
            "request_id": inputs["fid"],
            "audio": audio,
            "sample_rate": self.sample_rate,
            "duration_seconds": duration_seconds,
            "time_used": time_used,
            "rtf": (
                time_used / duration_seconds
                if duration_seconds > 0
                else float("inf")
            ),
            "profiling": profiling,
            "source_text": inputs["source_text"],
            "target_text": inputs["text"],
            "source_text_source": inputs["source_text_source"],
            "target_text_source": inputs["target_text_source"],
        }


__all__ = ["DotsTtsEditRuntime", "EditRuntimeInputs"]
