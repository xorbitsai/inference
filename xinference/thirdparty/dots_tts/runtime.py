from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, TypedDict

import librosa
import torch
from huggingface_hub import snapshot_download
from loguru import logger

from .data.pipelines.tokenizing import build_generation_schedule
from .data.pipelines.tts_pipeline import (
    DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
    DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    DEFAULT_TRAIN_TEMPLATE,
)
from .models.dots_tts.model import DotsTtsModel
from .utils.audio import high_quality_resample
from .utils.logging import categorized_log as logc
from .utils.profiling import (
    InferenceProfiler,
    activate_inference_profiler,
    inference_profiling,
    log_inference_profile,
)
from .utils.text import (
    attach_language_tag,
    detect,
    normalize_language_code,
    normalize_text,
)
from .utils.util import get_dtype

RUNTIME_TEMPLATE_BY_NAME = {
    "tts": DEFAULT_TRAIN_TEMPLATE,
    "instruction_tts": DEFAULT_INSTRUCTION_TTS_TEMPLATE,
    "text_to_audio": DEFAULT_TEXT_TO_AUDIO_TEMPLATE,
    "tts_interleave": DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
}
RUNTIME_TEMPLATE_NAMES = tuple(sorted(RUNTIME_TEMPLATE_BY_NAME))
DEFAULT_MAX_SEQUENCE_LENGTH = DotsTtsModel.DEFAULT_MAX_SEQUENCE_LENGTH


class RuntimeInputs(TypedDict, total=False):
    fid: str
    language: str
    text: str
    prompt_text: str
    template_name: str
    generation_schedule: torch.Tensor
    prompt_audio: torch.Tensor


class DotsTtsRuntime:
    # region Lifecycle and pretrained loading
    @staticmethod
    def _check_torch_env(precision: str) -> None:
        import warnings

        if torch.cuda.is_available():
            return
        msg = (
            f"CUDA is not available; torch will run on CPU. "
            f"(torch was built for CUDA {torch.version.cuda!r}.) "
            f"If your machine has a GPU, install a torch build matching your "
            f"CUDA driver from https://pytorch.org/get-started/locally/."
        )
        if precision.lower() in {"bfloat16", "float16", "fp16", "bf16"}:
            raise RuntimeError(
                f"{msg} Half-precision ({precision}) is not reliable on CPU; "
                f"install CUDA-matched torch or pass precision='float32'."
            )
        warnings.warn(msg, stacklevel=2)

    def __init__(
        self,
        model: DotsTtsModel,
        pretrained_path: Path,
        *,
        precision: str = "bfloat16",
        optimize: bool = False,
        max_generate_length: int = 500,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        vocoder_merge_steps: int = 4,
        warmup_on_optimize: bool = True,
    ):
        self._check_torch_env(precision)
        self.model = model
        self.pretrained_path = pretrained_path
        self.precision = precision
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(1)
        if self.device.type == "cuda" and self.precision.lower() in {
            "fp32",
            "torch.float32",
            "float32",
        }:
            torch.set_float32_matmul_precision("high")
        target_dtype = get_dtype(self.precision)
        self.model.core.to(dtype=target_dtype)
        self.model = self.model.to(self.device).eval()
        self.optimize = bool(optimize)
        self.max_generate_length = int(max_generate_length)
        self.max_sequence_length = int(max_sequence_length)
        self.vocoder_merge_steps = int(vocoder_merge_steps)
        if self.max_generate_length <= 0:
            raise ValueError("max_generate_length must be positive.")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")
        if self.vocoder_merge_steps < 1:
            raise ValueError(
                f"vocoder_merge_steps must be >= 1, got {self.vocoder_merge_steps}."
            )
        self.model.set_optimize(
            self.optimize,
            max_sequence_length=self.max_sequence_length,
        )
        self.sample_rate = int(self.model.config.vocoder.sample_rate)
        logger.info(
            logc(
                "runtime",
                "Runtime initialized: pretrained_path={} device={} sample_rate={} "
                "precision={} "
                "optimize={} max_audio_patch_count={} max_sequence_length={} "
                "vocoder_merge_steps={}",
            ),
            self.pretrained_path,
            self.device,
            self.sample_rate,
            self.precision,
            self.optimize,
            self.max_generate_length,
            self.max_sequence_length,
            self.vocoder_merge_steps,
        )
        if warmup_on_optimize and self.optimize and self.device.type == "cuda":
            self.run_warmup()

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
    ) -> DotsTtsRuntime:
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
        loaded_model = DotsTtsModel.from_pretrained(pretrained_path)
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

    @classmethod
    def _resolve_pretrained_path(
        cls,
        model_name_or_path: str,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> Path:
        logger.debug(
            logc(
                "runtime",
                "Resolving pretrained path: model={} revision={} cache_dir={}",
            ),
            model_name_or_path,
            revision,
            cache_dir,
        )
        resolved_path = Path(model_name_or_path).expanduser().resolve()
        if resolved_path.exists():
            logger.debug(
                logc("runtime", "Using local pretrained directory: path={}"),
                resolved_path,
            )
            return resolved_path

        logger.info(
            logc(
                "runtime",
                "Downloading pretrained snapshot: repo_id={} revision={} cache_dir={}",
            ),
            model_name_or_path,
            revision,
            cache_dir,
        )
        snapshot_dir = snapshot_download(
            repo_id=model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
        )
        resolved_path = Path(snapshot_dir).expanduser().resolve()
        logger.info(
            logc("runtime", "Pretrained snapshot ready: path={}"), resolved_path
        )
        return resolved_path

    # endregion Lifecycle and pretrained loading

    # region Warmup
    WARMUP_TEXT = "预热文本"
    WARMUP_PROMPT_TEXT = "预热参考文本"
    WARMUP_EOS_THRESHOLD = 2.0
    WARMUP_NUM_STEPS = 1  # ODE step count is compile-invariant; use 1 for speed.
    # One prompt patch count per KvPrefill-compilable DiT bucket:
    # N=30 lands in bucket 64, N=100 in bucket 128, N=150 in bucket 256.
    # Bucket 512's KvPrefill always falls back to eager (prefix_len exceeds the
    # 192-patch compile cap in CachedDiTRunner), so no candidate targets it.
    WARMUP_PROMPT_PATCH_COUNTS: tuple[int, ...] = (30, 100, 150)

    def run_warmup(self) -> None:
        prompt_patch_counts = tuple(
            int(candidate)
            for candidate in self.WARMUP_PROMPT_PATCH_COUNTS
            if int(candidate) < self.max_generate_length
        )
        start_time = time.time()
        logger.info(
            logc(
                "runtime",
                "Warmup started: iterations={} vocoder_merge_steps={} "
                "prompt_patch_counts={} max_generate_length={}",
            ),
            1 + len(prompt_patch_counts),
            self.vocoder_merge_steps,
            prompt_patch_counts,
            self.max_generate_length,
        )
        # Request 1: no-prompt decode down to the minimum patch count that
        # crosses into every reachable DiT bucket (patches within
        # [prev_bucket + 1, max_generate_length] all land in the largest
        # reachable bucket). Covers FirstPatchStep@64, NextPatchStep across
        # all reachable buckets, and both vocoder chunk sizes (unmerged +
        # merged after >= vocoder_merge_steps + 2 patches).
        self._run_warmup_iteration(
            prompt_patch_count=0,
            target_patch_count=self._min_patches_for_next_bucket_coverage(),
        )
        # Requests 2..N: each short prompt request triggers KvPrefillStep
        # compile in exactly one DiT bucket (the bucket the initial
        # post-prompt decode lands in). Bucket transitions during decode
        # do not fire prefill because copy_prefix_from keeps valid_tokens
        # aligned with persistent_len across the switch.
        for prompt_patch_count in prompt_patch_counts:
            self._run_warmup_iteration(
                prompt_patch_count=prompt_patch_count,
                target_patch_count=1,
            )
        logger.info(
            logc("runtime", "Warmup completed: elapsed_seconds={:.2f}"),
            time.time() - start_time,
        )

    def _min_patches_for_next_bucket_coverage(self) -> int:
        # To trigger NextPatchStep@bucket compile, decode must reach at least
        # patch (prev_bucket + 1) where prev_bucket is the largest bucket
        # strictly smaller than the largest reachable bucket. The largest
        # reachable bucket is the smallest _GENERATE_LENGTH_BUCKETS entry
        # >= max_generate_length (decode caps at max_generate_length patches).
        buckets = tuple(self.model._GENERATE_LENGTH_BUCKETS)
        largest = next(
            (b for b in buckets if b >= self.max_generate_length),
            buckets[-1],
        )
        prev_edge = 0
        for bucket in buckets:
            if bucket >= largest:
                break
            prev_edge = bucket
        return min(self.max_generate_length, prev_edge + 1)

    def _run_warmup_iteration(
        self,
        *,
        prompt_patch_count: int,
        target_patch_count: int,
    ) -> None:
        iteration_start = time.time()
        inputs = self._build_warmup_inputs(prompt_patch_count=prompt_patch_count)
        chunk_count = 0
        emitted_samples = 0
        # Each yielded chunk corresponds to a known patch count: the first
        # VOCODER_STREAM_INITIAL_UNMERGED_PATCHES chunks are 1 patch each,
        # subsequent chunks are vocoder_merge_steps patches each.
        initial_unmerged = int(self.model.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES)
        if self.model.config.sampling is not None:
            warmup_ode_method, warmup_num_steps, warmup_guidance_scale = (
                self.resolve_sampling_options(
                    ode_method=None,
                    num_steps=None,
                    guidance_scale=None,
                )
            )
        else:
            warmup_ode_method = "euler"
            warmup_num_steps = self.WARMUP_NUM_STEPS
            warmup_guidance_scale = 1.2
        stream = self.model.generate_audio_stream(
            inputs,
            precision=self.precision,
            ode_method=warmup_ode_method,
            num_steps=warmup_num_steps,
            guidance_scale=warmup_guidance_scale,
            speaker_scale=1.5,
            eos_threshold=self.WARMUP_EOS_THRESHOLD,
            vocoder_merge_steps=self.vocoder_merge_steps,
        )
        emitted_patches = 0
        try:
            for chunk in stream:
                chunk_count += 1
                emitted_samples += int(chunk.shape[-1])
                emitted_patches += (
                    1
                    if chunk_count <= initial_unmerged
                    else int(self.vocoder_merge_steps)
                )
                if emitted_patches >= int(target_patch_count):
                    break
        finally:
            stream.close()
        logger.info(
            logc(
                "runtime",
                "Warmup iteration finished: prompt_patches={} "
                "target_patches={} decoded_patches~{} chunks={} "
                "audio_seconds={:.2f} elapsed_seconds={:.2f}",
            ),
            prompt_patch_count,
            target_patch_count,
            emitted_patches,
            chunk_count,
            emitted_samples / self.sample_rate,
            time.time() - iteration_start,
        )

    def _build_warmup_inputs(self, *, prompt_patch_count: int) -> RuntimeInputs:
        use_prompt = int(prompt_patch_count) > 0
        text = self.WARMUP_TEXT
        prompt_text = self.WARMUP_PROMPT_TEXT if use_prompt else ""
        prompt_audio: torch.Tensor | None = None
        if use_prompt:
            samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
            if self.max_generate_length <= int(prompt_patch_count):
                raise ValueError(
                    "Warmup prompt patch count must be less than max_generate_length: "
                    f"prompt_patch_count={int(prompt_patch_count)} "
                    f"max_generate_length={self.max_generate_length}."
                )
            prompt_samples = int(prompt_patch_count) * samples_per_patch
            prompt_audio = torch.zeros(
                (1, prompt_samples),
                dtype=torch.float32,
                device=self.device,
            )

        schedule_spec = self._build_runtime_generation_schedule(
            template_name="tts",
            text=text,
            prompt_text=prompt_text,
            prompt_audio_patch_count=0,
        )
        schedule = torch.tensor(
            schedule_spec["schedule_ids"],
            dtype=torch.long,
            device=self.device,
        )
        self._validate_generation_schedule_length(schedule.numel())
        inputs: RuntimeInputs = {
            "fid": "warmup",
            "language": "",
            "text": text,
            "prompt_text": prompt_text,
            "template_name": "tts",
            "generation_schedule": schedule.unsqueeze(0),
        }
        if prompt_audio is not None:
            inputs["prompt_audio"] = prompt_audio
        return inputs

    # endregion Warmup

    # region Request normalization and metadata
    @staticmethod
    def _build_request_id(
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str,
        language: str | None = None,
    ) -> str:
        payload = {
            "text": text,
            "prompt_audio_path": prompt_audio_path,
            "prompt_text": prompt_text,
            "template_name": template_name,
        }
        if language is not None:
            payload["language"] = language
        digest = hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def _load_prompt_audio(
        self,
        prompt_audio_path: str,
    ) -> torch.Tensor:
        logger.debug(logc("io", "Loading prompt audio: path={}"), prompt_audio_path)
        prompt_audio, sample_rate = librosa.load(prompt_audio_path, sr=None, mono=True)
        prompt_audio = librosa.effects.trim(prompt_audio, top_db=30)[0]
        prompt_audio = torch.from_numpy(prompt_audio).unsqueeze(0)
        prompt_audio = high_quality_resample(
            prompt_audio,
            orig_sr=sample_rate,
            target_sr=self.sample_rate,
        )
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        logger.debug(
            logc(
                "io",
                "Prompt audio loaded: path={} original_sample_rate={} resampled_sample_rate={} "
                "samples={}",
            ),
            prompt_audio_path,
            sample_rate,
            self.sample_rate,
            prompt_audio.shape[-1],
        )
        return prompt_audio

    def _resolve_language(
        self,
        language: str | None,
        *,
        text: str,
    ) -> str | None:
        if language is None:
            return None

        stripped = language.strip()
        if not stripped or stripped.lower() == "none":
            return None
        if stripped.lower() == "auto_detect":
            return normalize_language_code(detect(text))

        normalized_language = normalize_language_code(stripped)
        if normalized_language is None:
            raise ValueError(
                f"Unsupported language={language!r}. "
                "Expected 'none', 'auto_detect', or a valid language code/name."
            )
        return normalized_language

    def _process_prompt_text(
        self,
        prompt_text: str | None,
        *,
        language: str | None = None,
    ) -> str:
        if prompt_text is None:
            return ""
        prompt_text = prompt_text.strip()
        if not prompt_text:
            return ""

        prompt_text += "\n"
        if language is not None:
            prompt_text = attach_language_tag(prompt_text, language)
        return prompt_text

    def _process_text(
        self,
        text: str,
        *,
        language: str | None = None,
        normalize: bool = False,
    ) -> tuple[str, str | None]:
        stripped = text.strip()
        if normalize:
            stripped = normalize_text(stripped)
        resolved_language = self._resolve_language(language, text=stripped)
        return stripped, resolved_language

    def _estimate_prompt_audio_patch_count(
        self,
        *,
        prompt_audio: torch.Tensor | None,
        prompt_text: str,
    ) -> int:
        if prompt_audio is None or not prompt_text:
            return 0
        samples_per_patch = int(self.model.config.patch_size * self.model.hop_size)
        prompt_samples = int(prompt_audio.shape[-1])
        return (prompt_samples + samples_per_patch - 1) // samples_per_patch

    # endregion Request normalization and metadata

    # region Generation schedule assembly
    def _normalize_template_name(self, template_name: str | None) -> str:
        if template_name is None:
            return "tts"
        if template_name not in RUNTIME_TEMPLATE_BY_NAME:
            raise ValueError(
                f"Unknown template_name={template_name!r}. "
                f"Expected one of {RUNTIME_TEMPLATE_NAMES}."
            )
        return template_name

    def _build_runtime_generation_schedule(
        self,
        *,
        template_name: str,
        text: str,
        prompt_text: str,
        prompt_audio_patch_count: int,
    ) -> dict[str, Any]:
        del prompt_audio_patch_count
        return build_generation_schedule(
            text=f"{prompt_text}{text}",
            tokenizer=self.model.tokenizer,
            template=RUNTIME_TEMPLATE_BY_NAME[template_name],
            max_audio_tokens=self.max_generate_length,
        )

    def _validate_generation_schedule_length(self, schedule_length: int) -> None:
        if int(schedule_length) > self.max_sequence_length:
            raise ValueError(
                "generation_schedule length exceeds max_sequence_length: "
                f"schedule_length={int(schedule_length)} "
                f"max_sequence_length={self.max_sequence_length}."
            )

    def _prepare_inputs(
        self,
        *,
        text: str,
        prompt_audio_path: str | None,
        prompt_text: str | None,
        template_name: str | None,
        language: str | None = None,
        normalize_text: bool = False,
    ) -> RuntimeInputs:
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
        }

        if prompt_audio_path:
            inputs["prompt_audio"] = self._load_prompt_audio(prompt_audio_path)
        prompt_audio_patch_count = self._estimate_prompt_audio_patch_count(
            prompt_audio=inputs.get("prompt_audio"),
            prompt_text=normalized_prompt_text,
        )
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

    # endregion Generation schedule assembly

    # region Public generation APIs
    def resolve_sampling_options(
        self,
        *,
        ode_method: str | None,
        num_steps: int | None,
        guidance_scale: float | None,
    ) -> tuple[str, int, float]:
        """Apply artifact defaults and reject overrides of locked contracts."""

        sampling = self.model.config.sampling
        if sampling is not None:
            return sampling.resolve(
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
            )
        return (
            "euler" if ode_method is None else str(ode_method),
            10 if num_steps is None else int(num_steps),
            1.2 if guidance_scale is None else float(guidance_scale),
        )

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
        profile_inference: bool = False,
        log_profile_calls: bool = False,
    ) -> Iterator[torch.Tensor]:
        ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
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
        profile_inference: bool = False,
        log_profile_calls: bool = False,
    ) -> dict[str, Any]:
        ode_method, num_steps, guidance_scale = self.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        inputs = self._prepare_inputs(
            text=text,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            language=language,
            normalize_text=normalize_text,
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

    # endregion Public generation APIs
