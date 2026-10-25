"""High-level API for Ming-3.0-tts-tiny multi-task inference."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from packaging.version import Version

from .task_protocol import (
    FORMAL_CFG,
    FORMAL_FLOW_STEPS,
    FORMAL_SIGMA,
    FORMAL_TEMPERATURE,
    configure_release_environment,
    podcast_frame_bounds_to_decode_steps,
    podcast_generation_frame_bounds,
    resolve_generation_seed,
    resolve_task_request,
    validate_release_bundle,
    validate_release_engine,
)


REQUIRED_TRANSFORMERS = "4.46.3"


def resolve_model_dir(model_path: str) -> str:
    """Return a local checkpoint directory, downloading a Hub repo id."""

    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_path)


def require_transformers() -> None:
    import transformers

    if Version(transformers.__version__) != Version(REQUIRED_TRANSFORMERS):
        raise RuntimeError(
            f"Ming-3.0-tts-tiny requires transformers=={REQUIRED_TRANSFORMERS}, "
            f"found {transformers.__version__}; install requirements-tiny.txt"
        )


class MingTinyTTS:
    """Load one Ming-3.0-tts-tiny checkpoint once and run every released task.

    Use :meth:`from_pretrained` rather than constructing the low-level engine
    directly.  Task-specific conditioning and prompt policies are resolved by
    this class so callers never need to pick a speaker mode or serialize
    structured-control JSON by hand.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @property
    def sample_rate(self) -> int:
        return self.engine.sample_rate

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        moe_backend: str = "auto",
        runtime_state_dtype: str = "auto",
    ) -> "MingTinyTTS":
        """Load a local checkpoint directory or a Hugging Face repo id."""

        if dtype != "bf16":
            raise ValueError("Ming-3.0-tts-tiny public runtime requires dtype='bf16'")
        if moe_backend not in {"auto", "loop"}:
            raise ValueError(
                "Ming-3.0-tts-tiny public runtime requires moe_backend='auto' or 'loop'"
            )
        if runtime_state_dtype not in {"auto", "stable_fp32"}:
            raise ValueError(
                "Ming-3.0-tts-tiny public runtime requires runtime_state_dtype="
                "'auto' or 'stable_fp32'"
            )
        require_transformers()
        configure_release_environment()
        model_dir = resolve_model_dir(model_path)
        validate_release_bundle(model_dir)
        from .inference import MingTinyEngine

        engine = MingTinyEngine(
            model_dir,
            device=device,
            dtype=dtype,
            moe_backend=moe_backend,
            runtime_state_dtype=runtime_state_dtype,
        )
        validate_release_engine(engine)
        return cls(engine)

    def generate(
        self,
        *,
        task: str,
        text: str,
        output_path: Optional[str] = None,
        prompt_wav: Optional[str | Sequence[str]] = None,
        prompt_text: Optional[str | Sequence[str]] = None,
        target_language: Optional[str] = None,
        instruction: Optional[str] = None,
        dialect: Optional[str] = None,
        emotion: Optional[str] = None,
        style: Optional[str] = None,
        speed: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
        ip: Optional[str] = None,
        seed: Optional[int] = None,
        cfg_scale: float = FORMAL_CFG,
        flow_steps: int = FORMAL_FLOW_STEPS,
        sigma: float = FORMAL_SIGMA,
        temperature: float = FORMAL_TEMPERATURE,
        min_decode_steps: Optional[int] = None,
        max_decode_steps: Optional[int] = None,
        stop_threshold: Optional[float] = None,
        min_generation_frames: Optional[int] = None,
        max_generation_frames: Optional[int] = None,
        log_every_n_steps: int = 0,
    ):
        request = resolve_task_request(
            task,
            text=text,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
            target_language=target_language,
            instruction=instruction,
            dialect=dialect,
            emotion=emotion,
            style=style,
            speed=speed,
            pitch=pitch,
            volume=volume,
            ip=ip,
        )
        seed = resolve_generation_seed(request.task, seed)
        self._validate_reference_paths(request.prompt_wav)

        decode_alignment_override = None
        if request.task == "podcast":
            if min_decode_steps is not None or max_decode_steps is not None:
                raise ValueError(
                    "Podcast uses 12.5 Hz generation-frame bounds; do not "
                    "pass decode-step bounds directly"
                )
            if (min_generation_frames is None) != (max_generation_frames is None):
                raise ValueError(
                    "min_generation_frames and max_generation_frames must be paired"
                )
            if min_generation_frames is None:
                min_generation_frames, max_generation_frames = (
                    podcast_generation_frame_bounds(request.text)
                )
            geometry = getattr(self.engine, "prompt_geometry", None)
            if not isinstance(geometry, dict):
                raise RuntimeError("Podcast requires checkpoint prompt geometry")
            min_decode_steps, max_decode_steps = podcast_frame_bounds_to_decode_steps(
                min_generation_frames,
                max_generation_frames,
                int(geometry["flow_patch_size"]),
            )
            alignments = getattr(self.engine, "task_decode_alignment", None)
            decode_alignment_override = (
                alignments.get("podcast") if isinstance(alignments, dict) else None
            )
            if decode_alignment_override not in {"old_shift", "unshifted"}:
                raise RuntimeError(
                    "Podcast requires task_decode_alignment.podcast in config.json"
                )
        elif min_generation_frames is not None or max_generation_frames is not None:
            raise ValueError("generation-frame bounds are Podcast-only")

        output = self.engine.synthesize(
            request.text,
            instruction=request.instruction,
            prompt_wav=request.prompt_wav,
            prompt_text=request.prompt_text,
            target_language=request.target_language,
            include_target_language=request.include_target_language,
            conditioning=request.conditioning,
            assistant_instruction=request.assistant_instruction,
            allow_mixed_rml=request.allow_mixed_rml,
            decode_alignment_override=decode_alignment_override,
            seed=seed,
            cfg_scale=cfg_scale,
            flow_steps=flow_steps,
            sigma=sigma,
            temperature=temperature,
            min_decode_steps=min_decode_steps,
            max_decode_steps=max_decode_steps,
            stop_threshold=stop_threshold,
            log_every_n_steps=log_every_n_steps,
        )
        self._validate_output_contract(request, output)
        if output_path is not None:
            from .inference import save_waveform

            save_waveform(output_path, output.waveform, self.engine.sample_rate)
        return output

    @staticmethod
    def _validate_reference_paths(
        prompt_wav: Optional[str | tuple[str, ...]],
    ) -> None:
        if prompt_wav is None:
            return
        paths = (prompt_wav,) if isinstance(prompt_wav, str) else prompt_wav
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

    @staticmethod
    def _validate_output_contract(request, output) -> None:
        speaker_count = (
            len(request.prompt_wav) if request.conditioning == "multi_speaker" else None
        )
        expected = {
            "prompt_latent": (1, 1, True),
            "multi_speaker": (speaker_count, speaker_count, True),
            "speaker_only": (1, 1, False),
            "masked_speaker": (1, 0, False),
            "zero_speaker": (1, 0, False),
            "no_rml": (0, 0, False),
            "prompt_latent_no_rml": (0, 0, True),
        }
        slots, rows, needs_prompt_latent = expected[request.conditioning]
        actual_slots = getattr(output, "speaker_slot_count", None)
        actual_rows = getattr(output, "campplus_speaker_rows", None)
        actual_prompt_steps = getattr(output, "prompt_latent_steps", None)
        valid_slots = isinstance(actual_slots, int) and not isinstance(
            actual_slots, bool
        )
        valid_rows = isinstance(actual_rows, int) and not isinstance(actual_rows, bool)
        if (
            not valid_slots
            or not valid_rows
            or actual_slots != slots
            or actual_rows != rows
        ):
            raise RuntimeError(
                f"task {request.task} conditioning proof mismatch: "
                f"slots={actual_slots}/{slots}, CampPlus={actual_rows}/{rows}"
            )
        if (
            not isinstance(actual_prompt_steps, int)
            or isinstance(actual_prompt_steps, bool)
            or actual_prompt_steps < 0
        ):
            raise RuntimeError(
                f"task {request.task} returned invalid prompt latent provenance: "
                f"{actual_prompt_steps!r}"
            )
        if needs_prompt_latent and actual_prompt_steps <= 0:
            raise RuntimeError(
                f"task {request.task} produced no AudioVAE prompt continuation"
            )
        if not needs_prompt_latent and actual_prompt_steps != 0:
            raise RuntimeError(
                f"task {request.task} unexpectedly used AudioVAE prompt continuation"
            )
