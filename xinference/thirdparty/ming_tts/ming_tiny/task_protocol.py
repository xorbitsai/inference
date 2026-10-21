"""Public task contract for the Ming-3.0-tts-tiny multi-task checkpoint.

The runtime in :mod:`ming_tiny.inference` owns tensor execution.  This
module owns the task-facing part of the contract: which conditioning path a
task uses, where its instruction is placed, and the exact structured-control
JSON learned during training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence

from .prompt_contract import (
    BASE_PROTOCOL,
    _canonical_protocol,
    normalize_target_language,
)


FORMAL_SEED = 1234
FORMAL_PODCAST_SEED = 1235
FORMAL_CFG = 2.0
FORMAL_TEMPERATURE = 0.0
FORMAL_FLOW_STEPS = 10
FORMAL_SIGMA = 0.25

CONTROL_FIELD_ORDER = (
    "序号",
    "说话人",
    "方言",
    "情感",
    "风格",
    "语速",
    "基频",
    "音量",
    "BGM",
    "IP",
)
BGM_FIELD_ORDER = ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")

EMOTION_CONTROL_FIELD_ORDER = (
    "序号",
    "说话人",
    "方言",
    "风格",
    "语速",
    "基频",
    "音量",
    "情感",
    "BGM",
    "IP",
)

DIALECT_VALUES = frozenset({"广粤话", "川渝话"})
EMOTION_VALUES = frozenset({"中性", "悲伤", "愤怒", "高兴", "惊讶", "恐惧", "厌恶"})
SPEED_VALUES = frozenset({"慢速", "中速", "快速"})
PITCH_VALUES = frozenset({"低", "中", "高"})
VOLUME_VALUES = frozenset({"低", "中", "高"})
PARALINGUISTIC_TAGS = (
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
)

EXPECTED_ENGINE_CONTRACT = {
    "protocol_version": "thinking_v1",
    "prompt_format": BASE_PROTOCOL,
    "tokenizer_contract": BASE_PROTOCOL,
    "sampling_contract": "native_fp32",
    "decode_alignment": "old_shift",
    "enable_tts_target_language": True,
    "infer_tts_language_from_text": True,
    "task_contract": "lingv3_sft_multitask_v1",
    "with_rml": True,
}
# config.json may still carry the pre-rename protocol name.
_PROTOCOL_FIELDS = frozenset({"protocol_version", "prompt_format", "tokenizer_contract"})
FORMAL_RUNTIME_ENVIRONMENT = {
    "BAILING_KDA_FUSED_RECURRENT": "0",
    "BAILING_MLA_ATTN": "sdpa",
    "MING_TINY_AUDIO_VAE_ATTN": "sdpa",
    "FLA_USE_CUDA_KDA_FWD": "0",
    "FLA_USE_CUDA_KDA_BWD": "0",
}

DIALECT_ALIASES = {
    "cantonese": "广粤话",
    "canton": "广粤话",
    "粤语": "广粤话",
    "广州话": "广粤话",
    "sichuan": "川渝话",
    "sichuanese": "川渝话",
    "chuanyu": "川渝话",
    "chuanyu dialect": "川渝话",
    "四川话": "川渝话",
    "重庆话": "川渝话",
}
EMOTION_ALIASES = {
    "happy": "高兴",
    "sad": "悲伤",
    "angry": "愤怒",
    "neutral": "中性",
    "surprised": "惊讶",
    "surprise": "惊讶",
    "fearful": "恐惧",
    "fear": "恐惧",
    "disgusted": "厌恶",
    "disgust": "厌恶",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    conditioning: str
    include_target_language: bool
    instruction_kind: str
    reference_count: str
    prompt_text_mode: str
    allowed_languages: Optional[frozenset[str]] = None


TASK_SPECS: Mapping[str, TaskSpec] = {
    "seed": TaskSpec(
        "seed",
        "prompt_latent",
        True,
        "none",
        "one",
        "one",
        frozenset({"Chinese", "English"}),
    ),
    "multilingual": TaskSpec(
        "multilingual",
        "prompt_latent",
        True,
        "none",
        "one",
        "one",
        frozenset({"German", "Spanish", "French", "Korean", "Japanese"}),
    ),
    "podcast": TaskSpec("podcast", "multi_speaker", True, "none", "many", "many"),
    "basic": TaskSpec("basic", "speaker_only", False, "basic", "one", "none"),
    "emotion": TaskSpec("emotion", "speaker_only", False, "emotion", "one", "none"),
    "dialect": TaskSpec("dialect", "speaker_only", False, "dialect", "one", "none"),
    "voice_design": TaskSpec(
        "voice_design", "masked_speaker", False, "style", "none", "none"
    ),
    "ip": TaskSpec("ip", "masked_speaker", False, "ip", "none", "none"),
    "freeform": TaskSpec(
        "freeform", "masked_speaker", False, "freeform", "optional_one", "none"
    ),
    "paralinguistic": TaskSpec(
        "paralinguistic",
        "prompt_latent",
        True,
        "none",
        "one",
        "one",
        frozenset({"Chinese", "English"}),
    ),
}
TASKS = tuple(TASK_SPECS)

# Task names used by the 0.5b API and the evaluation manifests.
TASK_ALIASES = {
    "tts": "seed",
    "clone": "seed",
    "zero_shot": "seed",
    "basic_attributes": "basic",
    "basic_attribute": "basic",
    "emotion_cv3": "emotion",
    "emotion_cv3n": "emotion",
    "style": "voice_design",
    "voice": "voice_design",
    "voice_actor": "ip",
    "nvbench": "paralinguistic",
}

_PODCAST_SPEAKER_RE = re.compile(r"^\s*speaker_(\d+)\s*:")


@dataclass(frozen=True)
class ResolvedTaskRequest:
    task: str
    text: str
    instruction: Optional[str]
    prompt_wav: Optional[str | tuple[str, ...]]
    prompt_text: Optional[str]
    target_language: Optional[str]
    include_target_language: bool
    conditioning: str
    assistant_instruction: bool = True
    allow_mixed_rml: bool = True


def configure_release_environment() -> None:
    """Install missing numerical defaults and reject conflicting ones."""

    conflicts = {}
    for name, expected in FORMAL_RUNTIME_ENVIRONMENT.items():
        actual = os.environ.get(name)
        if actual is None:
            os.environ[name] = expected
        elif actual != expected:
            conflicts[name] = (expected, actual)
    if conflicts:
        raise RuntimeError(
            "Ming-3.0-tts-tiny runtime environment mismatch: " + repr(conflicts)
        )


def validate_release_bundle(
    model_dir: str,
    *,
    tokenizer_dir: Optional[str] = None,
    campplus_path: Optional[str] = None,
) -> None:
    """Cheap preflight before allocating the model."""

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    tokenizer_root = Path(tokenizer_dir or root).expanduser().resolve()
    if tokenizer_root != root:
        raise RuntimeError("Ming-3.0-tts-tiny requires the tokenizer shipped in model_dir")
    expected_campplus = (root / "campplus.onnx").resolve()
    selected_campplus = Path(campplus_path or expected_campplus).expanduser().resolve()
    if selected_campplus != expected_campplus:
        raise RuntimeError("Ming-3.0-tts-tiny requires the campplus.onnx shipped in model_dir")
    required_files = (
        root / "config.json",
        root / "model.safetensors.index.json",
        root / "tokenizer.json",
        root / "tokenizer_config.json",
        expected_campplus,
    )
    missing = [str(path.name) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete Ming-3.0-tts-tiny bundle; missing files: {missing}"
        )
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError("invalid JSON in Ming-3.0-tts-tiny bundle") from error
    tts_config = config.get("tts_config")
    if not isinstance(tts_config, Mapping):
        raise RuntimeError("Ming-3.0-tts-tiny config.json has no tts_config")
    tts_config = {
        name: (
            _canonical_protocol(value)
            if name in _PROTOCOL_FIELDS and value is not None
            else value
        )
        for name, value in tts_config.items()
    }
    expected = {
        **EXPECTED_ENGINE_CONTRACT,
        "sample_rate": 44_100,
        "speaker_embedding_dim": 192,
        "latent_dim": 64,
        "patch_size": 1,
        "history_size": 1,
        "audio_patch_token_id": 157_164,
        "audio_start_token_id": 157_165,
        "audio_end_token_id": 157_166,
        "speaker_start_token_id": 157_175,
        "speaker_end_token_id": 157_176,
        "prompt_eos_token_id": 156_895,
        "minimum_decode_steps": 13,
        "maximum_decode_steps": 250,
        "stop_threshold": 0.5,
    }
    mismatches = {
        name: (wanted, tts_config.get(name))
        for name, wanted in expected.items()
        if tts_config.get(name) != wanted
    }
    task_alignment = tts_config.get("task_decode_alignment")
    if not isinstance(task_alignment, Mapping) or (
        task_alignment.get("podcast") != "old_shift"
    ):
        mismatches["task_decode_alignment.podcast"] = (
            "old_shift",
            task_alignment,
        )
    if mismatches:
        raise RuntimeError("Ming-3.0-tts-tiny bundle contract mismatch: " + repr(mismatches))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise RuntimeError("model.safetensors.index.json has no weight_map")
    missing_shards = sorted(
        shard for shard in set(weight_map.values()) if not (root / shard).is_file()
    )
    if missing_shards:
        raise FileNotFoundError(
            f"Ming-3.0-tts-tiny bundle is missing weight shards: {missing_shards}"
        )


def validate_release_engine(engine) -> None:
    """Fail closed unless an engine matches the released checkpoint contract."""

    configure_release_environment()
    actual = {
        name: getattr(engine, name, None) for name in EXPECTED_ENGINE_CONTRACT
    }
    mismatches = {
        name: (expected, actual[name])
        for name, expected in EXPECTED_ENGINE_CONTRACT.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "Ming-3.0-tts-tiny checkpoint contract mismatch: " + repr(mismatches)
        )
    if int(getattr(engine, "sample_rate", 0)) != 44_100:
        raise RuntimeError("Ming-3.0-tts-tiny requires sample_rate=44100")
    if getattr(engine, "moe_backend", None) != "loop":
        raise RuntimeError("Ming-3.0-tts-tiny requires moe_backend=loop")
    if getattr(engine, "runtime_state_dtype", None) != "stable_fp32":
        raise RuntimeError("Ming-3.0-tts-tiny requires runtime_state_dtype=stable_fp32")
    if str(getattr(engine, "dtype", None)) != "torch.bfloat16":
        raise RuntimeError("Ming-3.0-tts-tiny requires dtype=bf16")
    if os.path.realpath(getattr(engine, "model_dir", "")) != os.path.realpath(
        getattr(engine, "tokenizer_dir", "")
    ):
        raise RuntimeError("Ming-3.0-tts-tiny requires the tokenizer shipped in model_dir")
    if getattr(engine, "campplus_path", None) is None:
        raise RuntimeError("Ming-3.0-tts-tiny bundle is missing campplus.onnx")
    expected_campplus = os.path.join(engine.model_dir, "campplus.onnx")
    if os.path.realpath(engine.campplus_path) != os.path.realpath(expected_campplus):
        raise RuntimeError("Ming-3.0-tts-tiny requires the campplus.onnx shipped in model_dir")
    if getattr(engine, "speaker_extractor", None) is None:
        raise RuntimeError("Ming-3.0-tts-tiny did not initialize CampPlus")

    tts_config = getattr(getattr(engine, "config", None), "tts_config", None)
    if not isinstance(tts_config, Mapping):
        raise RuntimeError("Ming-3.0-tts-tiny engine has no TTS config")
    expected_generation = {
        "sample_rate": 44_100,
        "speaker_embedding_dim": 192,
        "latent_dim": 64,
        "patch_size": 1,
        "history_size": 1,
        "audio_patch_token_id": 157_164,
        "audio_start_token_id": 157_165,
        "audio_end_token_id": 157_166,
        "speaker_start_token_id": 157_175,
        "speaker_end_token_id": 157_176,
        "prompt_eos_token_id": 156_895,
        "minimum_decode_steps": 13,
        "maximum_decode_steps": 250,
        "stop_threshold": 0.5,
    }
    generation_mismatches = {
        name: (expected, tts_config.get(name))
        for name, expected in expected_generation.items()
        if tts_config.get(name) != expected
    }
    if generation_mismatches:
        raise RuntimeError(
            "Ming-3.0-tts-tiny decode defaults mismatch: " + repr(generation_mismatches)
        )
    geometry = getattr(engine, "prompt_geometry", None)
    expected_geometry = {
        "sample_rate": 44_100,
        "latent_dim": 64,
        "audio_patch_size": 4,
        "flow_patch_size": 1,
        "samples_per_audio_latent": 3_528,
        "pad_align_samples": 3_528,
    }
    if not isinstance(geometry, Mapping) or any(
        geometry.get(name) != expected for name, expected in expected_geometry.items()
    ):
        raise RuntimeError("Ming-3.0-tts-tiny prompt geometry mismatch: " + repr(geometry))
    task_alignments = getattr(engine, "task_decode_alignment", None)
    if not isinstance(task_alignments, Mapping) or (
        task_alignments.get("podcast") != "old_shift"
    ):
        raise RuntimeError(
            "Ming-3.0-tts-tiny requires task_decode_alignment.podcast=old_shift"
        )

    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Ming-3.0-tts-tiny engine has no tokenizer")
    special_tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    offenders = sorted(special_tokens.intersection(PARALINGUISTIC_TAGS))
    if offenders:
        raise RuntimeError(
            "paralinguistic tags must remain ordinary tokenizer tokens: "
            + repr(offenders)
        )
    for tag in PARALINGUISTIC_TAGS:
        if not tokenizer.encode(tag, add_special_tokens=False):
            raise RuntimeError(f"tokenizer produced no IDs for {tag!r}")


def normalize_task(task: str) -> str:
    value = str(task or "").strip().lower().replace("-", "_")
    value = TASK_ALIASES.get(value, value)
    if value not in TASK_SPECS:
        raise ValueError(
            f"unsupported task {task!r}; expected one of: "
            + ", ".join(sorted(TASK_SPECS))
        )
    return value


def resolve_generation_seed(task: str, seed: Optional[int] = None) -> int:
    if seed is not None:
        return seed
    return FORMAL_PODCAST_SEED if normalize_task(task) == "podcast" else FORMAL_SEED


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def build_control_instruction(
    *,
    dialect: Optional[str] = None,
    emotion: Optional[str] = None,
    style: Optional[str] = None,
    speed: Optional[str] = None,
    pitch: Optional[str] = None,
    volume: Optional[str] = None,
    ip: Optional[str] = None,
) -> str:
    """Serialize the exact structured-control bytes the checkpoint learned.

    Field order is semantic.  Keep this default ordering stable; the emotion
    task applies its archived evaluation ordering separately.
    """

    for name, value in (
        ("dialect", dialect),
        ("emotion", emotion),
        ("style", style),
        ("speed", speed),
        ("pitch", pitch),
        ("volume", volume),
        ("ip", ip),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
    if dialect is not None:
        dialect = DIALECT_ALIASES.get(dialect.strip().lower(), dialect.strip())
        if dialect not in DIALECT_VALUES:
            raise ValueError(f"dialect must be one of {sorted(DIALECT_VALUES)}")
    if emotion is not None:
        emotion = EMOTION_ALIASES.get(emotion.strip().lower(), emotion.strip())
        if emotion not in EMOTION_VALUES:
            raise ValueError(f"emotion must be one of {sorted(EMOTION_VALUES)}")
    if speed is not None and speed not in SPEED_VALUES:
        raise ValueError(f"speed must be one of {sorted(SPEED_VALUES)}")
    if pitch is not None and pitch not in PITCH_VALUES:
        raise ValueError(f"pitch must be one of {sorted(PITCH_VALUES)}")
    if volume is not None and volume not in VOLUME_VALUES:
        raise ValueError(f"volume must be one of {sorted(VOLUME_VALUES)}")
    if style is not None:
        style = _required_text(style, "style")
    if ip is not None:
        ip = _required_text(ip, "ip")

    item = {
        "序号": 1,
        "说话人": "speaker_1",
        "方言": dialect,
        "情感": emotion,
        "风格": style,
        "语速": speed,
        "基频": pitch,
        "音量": volume,
        "BGM": {name: None for name in BGM_FIELD_ORDER},
        "IP": ip,
    }
    if tuple(item) != CONTROL_FIELD_ORDER:
        raise AssertionError("structured-control field order changed")
    return json.dumps({"audio_sequence": [item]}, ensure_ascii=False)


def _validate_instruction_for_task(
    spec: TaskSpec,
    *,
    instruction: Optional[str],
    dialect: Optional[str],
    emotion: Optional[str],
    style: Optional[str],
    speed: Optional[str],
    pitch: Optional[str],
    volume: Optional[str],
    ip: Optional[str],
) -> Optional[str]:
    controls = {
        "dialect": dialect,
        "emotion": emotion,
        "style": style,
        "speed": speed,
        "pitch": pitch,
        "volume": volume,
        "ip": ip,
    }
    if spec.instruction_kind == "none":
        if instruction is not None or any(
            value is not None for value in controls.values()
        ):
            raise ValueError(f"task {spec.name} does not accept an instruction")
        return None
    if spec.instruction_kind == "freeform":
        if any(value is not None for value in controls.values()):
            raise ValueError("freeform accepts --instruction, not structured controls")
        instruction = _required_text(instruction, "instruction")
        if instruction.startswith("Speaking instruction:"):
            return instruction
        return f"Speaking instruction: Speak in this style: {instruction}"
    if instruction is not None:
        raise ValueError(
            f"task {spec.name} builds canonical structured JSON; do not pass raw instruction"
        )

    if spec.instruction_kind == "basic":
        if speed is None or pitch is None or volume is None:
            raise ValueError("basic requires speed, pitch, and volume")
        allowed = {"speed", "pitch", "volume"}
    else:
        required = spec.instruction_kind
        if controls[required] is None:
            raise ValueError(f"task {spec.name} requires {required}")
        allowed = {required}
    unexpected = [
        name
        for name, value in controls.items()
        if value is not None and name not in allowed
    ]
    if unexpected:
        raise ValueError(f"task {spec.name} received unrelated controls: {unexpected}")
    serialized = build_control_instruction(**controls)
    if spec.name != "emotion":
        return serialized

    payload = json.loads(serialized)
    item = payload["audio_sequence"][0]
    payload["audio_sequence"][0] = {
        name: item[name] for name in EMOTION_CONTROL_FIELD_ORDER
    }
    return json.dumps(payload, ensure_ascii=False)


def _one_path(value: object, name: str) -> str:
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{name} expects one path, not a list")
    return _required_text(value, name)


def _many_texts(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an ordered list")
    result = tuple(_required_text(item, name) for item in value)
    if len(result) < 2:
        raise ValueError(f"{name} must contain at least two entries")
    return result


def build_podcast_prompt_transcript(prompt_texts: Sequence[str]) -> str:
    texts = _many_texts(prompt_texts, "prompt_text")
    return "".join(
        f" speaker_{index}:{text}\n" for index, text in enumerate(texts, start=1)
    )


def validate_podcast_script(text: str, speaker_count: int) -> None:
    seen = set()
    for line in _required_text(text, "text").splitlines():
        if not line.strip():
            continue
        match = _PODCAST_SPEAKER_RE.match(line)
        if match is None:
            raise ValueError("every Podcast line must start with speaker_N:")
        seen.add(int(match.group(1)))
    expected = set(range(1, speaker_count + 1))
    if seen != expected:
        raise ValueError(
            "Podcast speaker IDs must exactly match prompt speakers 1.."
            f"{speaker_count}; found {sorted(seen)}"
        )


def canonicalize_podcast_script(text: str, speaker_count: int) -> str:
    """Return the exact multi-speaker serialization used in training."""

    validate_podcast_script(text, speaker_count)
    turns = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _PODCAST_SPEAKER_RE.match(line)
        if match is None:  # Kept explicit for type checkers and future edits.
            raise ValueError("every Podcast line must start with speaker_N:")
        turn_text = line[match.end() :].strip()
        if not turn_text:
            raise ValueError("Podcast turns must not be empty")
        turns.append(f" speaker_{int(match.group(1))}:{turn_text}\n")
    return "".join(turns)


def resolve_task_request(
    task: str,
    *,
    text: str,
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
) -> ResolvedTaskRequest:
    """Validate and lower a public request to the HF engine contract."""

    canonical_task = normalize_task(task)
    spec = TASK_SPECS[canonical_task]
    _required_text(text, "text")
    # Target text is validated stripped but forwarded as given, which is what
    # the published evaluation did; Podcast is canonicalized below.
    target_text = text
    resolved_instruction = _validate_instruction_for_task(
        spec,
        instruction=instruction,
        dialect=dialect,
        emotion=emotion,
        style=style,
        speed=speed,
        pitch=pitch,
        volume=volume,
        ip=ip,
    )

    resolved_wav: Optional[str | tuple[str, ...]]
    resolved_prompt_text: Optional[str]
    conditioning = spec.conditioning
    if spec.reference_count == "optional_one":
        # Freeform defaults to instruction-only generation. Keep explicit
        # speaker references available for existing callers.
        if prompt_text is not None:
            raise ValueError(f"task {canonical_task} does not use prompt_text")
        resolved_wav = None if prompt_wav is None else _one_path(prompt_wav, "prompt_wav")
        resolved_prompt_text = None
        if resolved_wav is not None:
            conditioning = "speaker_only"
    elif spec.reference_count == "none":
        if prompt_wav is not None or prompt_text is not None:
            raise ValueError(f"task {canonical_task} does not accept reference audio")
        resolved_wav = None
        resolved_prompt_text = None
    elif spec.reference_count == "one":
        resolved_wav = _one_path(prompt_wav, "prompt_wav")
        if spec.prompt_text_mode == "one":
            if isinstance(prompt_text, (list, tuple)):
                raise ValueError("prompt_text expects one transcript")
            resolved_prompt_text = _required_text(prompt_text, "prompt_text")
        else:
            if prompt_text is not None:
                raise ValueError(f"task {canonical_task} does not use prompt_text")
            resolved_prompt_text = None
    else:
        wavs = _many_texts(prompt_wav, "prompt_wav")
        texts = _many_texts(prompt_text, "prompt_text")
        if len(wavs) != len(texts):
            raise ValueError("Podcast needs one prompt_text per prompt_wav")
        target_text = canonicalize_podcast_script(target_text, len(wavs))
        resolved_wav = wavs
        resolved_prompt_text = build_podcast_prompt_transcript(texts)

    language = normalize_target_language(target_language)
    if canonical_task == "podcast" and language is not None:
        raise ValueError("Podcast infers Target language from its dialogue text")
    if spec.include_target_language:
        if spec.allowed_languages is not None and language is None:
            raise ValueError(f"task {canonical_task} requires target_language")
        if spec.allowed_languages and language not in spec.allowed_languages:
            raise ValueError(
                f"task {canonical_task} does not support target_language={language!r}; "
                f"expected one of {sorted(spec.allowed_languages)}"
            )
    elif language is not None:
        raise ValueError(f"task {canonical_task} must not add a Target language line")

    return ResolvedTaskRequest(
        task=canonical_task,
        text=target_text,
        instruction=resolved_instruction,
        prompt_wav=resolved_wav,
        prompt_text=resolved_prompt_text,
        target_language=language,
        include_target_language=spec.include_target_language,
        conditioning=conditioning,
    )


def podcast_frame_bounds_to_decode_steps(
    minimum_frames: int,
    maximum_frames: int,
    flow_patch_size: int,
) -> tuple[int, int]:
    for name, value in (
        ("minimum_frames", minimum_frames),
        ("maximum_frames", maximum_frames),
        ("flow_patch_size", flow_patch_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if not 8 <= minimum_frames <= maximum_frames <= 600:
        raise ValueError("Podcast frame bounds must satisfy 8 <= min <= max <= 600")
    if flow_patch_size < 1:
        raise ValueError("flow_patch_size must be positive")
    minimum_steps = minimum_frames // flow_patch_size
    maximum_steps = maximum_frames // flow_patch_size
    if maximum_steps < 1 or minimum_steps > maximum_steps:
        raise ValueError("Podcast frame bounds cannot be represented by this model")
    return minimum_steps, maximum_steps


def podcast_generation_frame_bounds(text: str) -> tuple[int, int]:
    """Derive the 12.5 Hz frame budget from the dialogue text."""

    minimum_frames = 0
    maximum_frames = 0
    for line in _required_text(text, "text").splitlines():
        if not line.strip():
            continue
        match = _PODCAST_SPEAKER_RE.match(line)
        if match is None:
            raise ValueError("every Podcast line must start with speaker_N:")
        turn_text = line[match.end() :].strip()
        if not turn_text:
            raise ValueError("Podcast turns must not be empty")
        if turn_text.isascii():
            units = len(turn_text.split())
            minimum_frames += units
            maximum_frames += units * 8
        else:
            units = len(turn_text)
            minimum_frames += units
            maximum_frames += units * 4
    maximum_frames = max(8, min(600, maximum_frames))
    minimum_frames = min(maximum_frames, max(8, minimum_frames))
    return minimum_frames, maximum_frames
