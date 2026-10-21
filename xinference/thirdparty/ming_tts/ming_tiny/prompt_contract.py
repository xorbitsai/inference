"""Tokenizer and prompt contracts used by Ming-3.0-tts-tiny checkpoints."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional


BASE_PROTOCOL = "base_v1"
THINKING_V1_PROTOCOL = "thinking_v1"

# Published checkpoints predate the rename and still write the internal
# codename into config.json, so accept it wherever a protocol name is read.
_PROTOCOL_ALIASES = {"legacy_antmmf": BASE_PROTOCOL}


def _canonical_protocol(value: object) -> str:
    text = str(value)
    return _PROTOCOL_ALIASES.get(text, text)
NATIVE_FP32_SAMPLING = "native_fp32"
DEVICE_DTYPE_SAMPLING = "device_dtype"
OLD_SHIFT_DECODE_ALIGNMENT = "old_shift"
UNSHIFTED_DECODE_ALIGNMENT = "unshifted"

# Keep this order identical to the native training-side multimodal token list.
# The released checkpoint learned these exact embedding rows.
BASE_MULTIMODAL_TOKENS = (
    "<imagePatch>",
    "<image>",
    "</image>",
    "<video>",
    "</video>",
    "<gen_imagePatch>",
    "<gen_image>",
    "</gen_image>",
    "<imageHere>",
    "<end_of_chunk>",
    "<end_of_audio>",
    "<audioPatch>",
    "<audio>",
    "</audio>",
    "<gen_audioPatch>",
    "<gen_audio>",
    "</gen_audio>",
    "<audioHere>",
    "<framePatch>",
    "<text>",
    "<asr>",
    "<tts>",
    "<spk>",
    "</spk>",
)
BASE_TOKENIZER_SIZE = 157153
BASE_TOKEN_IDS = {
    token: BASE_TOKENIZER_SIZE + index
    for index, token in enumerate(BASE_MULTIMODAL_TOKENS)
}
BASE_TTS_TOKEN_IDS = {
    "<audioPatch>": 157164,
    "<audio>": 157165,
    "</audio>": 157166,
    "<spk>": 157175,
    "</spk>": 157176,
}

BASE_USER_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
    "  speaker_1:"
)
BASE_DESCRIPTION_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
BASE_USER_PREFIX_NO_RML = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
BASE_TEXT_PREFIX = "\n Text input:\n"
BASE_TEXT_PREFIX_NO_RML = " Text input:\n"
BASE_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"

THINKING_SYSTEM_PREFIX = "<role>SYSTEM</role>detailed thinking off"
THINKING_USER_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
    "  speaker_1:"
)
THINKING_DESCRIPTION_PREFIX = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
THINKING_USER_PREFIX_NO_RML = (
    "<role>HUMAN</role>"
    "Please generate speech based on the following description.\n"
)
THINKING_TEXT_PREFIX = "\n Text input:\n"
THINKING_TEXT_PREFIX_NO_RML = " Text input:\n"
THINKING_ASSISTANT_PREFIX = "<role>ASSISTANT</role>\n<think></think>"

_CHINESE_TEXT_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TTS_INLINE_CONTROL_PATTERN = re.compile(
    r"\[[^\]\n]{1,128}\]|\([A-Za-z][A-Za-z _-]{0,127}\)"
)
_TTS_SPEAKER_LABEL_PATTERN = re.compile(
    r"(?im)^\s*speaker[_ ]?\d+\s*[:\uff1a]\s*"
)

TARGET_LANGUAGE_NAMES = {
    "arabic": "Arabic",
    "canton": "Canton",
    "cantonese": "Canton",
    "chinese": "Chinese",
    "czech": "Czech",
    "dutch": "Dutch",
    "english": "English",
    "finnish": "Finnish",
    "french": "French",
    "german": "German",
    "greek": "Greek",
    "hindi": "Hindi",
    "indonesian": "Indonesian",
    "italian": "Italian",
    "japanese": "Japanese",
    "korean": "Korean",
    "polish": "Polish",
    "portuguese": "Portuguese",
    "romanian": "Romanian",
    "russian": "Russian",
    "spanish": "Spanish",
    "thai": "Thai",
    "turkish": "Turkish",
    "ukrainian": "Ukrainian",
    "vietnamese": "Vietnamese",
    "ar": "Arabic",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hk": "Canton",
}


def normalize_target_language(target_language: str | None) -> str | None:
    """Return the training prompt's canonical target-language name."""
    if target_language is None:
        return None
    if not isinstance(target_language, str) or not target_language.strip():
        raise ValueError("target language must be a non-empty string")
    key = target_language.strip().lower().replace("_", "-")
    language = TARGET_LANGUAGE_NAMES.get(key)
    if language is None:
        raise ValueError(f"unsupported target language: {target_language!r}")
    return language


def _target_language_line(
    target_text: str,
    tts_config: Mapping[str, object],
    target_language: str | None = None,
    include_target_language: bool | None = None,
) -> str:
    """Build the SFT target-language suffix from metadata or target text.

    ``include_target_language`` is a per-sample override for mixed checkpoints
    whose training groups used different SampleBuilder language policies. A
    true override still requires the converted bundle to advertise support;
    false only suppresses the line for the selected task.
    """
    enabled = tts_config.get("enable_tts_target_language", False)
    infer_from_text = tts_config.get("infer_tts_language_from_text", False)
    if not isinstance(enabled, bool) or not isinstance(infer_from_text, bool):
        raise TypeError("TTS target-language flags must be booleans")
    if include_target_language is not None and not isinstance(
        include_target_language, bool
    ):
        raise TypeError("include_target_language must be a boolean or None")
    if include_target_language is True and not enabled:
        raise ValueError(
            "per-sample target-language inclusion requires a converted "
            "bundle with enable_tts_target_language=true"
        )
    if include_target_language is False:
        return ""
    if not enabled:
        return ""
    explicit_language = normalize_target_language(target_language)
    if explicit_language is not None:
        return f"Target language: {explicit_language}.\n"
    if not infer_from_text:
        raise ValueError(
            "Ming-3.0-tts-tiny SFT target-language prompts require "
            "infer_tts_language_from_text=true"
        )

    lexical_text = _TTS_SPEAKER_LABEL_PATTERN.sub("", str(target_text))
    lexical_text = _TTS_INLINE_CONTROL_PATTERN.sub("", lexical_text)
    # Frozen old-SFT2 uses an any-Han rule after removing control/role text;
    # a mixed Chinese sentence must not flip to English merely because it
    # contains more Latin words.
    language = (
        "Chinese" if _CHINESE_TEXT_PATTERN.search(lexical_text) else "English"
    )
    return f"Target language: {language}.\n"


def protocol_version(tts_config: Mapping[str, object]) -> str:
    value = tts_config.get("protocol_version")
    if value is None:
        value = (
            THINKING_V1_PROTOCOL
            if tts_config.get("audio_patch_token_id") == 157168
            else BASE_PROTOCOL
        )
    value = _canonical_protocol(value)
    if value not in {BASE_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            f"Unsupported Ming-3.0-tts-tiny protocol_version={value!r}; expected "
            f"{BASE_PROTOCOL!r} or {THINKING_V1_PROTOCOL!r}"
        )
    return value


def prompt_format(tts_config: Mapping[str, object]) -> str:
    """Return the chat/prompt layout independently of token row names.

    Older bundles used ``protocol_version`` for both dimensions. Newer
    Megatron tasks can deliberately use the Bailing V3 runtime while their
    tokenizer and prompt retain the base multimodal rows.
    """

    value = _canonical_protocol(
        tts_config.get("prompt_format", protocol_version(tts_config))
    )
    if value not in {BASE_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            f"Unsupported Ming-3.0-tts-tiny prompt_format={value!r}; expected "
            f"{BASE_PROTOCOL!r} or {THINKING_V1_PROTOCOL!r}"
        )
    return value


def tokenizer_contract(tts_config: Mapping[str, object]) -> str:
    """Return the multimodal token vocabulary used during training."""

    value = _canonical_protocol(
        tts_config.get("tokenizer_contract", protocol_version(tts_config))
    )
    if value not in {BASE_PROTOCOL, THINKING_V1_PROTOCOL}:
        raise ValueError(
            "Unsupported Ming-3.0-tts-tiny tokenizer_contract="
            f"{value!r}; expected {BASE_PROTOCOL!r} or "
            f"{THINKING_V1_PROTOCOL!r}"
        )
    return value


def sampling_contract(tts_config: Mapping[str, object]) -> str:
    """Return the Flow RNG/dtype contract independently of prompt layout.

    Old bundles overloaded ``protocol_version`` for this choice. A mixed
    Megatron contract can use the Bailing V3 cache/stop order while the rc1.6
    Flow sampler still creates fp32 history and CPU fp32 initial noise with
    the global RNG streams.
    """

    default = (
        NATIVE_FP32_SAMPLING
        if protocol_version(tts_config) == BASE_PROTOCOL
        else DEVICE_DTYPE_SAMPLING
    )
    value = str(tts_config.get("sampling_contract", default))
    if value not in {NATIVE_FP32_SAMPLING, DEVICE_DTYPE_SAMPLING}:
        raise ValueError(
            "Unsupported Ming-3.0-tts-tiny sampling_contract="
            f"{value!r}; expected {NATIVE_FP32_SAMPLING!r} or "
            f"{DEVICE_DTYPE_SAMPLING!r}"
        )
    return value


def decode_alignment(tts_config: Mapping[str, object]) -> str:
    """Return the training label shift used by autoregressive TTS decode.

    Historical bundles coupled this choice to ``protocol_version``.  Newer
    Megatron jobs may use the modern cache/runtime protocol while retaining
    ``shift_tts_labels_like_old=true``.  Keep the explicit dimension optional
    so existing converted bundles retain their exact behavior.
    """

    default = (
        OLD_SHIFT_DECODE_ALIGNMENT
        if protocol_version(tts_config) == BASE_PROTOCOL
        else UNSHIFTED_DECODE_ALIGNMENT
    )
    value = str(tts_config.get("decode_alignment", default))
    if value not in {
        OLD_SHIFT_DECODE_ALIGNMENT,
        UNSHIFTED_DECODE_ALIGNMENT,
    }:
        raise ValueError(
            "Unsupported Ming-3.0-tts-tiny decode_alignment="
            f"{value!r}; expected {OLD_SHIFT_DECODE_ALIGNMENT!r} or "
            f"{UNSHIFTED_DECODE_ALIGNMENT!r}"
        )
    return value


def task_decode_alignments(
    tts_config: Mapping[str, object],
) -> Dict[str, str]:
    """Validate task-specific label-shift provenance from conversion.

    Multi-task checkpoints may contain training groups with a different label
    shift.  The choice must be written into the converted bundle; callers are
    not allowed to infer it from an evaluation dataset at runtime.
    """

    raw = tts_config.get("task_decode_alignment", {})
    if raw in (None, {}):
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("task_decode_alignment must be a JSON object")
    result: Dict[str, str] = {}
    for raw_task, raw_alignment in raw.items():
        task = str(raw_task).strip().lower().replace("-", "_")
        if not task:
            raise ValueError("task_decode_alignment contains an empty task")
        alignment = str(raw_alignment).strip()
        if alignment not in {
            OLD_SHIFT_DECODE_ALIGNMENT,
            UNSHIFTED_DECODE_ALIGNMENT,
        }:
            raise ValueError(
                "Unsupported task decode alignment for "
                f"{task!r}: {alignment!r}"
            )
        if task in result:
            raise ValueError(
                f"task_decode_alignment contains duplicate task {task!r}"
            )
        result[task] = alignment
    return result


def with_rml(tts_config: Mapping[str, object]) -> bool:
    """Return whether the training prompt contains speaker RML.

    Historical bundles predate this explicit field and all used
    ``with_RML=True``.  Defaulting to ``True`` therefore preserves their
    prompt and speaker-injection behavior byte-for-byte.  A no-RML bundle must
    opt in explicitly with the JSON boolean ``with_rml=false``.
    """

    value = tts_config.get("with_rml", True)
    if not isinstance(value, bool):
        raise ValueError(
            f"TTS with_rml must be a JSON boolean, got {value!r}"
        )
    return value


def encode(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def ensure_base_tokenizer(tokenizer) -> int:
    """Add the native multimodal tokens in their training-time order.

    A later Ming-3.0-tts-tiny tokenizer cannot be repaired by appending aliases: its
    occupied embedding rows already have different meanings.  Fail with an
    actionable error in that case.
    """

    actual = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in BASE_MULTIMODAL_TOKENS
    }
    if actual == BASE_TOKEN_IDS:
        return 0

    absent = all(token_id is None for token_id in actual.values())
    if absent and len(tokenizer) == BASE_TOKENIZER_SIZE:
        added = tokenizer.add_tokens(
            list(BASE_MULTIMODAL_TOKENS), special_tokens=True
        )
        if added != len(BASE_MULTIMODAL_TOKENS):
            raise RuntimeError(
                "Failed to add the complete native multimodal token list: "
                f"added {added}/{len(BASE_MULTIMODAL_TOKENS)}"
            )
        repaired = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in BASE_MULTIMODAL_TOKENS
        }
        if repaired != BASE_TOKEN_IDS:
            raise RuntimeError(
                "Tokenizer assigned unexpected native token IDs: "
                f"{repaired}"
            )
        return added

    mismatches = {
        token: (BASE_TOKEN_IDS[token], actual[token])
        for token in BASE_MULTIMODAL_TOKENS
        if actual[token] != BASE_TOKEN_IDS[token]
    }
    preview = ", ".join(
        f"{token}: expected {want}, got {got}"
        for token, (want, got) in list(mismatches.items())[:8]
    )
    raise ValueError(
        "This tokenizer does not have the base multimodal token rows and cannot "
        "be repaired by appending tokens. Load the base tokenizer from "
        "the checkpoint's training-side for_infer directory (or convert the "
        "model with --tokenizer-dir). "
        + preview
    )


def _expected_tts_tokens(
    tts_config: Mapping[str, object],
) -> Dict[str, int]:
    if tokenizer_contract(tts_config) == BASE_PROTOCOL:
        token_names = {
            "<audioPatch>": "audio_patch_token_id",
            "<audio>": "audio_start_token_id",
            "</audio>": "audio_end_token_id",
            "<spk>": "speaker_start_token_id",
            "</spk>": "speaker_end_token_id",
        }
    else:
        token_names = {
            "<|audio_pad|>": "audio_patch_token_id",
            "<|audio_start|>": "audio_start_token_id",
            "<|audio_end|>": "audio_end_token_id",
            "<spk>": "speaker_start_token_id",
            "</spk>": "speaker_end_token_id",
        }
    return {
        token: int(tts_config[config_name])
        for token, config_name in token_names.items()
    }


def validate_tts_tokenizer(
    tokenizer,
    tts_config: Mapping[str, object],
    config_eos,
) -> Dict[str, int]:
    expected = _expected_tts_tokens(tts_config)
    actual = {
        token: tokenizer.convert_tokens_to_ids(token) for token in expected
    }
    mismatches = {
        token: (expected[token], actual[token])
        for token in expected
        if actual[token] != expected[token]
    }
    if mismatches:
        details = ", ".join(
            f"{token}: expected {want}, got {got}"
            for token, (want, got) in mismatches.items()
        )
        raise ValueError(
            "Tokenizer IDs do not match Ming-3.0-tts-tiny training: " + details
        )
    for token, token_id in expected.items():
        encoded = encode(tokenizer, token)
        if encoded != [token_id]:
            raise ValueError(
                f"{token!r} must encode to [{token_id}], got {encoded}"
            )

    config_eos_ids = (
        set(config_eos)
        if isinstance(config_eos, (list, tuple, set))
        else {config_eos}
    )
    if tokenizer.eos_token_id not in config_eos_ids:
        raise ValueError(
            "Tokenizer EOS does not match the Ming-3.0-tts-tiny config: "
            f"{tokenizer.eos_token_id} not in {sorted(config_eos_ids)}"
        )
    expected_prompt_eos = tts_config.get("prompt_eos_token_id")
    if (
        expected_prompt_eos is not None
        and tokenizer.eos_token_id != int(expected_prompt_eos)
    ):
        raise ValueError(
            "Tokenizer EOS does not match the training prompt separator: "
            f"expected {int(expected_prompt_eos)}, got "
            f"{tokenizer.eos_token_id}"
        )
    return actual


def _build_base_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
    speaker_slot: bool | None = None,
    assistant_instruction: bool = False,
    target_language: str | None = None,
    include_target_language: bool | None = None,
) -> tuple[List[int], Optional[int]]:
    # Exact sample_instruct_tts(with_RML=True, RML_length=1,
    # with_caption=True, caption="") sequence.  There is deliberately no
    # SYSTEM prefix, EOS separator, or <think></think>.
    rml_enabled = (
        with_rml(tts_config) if speaker_slot is None else bool(speaker_slot)
    )
    description_prefix = (
        BASE_DESCRIPTION_PREFIX
        + _target_language_line(
            target_text,
            tts_config,
            target_language=target_language,
            include_target_language=include_target_language,
        )
    )
    if instruction is not None and not assistant_instruction:
        user_prefix = description_prefix + instruction + "\n"
        if rml_enabled:
            user_prefix += "  speaker_1:"
    else:
        user_prefix = description_prefix + ("  speaker_1:" if rml_enabled else "")
    ids = encode(tokenizer, user_prefix)
    speaker_embedding_position: Optional[int] = None
    if rml_enabled:
        ids.append(int(tts_config["speaker_start_token_id"]))
        speaker_embedding_position = len(ids)
        ids.append(int(tts_config["audio_patch_token_id"]))
        ids.append(int(tts_config["speaker_end_token_id"]))
    ids.extend(
        encode(
            tokenizer,
            BASE_TEXT_PREFIX if rml_enabled else BASE_TEXT_PREFIX_NO_RML,
        )
    )
    if prompt_text is not None:
        ids.extend(encode(tokenizer, prompt_text))
    ids.extend(encode(tokenizer, target_text))
    ids.extend(encode(tokenizer, BASE_ASSISTANT_PREFIX))
    if instruction is not None and assistant_instruction:
        ids.extend(encode(tokenizer, instruction))
        ids.append(tokenizer.eos_token_id)
    ids.append(int(tts_config["audio_start_token_id"]))
    ids.extend(
        [int(tts_config["audio_patch_token_id"])]
        * prompt_latent_tokens
    )
    return ids, speaker_embedding_position


def build_multispeaker_tts_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    speaker_count: int,
    prompt_text: str,
    prompt_latent_tokens: int,
    target_language: str | None = None,
    include_target_language: bool | None = None,
) -> tuple[List[int], List[int]]:
    """Build the published MoE Podcast continuation prompt for N speakers.

    Each prompt speaker has one ordered RML placeholder.  The combined prompt
    transcript precedes the target dialogue, and the projected AudioVAE
    continuation occupies the final audio-patch placeholders.
    """

    if prompt_format(tts_config) != BASE_PROTOCOL:
        raise ValueError(
            f"Podcast currently requires prompt_format={BASE_PROTOCOL}"
        )
    if not isinstance(speaker_count, int) or isinstance(speaker_count, bool):
        raise TypeError("speaker_count must be an integer")
    if speaker_count < 2:
        raise ValueError("Podcast requires at least two prompt speakers")
    if not target_text or not str(target_text).strip():
        raise ValueError("Podcast target_text must not be empty")
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("Podcast prompt_text must be non-empty text")
    if (
        not isinstance(prompt_latent_tokens, int)
        or isinstance(prompt_latent_tokens, bool)
        or prompt_latent_tokens < 1
    ):
        raise ValueError("Podcast prompt_latent_tokens must be positive")
    if not with_rml(tts_config):
        raise ValueError("Podcast requires a with_rml=true model bundle")

    ids = encode(
        tokenizer,
        BASE_DESCRIPTION_PREFIX
        + _target_language_line(
            target_text,
            tts_config,
            target_language=target_language,
            include_target_language=include_target_language,
        ),
    )
    speaker_positions: List[int] = []
    for index in range(1, speaker_count + 1):
        prefix = ("" if index == 1 else "\n") + f"  speaker_{index}:"
        ids.extend(encode(tokenizer, prefix))
        ids.append(int(tts_config["speaker_start_token_id"]))
        speaker_positions.append(len(ids))
        ids.append(int(tts_config["audio_patch_token_id"]))
        ids.append(int(tts_config["speaker_end_token_id"]))
    ids.extend(encode(tokenizer, BASE_TEXT_PREFIX))
    ids.extend(encode(tokenizer, prompt_text))
    ids.extend(encode(tokenizer, str(target_text)))
    ids.extend(encode(tokenizer, BASE_ASSISTANT_PREFIX))
    ids.append(int(tts_config["audio_start_token_id"]))
    ids.extend(
        [int(tts_config["audio_patch_token_id"])]
        * prompt_latent_tokens
    )
    return ids, speaker_positions


def _build_thinking_v1_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
    speaker_slot: bool | None = None,
    assistant_instruction: bool = False,
    target_language: str | None = None,
    include_target_language: bool | None = None,
) -> tuple[List[int], Optional[int]]:
    ids = encode(tokenizer, THINKING_SYSTEM_PREFIX)
    ids.append(tokenizer.eos_token_id)
    rml_enabled = (
        with_rml(tts_config) if speaker_slot is None else bool(speaker_slot)
    )
    description_prefix = (
        THINKING_DESCRIPTION_PREFIX
        + _target_language_line(
            target_text,
            tts_config,
            target_language=target_language,
            include_target_language=include_target_language,
        )
    )
    if instruction is not None and not assistant_instruction:
        user_prefix = description_prefix + instruction + "\n"
        if rml_enabled:
            user_prefix += "  speaker_1:"
    else:
        user_prefix = description_prefix + ("  speaker_1:" if rml_enabled else "")
    ids.extend(encode(tokenizer, user_prefix))
    speaker_embedding_position: Optional[int] = None
    if rml_enabled:
        ids.append(int(tts_config["speaker_start_token_id"]))
        speaker_embedding_position = len(ids)
        ids.append(int(tts_config["audio_patch_token_id"]))
        ids.append(int(tts_config["speaker_end_token_id"]))
    ids.extend(
        encode(
            tokenizer,
            THINKING_TEXT_PREFIX if rml_enabled else THINKING_TEXT_PREFIX_NO_RML,
        )
    )
    if prompt_text is not None:
        ids.extend(encode(tokenizer, prompt_text))
    ids.extend(encode(tokenizer, target_text))
    ids.append(tokenizer.eos_token_id)
    ids.extend(encode(tokenizer, THINKING_ASSISTANT_PREFIX))
    if instruction is not None and assistant_instruction:
        ids.extend(encode(tokenizer, instruction))
        ids.append(tokenizer.eos_token_id)
    ids.append(int(tts_config["audio_start_token_id"]))
    ids.extend(
        [int(tts_config["audio_patch_token_id"])]
        * prompt_latent_tokens
    )
    return ids, speaker_embedding_position


def build_tts_prompt_ids(
    tokenizer,
    target_text: str,
    tts_config: Mapping[str, object],
    *,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_latent_tokens: int = 0,
    speaker_slot: bool | None = None,
    assistant_instruction: bool = False,
    target_language: str | None = None,
    include_target_language: bool | None = None,
) -> tuple[List[int], Optional[int]]:
    """Build the LingV2-compatible zero-shot prompt token sequence.

    When reference conditioning is present, the reference transcript is
    placed immediately before the target transcript and one audio placeholder
    is appended for every projected prompt-latent patch.  Passing neither
    keeps the historical speaker-only/no-reference prompt byte-for-byte.
    ``speaker_slot`` lets a mixed SFT checkpoint select the training contract
    per task: speaker-conditioned and muted-IP rows retain one RML slot,
    while Voice Design omits the complete speaker line.  ``None`` preserves
    the converted bundle's historical global ``with_rml`` default.

    Latest SFT training places an optional instruction on the assistant side,
    after the assistant chat prefix and before ``<audio>``.  Select that
    behavior explicitly with ``assistant_instruction=True``; the default
    preserves historical bundles that placed instructions in the user
    description.  The terminating tokenizer EOS has loss mask zero during
    SFT training and is therefore part of inference context as well.
    NV-Bench control tags remain ordinary target-text BPE; this function never
    adds task-specific special tokens.
    """

    if instruction is not None:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
    if prompt_latent_tokens < 0:
        raise ValueError("prompt_latent_tokens must be non-negative")
    has_prompt_text = prompt_text is not None
    has_prompt_latent = prompt_latent_tokens > 0
    if has_prompt_text != has_prompt_latent:
        raise ValueError(
            "prompt_text and prompt latent tokens must be provided together"
        )
    if prompt_text == "":
        raise ValueError("prompt_text must not be empty")

    if prompt_format(tts_config) == BASE_PROTOCOL:
        return _build_base_prompt_ids(
            tokenizer,
            target_text,
            tts_config,
            instruction=instruction,
            prompt_text=prompt_text,
            prompt_latent_tokens=prompt_latent_tokens,
            speaker_slot=speaker_slot,
            assistant_instruction=assistant_instruction,
            target_language=target_language,
            include_target_language=include_target_language,
        )
    return _build_thinking_v1_prompt_ids(
        tokenizer,
        target_text,
        tts_config,
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_latent_tokens=prompt_latent_tokens,
        speaker_slot=speaker_slot,
        assistant_instruction=assistant_instruction,
        target_language=target_language,
        include_target_language=include_target_language,
    )
