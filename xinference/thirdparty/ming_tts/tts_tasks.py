# Copyright (c) Ant Group. All rights reserved.
"""Task and prompt contracts for Ming-3.0-tts.

The strings and JSON key order in this module are part of the model input
protocol: the released checkpoint was trained against these exact captions.
Import the builders from here instead of re-assembling prompts per script.
"""

from __future__ import annotations

import json
import warnings
from collections import OrderedDict

# Inline paralinguistic tags live with the text frontend because every
# cleaning step has to route around them; re-exported here as part of the
# task-level vocabulary.
from .frontend.toolkit import PARALINGUISTIC_TAGS


DEFAULT_TTS_PROMPT = "Please generate speech based on the following description.\n"

# Language IDs accepted by ``language_id``. Short codes and full names both
# resolve to the name used in the ``Target language:`` line.
_TARGET_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    # Cantonese is a Chinese dialect, not a separate target language.
    "yue": "Chinese",
    "zh-yue": "Chinese",
    "cantonese": "Chinese",
}

# The five languages added in Ming-3.0 on top of Chinese/English. These
# require an explicit ``language_id``.
MULTILINGUAL_LANGUAGE_IDS = {
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
}

BASIC_ATTRIBUTE_VALUES = {
    "speed": ("慢速", "中速", "快速"),
    "pitch": ("低", "中", "高"),
    "volume": ("低", "中", "高"),
}

EMOTION_VALUES = ("高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性")

DIALECT_VALUES = ("chuanyu dialect", "cantonese")

# English premium built-in voices. Chinese IP voices are listed in
# cookbooks/instructions.md.
EN_IP_VOICES = ("US_Emma", "US_James", "UK_George", "UK_Charlotte")

_ZH_FIELDS = ("序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感")
_EN_FIELDS = ("id", "speaker", "dialect", "style", "speed", "pitch", "volume", "emotion")
_BGM_FIELDS = ("Genre", "Mood", "Instrument", "Theme", "ENV", "SNR")

_TASK_ALIASES = {
    "TTS": "TTS",
    "SEED": "TTS",
    "PLAIN": "TTS",
    "MULTILINGUAL": "TTS",
    "MULTILINGUAL_TTS": "TTS",
    "BASIC": "BASIC",
    "EMOTION": "EMOTION",
    "DIALECT": "DIALECT",
    "IP": "IP",
    "VOICE_DESIGN": "STYLE",
    "STYLE": "STYLE",
    "FREEFORM": "STYLE",
}
_STRUCTURED_TASKS = {
    "BASIC", "EMOTION", "DIALECT", "IP", "VOICE_DESIGN", "STYLE",
}

TASKS = tuple(_TASK_ALIASES)


def build_tts_prompt(target_language=None):
    """Return the user-side prompt; the target language line is opt-in."""
    if target_language is None or not str(target_language).strip():
        return DEFAULT_TTS_PROMPT
    language = str(target_language).strip()
    language_name = _TARGET_LANGUAGE_NAMES.get(language.lower(), language)
    return DEFAULT_TTS_PROMPT + f"Target language: {language_name}.\n"


def normalize_tts_prompt(prompt):
    """Treat a missing/blank prompt as the canonical generation prompt."""
    if prompt is None or not str(prompt).strip():
        return DEFAULT_TTS_PROMPT
    return str(prompt)


def normalize_instruction(instruction):
    """Blank means no instruction, not an empty assistant EOS prefix."""
    if instruction is None or not str(instruction).strip():
        return None
    return str(instruction).strip()


def preserve_multilingual_text(language_id):
    """Return true when target text must bypass Chinese/English TN.

    The TN frontend treats any CJK ideograph as Chinese, so Japanese
    containing kanji would lose its kana. Explicit non-zh/en languages are
    therefore passed through unchanged.
    """
    if language_id is None or not str(language_id).strip():
        return False
    language = str(language_id).strip().lower().replace("_", "-")
    return language not in {
        "zh", "zh-cn", "chinese", "en", "english",
    }


def build_structured_instruction(
    *, dialect=None, style=None, speed=None, pitch=None, volume=None,
    emotion=None, ip=None, schema="zh",
):
    """Build the ordered assistant-side caption consumed during SFT."""
    schema = str(schema).strip().lower()
    values = (1, "speaker_1", dialect, style, speed, pitch, volume, emotion)
    if schema in {"zh", "chinese"}:
        fields, ip_key = _ZH_FIELDS, "IP"
    elif schema in {"en", "english"}:
        fields, ip_key = _EN_FIELDS, "ip"
    else:
        raise ValueError(f"Unsupported structured instruction schema: {schema!r}")

    item = OrderedDict(zip(fields, values))
    item["BGM"] = OrderedDict((key, None) for key in _BGM_FIELDS)
    item[ip_key] = ip
    payload = OrderedDict((("audio_sequence", [item]),))
    return json.dumps(payload, ensure_ascii=False)


def build_task_instruction(task, *, schema=None, **controls):
    """Build a task caption while preventing unrelated controls from leaking."""
    task = str(task).strip().upper()
    allowed = {
        "BASIC": {"speed", "pitch", "volume"},
        "DIALECT": {"dialect"},
        "EMOTION": {"emotion"},
        "VOICE_DESIGN": {"style"},
        "STYLE": {"style"},
        "IP": {"ip"},
    }
    if task not in allowed:
        raise ValueError(f"Unsupported TTS control task: {task!r}")
    unknown = set(controls) - allowed[task]
    if unknown:
        raise ValueError(f"{task} does not accept control fields: {sorted(unknown)!r}")
    if schema is None:
        schema = "en" if task == "DIALECT" else "zh"
    return build_structured_instruction(schema=schema, **controls)


def build_task_generation_kwargs(
    task,
    *,
    text,
    language_id=None,
    target_language=None,
    instruction=None,
    controls=None,
    instruction_schema=None,
    prompt_wav_path=None,
    prompt_text=None,
    use_spk_emb=False,
    use_zero_spk_emb=False,
    cfg=2.0,
    sigma=0.25,
    temperature=0.0,
    **kwargs,
):
    """Resolve a task request into the fields consumed by the model."""
    task = str(task).strip().upper().replace("-", "_")
    try:
        runtime_task = _TASK_ALIASES[task]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Ming TTS task {task!r}; expected one of "
            f"{sorted(_TASK_ALIASES)}") from error

    if task == "DIALECT" and any(
            value is not None and str(value).strip()
            for value in (language_id, target_language)):
        warnings.warn(
            "DIALECT does not use a language ID; ignoring language_id/"
            "target_language",
            UserWarning,
            stacklevel=2,
        )
        language_id = None
        target_language = None
    if language_id is not None and target_language is not None:
        if str(language_id).strip().lower() != \
                str(target_language).strip().lower():
            raise ValueError(
                "language_id and target_language must not conflict")
    language_id = (
        language_id if language_id is not None else target_language)
    if task in {"MULTILINGUAL", "MULTILINGUAL_TTS"} and (
            language_id is None or not str(language_id).strip()):
        raise ValueError("MULTILINGUAL requires language_id")

    text = str(text).strip()
    if not text:
        raise ValueError("text must not be empty")
    if use_spk_emb and use_zero_spk_emb:
        raise ValueError(
            "use_spk_emb and use_zero_spk_emb cannot both be true")
    if use_spk_emb and not (prompt_wav_path or kwargs.get("spk_emb_path")
                            or kwargs.get("speaker_embedding_path")):
        raise ValueError(
            "use_spk_emb requires a prompt wav or a speaker embedding")

    controls = dict(controls or {})
    if task == "FREEFORM":
        if controls:
            raise ValueError("FREEFORM uses instruction, not controls")
        task_instruction = normalize_instruction(instruction)
        if task_instruction is None:
            raise ValueError("FREEFORM requires a non-empty instruction")
    elif task in _STRUCTURED_TASKS:
        if instruction is not None:
            raise ValueError(
                f"{task} builds its instruction from controls; do not pass "
                "instruction")
        if not controls:
            raise ValueError(f"{task} requires task controls")
        prompt_task = "VOICE_DESIGN" if task == "STYLE" else task
        task_instruction = build_task_instruction(
            prompt_task, schema=instruction_schema, **controls)
    else:
        if controls:
            raise ValueError(f"{task} does not accept controls")
        task_instruction = normalize_instruction(instruction)

    request = {
        "prompt": build_tts_prompt(language_id),
        "text": text,
        "use_spk_emb": bool(use_spk_emb),
        "use_zero_spk_emb": bool(use_zero_spk_emb),
        "instruction": task_instruction,
        "prompt_wav_path": prompt_wav_path,
        "prompt_text": prompt_text,
        "cfg": float(cfg),
        "sigma": float(sigma),
        "temperature": float(temperature),
        "taskname": runtime_task,
        "text_language": language_id,
    }
    reserved = set(request).intersection(kwargs)
    if reserved:
        raise ValueError(
            f"generation kwargs cannot override task fields: {sorted(reserved)}")
    request.update(kwargs)
    return request


__all__ = [
    "BASIC_ATTRIBUTE_VALUES",
    "DEFAULT_TTS_PROMPT",
    "DIALECT_VALUES",
    "EMOTION_VALUES",
    "EN_IP_VOICES",
    "MULTILINGUAL_LANGUAGE_IDS",
    "PARALINGUISTIC_TAGS",
    "TASKS",
    "build_structured_instruction",
    "build_task_generation_kwargs",
    "build_task_instruction",
    "build_tts_prompt",
    "normalize_instruction",
    "normalize_tts_prompt",
    "preserve_multilingual_text",
]
