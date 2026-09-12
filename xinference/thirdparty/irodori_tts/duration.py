from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import torch

ALLOWED_ANNOTATION_EMOJIS: tuple[str, ...] = (
    "⏩",
    "⏱️",
    "⏸️",
    "🌬️",
    "🍭",
    "🎛️",
    "🎭",
    "🎵",
    "🐢",
    "🐱",
    "👂",
    "👃",
    "👅",
    "👌",
    "👏",
    "💋",
    "💥",
    "💦",
    "💪",
    "📄",
    "📞",
    "📢",
    "📣",
    "😆",
    "😊",
    "😌",
    "😎",
    "😏",
    "😒",
    "😖",
    "😟",
    "😠",
    "😪",
    "😭",
    "😮",
    "😮‍💨",
    "😰",
    "😱",
    "😲",
    "😴",
    "🙄",
    "🙏",
    "🤐",
    "🤔",
    "🤢",
    "🤧",
    "🤭",
    "🥤",
    "🥱",
    "🥴",
    "🥵",
    "🥹",
    "🥺",
    "🫣",
    "🫶",
    "📖",
)

_ALLOWED_ANNOTATION_EMOJI_PATTERN = re.compile(
    "|".join(sorted((re.escape(x) for x in ALLOWED_ANNOTATION_EMOJIS), key=len, reverse=True))
)


def _log1p_cap(count: int, cap: int) -> float:
    return math.log1p(float(min(max(int(count), 0), int(cap)))) / math.log1p(float(cap))


def _log1p_cap_float(value: float, cap: float) -> float:
    value = min(max(float(value), 0.0), float(cap))
    return math.log1p(value) / math.log1p(float(cap))


def _is_kana(ch: str) -> bool:
    code = ord(ch)
    return (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)


def _is_kanji(ch: str) -> bool:
    code = ord(ch)
    return (
        (0x3400 <= code <= 0x4DBF)
        or (0x4E00 <= code <= 0x9FFF)
        or (0xF900 <= code <= 0xFAFF)
        or (0x20000 <= code <= 0x2FA1F)
    )


def _is_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def count_annotation_emojis(text: str) -> int:
    return len(_ALLOWED_ANNOTATION_EMOJI_PATTERN.findall(text))


def build_duration_features(
    texts: Sequence[str] | Iterable[str],
    *,
    token_counts: Sequence[int] | torch.Tensor,
    max_text_len: int,
    has_speaker: Sequence[bool] | torch.Tensor,
) -> torch.Tensor:
    text_list = [str(x) for x in texts]
    if isinstance(token_counts, torch.Tensor):
        token_count_list = [int(x) for x in token_counts.detach().cpu().tolist()]
    else:
        token_count_list = [int(x) for x in token_counts]
    if isinstance(has_speaker, torch.Tensor):
        has_speaker_list = [bool(x) for x in has_speaker.detach().cpu().tolist()]
    else:
        has_speaker_list = [bool(x) for x in has_speaker]

    if len(text_list) != len(token_count_list) or len(text_list) != len(has_speaker_list):
        raise ValueError(
            "Duration feature inputs must have matching lengths: "
            f"texts={len(text_list)} token_counts={len(token_count_list)} "
            f"has_speaker={len(has_speaker_list)}"
        )
    if max_text_len <= 0:
        raise ValueError(f"max_text_len must be > 0, got {max_text_len}")

    rows: list[list[float]] = []
    for text, token_count, speaker_available in zip(
        text_list, token_count_list, has_speaker_list, strict=True
    ):
        char_count = max(len(text), 1)
        kana_count = sum(1 for ch in text if _is_kana(ch))
        kanji_count = sum(1 for ch in text if _is_kanji(ch))
        alnum_count = sum(1 for ch in text if _is_alnum(ch))
        emoji_count = count_annotation_emojis(text)

        period_count = text.count("。") + text.count(".")
        comma_count = text.count("、") + text.count(",")
        long_vowel_count = text.count("ー")
        ellipsis_count = text.count("…")
        exclamation_count = text.count("！") + text.count("!")
        question_count = text.count("？") + text.count("?")

        rows.append(
            [
                min(max(float(token_count), 0.0), float(max_text_len)) / float(max_text_len),
                _log1p_cap_float(float(char_count), 512.0),
                float(token_count) / float(char_count),
                _log1p_cap(period_count, 8),
                _log1p_cap(comma_count, 16),
                _log1p_cap(long_vowel_count, 8),
                _log1p_cap(ellipsis_count, 8),
                _log1p_cap(exclamation_count, 8),
                _log1p_cap(question_count, 8),
                _log1p_cap(emoji_count, 8),
                float(kana_count) / float(char_count),
                float(kanji_count) / float(char_count),
                float(alnum_count) / float(char_count),
                1.0 if speaker_available else 0.0,
            ]
        )

    return torch.tensor(rows, dtype=torch.float32)


def set_duration_has_speaker_feature(
    features: torch.Tensor,
    has_speaker: torch.Tensor,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"Expected duration features shape (B, D), got {tuple(features.shape)}")
    if features.shape[1] <= 0:
        raise ValueError(
            f"duration features must have at least one column, got {features.shape[1]}"
        )
    out = features.clone()
    out[:, -1] = has_speaker.to(device=features.device, dtype=features.dtype)
    return out
