from __future__ import annotations

import re
import unicodedata

SIMPLE_REPLACE_MAP: dict[str, str] = {
    "\t": "",
    "[n]": "",
    r"\[n\]": "",
    "　": "",
    "？": "?",
    "！": "!",
    "♥": "♡",
    "●": "○",
    "◯": "○",
    "〇": "○",
}

REGEX_REPLACE_MAP = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]"): "",
    re.compile(r"[\uff5e\u301C]"): "ー",
    re.compile(r"…{3,}"): "……",
}


def strip_outer_brackets(text: str) -> str:
    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}

    while True:
        if len(text) < 2:
            break

        start_char = text[0]
        end_char = text[-1]

        if start_char in pairs and pairs[start_char] == end_char:
            depth = 0
            is_enclosing_all = True

            for i, char in enumerate(text):
                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1

                if depth == 0 and i < len(text) - 1:
                    is_enclosing_all = False
                    break

            if is_enclosing_all and depth == 0:
                text = text[1:-1]
                continue

        break

    return text


def normalize_text(text: str) -> str:
    for old, new in SIMPLE_REPLACE_MAP.items():
        text = text.replace(old, new)

    for pattern, replacement in REGEX_REPLACE_MAP.items():
        text = pattern.sub(replacement, text)

    text = strip_outer_brackets(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.replace("...", "…")
    text = text.replace("..", "…")

    return text
