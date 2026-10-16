#!/usr/bin/env python3
"""Scan locale PO catalogs for common i18n quality issues."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from i18n_locales import KNOWN_LOCALES

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"

# Sphinx source docs are English; translated catalogs live under locale/.
SOURCE_LANGUAGE = "en"
SOURCE_LANGUAGE_NAME = "English"

LOCALE_DISPLAY_NAMES: dict[str, str] = {
    "de": "Deutsch",
    "es": "Español",
    "fr": "Français",
    "it": "Italiano",
    "ja": "日本語",
    "ko": "한국어",
    "pt_BR": "Português do Brasil",
    "zh_CN": "简体中文",
    "zh_TW": "繁體中文",
}

LOCALE_ENGLISH_NAMES: dict[str, str] = {
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt_BR": "Portuguese (Brazil)",
    "zh_CN": "Simplified Chinese",
    "zh_TW": "Traditional Chinese",
}

# Locales whose gettext Plural-Forms header correctly uses nplurals=1.
SINGLE_PLURAL_LOCALES = frozenset({"zh_CN", "zh_TW", "ja", "ko"})

# LLM reply preambles that may leak into msgstr values (language-agnostic).
COMMON_PROMPT_PATTERNS: tuple[str, ...] = (
    "here is the translation",
    "here's the translation",
    "translation of the provided",
    "below is the translation",
    "the translation is as follows",
)

# Locale-specific LLM reply preambles.
LOCALE_PROMPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "de": (
        "übersetzung:",
        "german translation",
        "deutsche übersetzung",
        "hier ist die übersetzung",
        "folgende übersetzung",
    ),
    "es": (
        "traducción solicitada",
        "traducción:",
        "aquí está la traducción",
        "la traducción es la siguiente",
        "spanish translation",
    ),
    "fr": (
        "voici la traduction",
        "french translation",
        "traduction demandée",
        "traduction :",
        "la traduction est la suivante",
    ),
    "it": (
        "traduzione richiesta",
        "traduzione:",
        "ecco la traduzione",
        "italian translation",
        "la traduzione è la seguente",
    ),
    "ja": (
        "以下は翻訳",
        "翻訳は以下",
        "日本語訳",
        "翻訳結果",
    ),
    "ko": (
        "다음은 번역",
        "번역입니다",
        "번역 결과",
        "한국어 번역",
        "아래는 번역",
    ),
    "pt_BR": (
        "tradução solicitada",
        "segue a tradução",
        "aqui está a tradução",
        "tradução:",
        "a tradução é a seguinte",
        "portuguese translation",
    ),
    "zh_CN": (
        "以下是翻译",
        "翻译如下",
        "中文翻译",
        "简体翻译",
    ),
    "zh_TW": (
        "以下是翻譯",
        "翻譯如下",
        "繁體中文翻譯",
        "正體中文翻譯",
    ),
}


def _prompt_patterns_for(locale: str) -> tuple[str, ...]:
    """Return prompt-leak patterns to check for a locale catalog."""
    patterns: list[str] = list(COMMON_PROMPT_PATTERNS)
    patterns.extend(LOCALE_PROMPT_PATTERNS.get(locale, ()))
    for other_locale, other_patterns in LOCALE_PROMPT_PATTERNS.items():
        if other_locale != locale:
            patterns.extend(other_patterns)
    return tuple(patterns)


def _locale_label(locale: str) -> str:
    native = LOCALE_DISPLAY_NAMES.get(locale)
    if not native:
        return locale
    encoding = sys.stdout.encoding or "utf-8"
    try:
        native.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        english = LOCALE_ENGLISH_NAMES.get(locale, locale)
        return f"{locale} ({english})"
    return f"{locale} ({native})"


def _parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    msgid: str | None = None
    msgstr: str | None = None
    in_msgid = in_msgstr = False

    def flush() -> None:
        nonlocal msgid, msgstr, in_msgid, in_msgstr
        if msgid is not None:
            pairs.append((msgid, msgstr or ""))
        msgid = msgstr = None
        in_msgid = in_msgstr = False

    for line in text.splitlines():
        if line.startswith("msgid "):
            flush()
            msgid = line[6:].strip().strip('"')
            in_msgid, in_msgstr = True, False
        elif line.startswith("msgstr "):
            msgstr = line[7:].strip().strip('"')
            in_msgid, in_msgstr = False, True
        elif line.startswith('"') and (in_msgid or in_msgstr):
            chunk = line.strip().strip('"')
            if in_msgid and msgid is not None:
                msgid += chunk
            elif in_msgstr and msgstr is not None:
                msgstr += chunk
        elif not line.startswith("#"):
            flush()
    flush()
    return pairs


def scan_locale(locale: str) -> dict[str, list[str]]:
    root = LOCALE_DIR / locale
    issues: dict[str, list[str]] = {
        "rest_colon": [],
        "rest_space_before_colons": [],
        "prompt_leak": [],
        "wrong_plural_header": [],
    }
    prompt_patterns = _prompt_patterns_for(locale)
    for po in sorted(root.rglob("*.po")):
        rel = str(po.relative_to(root))
        text = po.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        for pat in prompt_patterns:
            if pat in lower:
                issues["prompt_leak"].append(f"{rel}: matched {pat!r}")

        if (
            '"Plural-Forms: nplurals=1; plural=0;\\n"' in text
            and locale not in SINGLE_PLURAL_LOCALES
        ):
            issues["wrong_plural_header"].append(rel)

        for msgid, msgstr in _parse_pairs(text):
            if not msgstr:
                continue
            if msgid.rstrip().endswith("::") and "::" not in msgstr:
                issues["rest_colon"].append(f"{rel}: {msgstr[-60:]!r}")
            if re.search(r"[^:] ::", msgstr):
                issues["rest_space_before_colons"].append(f"{rel}: {msgstr[-60:]!r}")
    return issues


def main() -> int:
    supported = ", ".join(
        f"{loc} ({LOCALE_ENGLISH_NAMES[loc]})" for loc in KNOWN_LOCALES
    )
    parser = argparse.ArgumentParser(
        description=(
            "Scan Sphinx gettext PO catalogs for common translation quality "
            f"issues. Source language is {SOURCE_LANGUAGE_NAME} ({SOURCE_LANGUAGE}); "
            f"translated locales: {supported}."
        ),
    )
    parser.add_argument(
        "--locales",
        nargs="*",
        default=None,
        metavar="LOCALE",
        help=(
            "Locale directory names to scan (default: all known translated "
            f"locales: {', '.join(KNOWN_LOCALES)})"
        ),
    )
    args = parser.parse_args()
    locales = args.locales if args.locales is not None else list(KNOWN_LOCALES)

    exit_code = 0
    for locale in locales:
        if not (LOCALE_DIR / locale).is_dir():
            print(f"[skip] missing locale/{_locale_label(locale)}")
            continue
        print(f"\n=== locale/{_locale_label(locale)} ===")
        issues = scan_locale(locale)
        for kind, items in issues.items():
            print(f"{kind}: {len(items)}")
            for item in items[:8]:
                print(f"  - {item}")
            if len(items) > 8:
                print(f"  ... and {len(items) - 8} more")
            if items:
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
