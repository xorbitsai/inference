#!/usr/bin/env python3
"""Scan locale PO catalogs for common i18n quality issues."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"

PROMPT_PATTERNS = (
    "voici la traduction",
    "french translation",
    "traduction demandée",
    "here is the translation",
    "translation of the provided",
    "以下是翻译",
    "翻译如下",
    "以下は翻訳",
    "traducción solicitada",
)


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
    for po in sorted(root.rglob("*.po")):
        rel = str(po.relative_to(root))
        text = po.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        for pat in PROMPT_PATTERNS:
            if pat in lower:
                issues["prompt_leak"].append(f"{rel}: matched {pat!r}")

        if '"Plural-Forms: nplurals=1; plural=0;\\n"' in text and locale not in {
            "zh_CN",
            "zh_TW",
            "ja",
            "ko",
        }:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locales",
        nargs="*",
        default=["es", "ko", "zh_CN", "zh_TW"],
        help="Locale directories to scan",
    )
    args = parser.parse_args()
    for locale in args.locales:
        if not (LOCALE_DIR / locale).is_dir():
            print(f"[skip] missing locale/{locale}")
            continue
        print(f"\n=== locale/{locale} ===")
        issues = scan_locale(locale)
        for kind, items in issues.items():
            print(f"{kind}: {len(items)}")
            for item in items[:8]:
                print(f"  - {item}")
            if len(items) > 8:
                print(f"  ... and {len(items) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
