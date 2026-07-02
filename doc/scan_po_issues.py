#!/usr/bin/env python3
"""Scan .po catalogs for common i18n quality issues."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"

META_PATTERNS = re.compile(
    r"(?i)(traduzione\s*(richiesta)?\s*:|translation\s*:|\bas an ai\b|i cannot translate)"
)


def _parse_msg(block: str, field: str) -> str:
    pattern = rf'^{field} "(.*?)"$'
    match = re.search(pattern, block, re.M | re.S)
    if match:
        return match.group(1)
    if f'{field} ""' not in block:
        return ""
    multiline = re.search(rf'{field} ""\n((?:".*"\n)*)', block)
    if not multiline:
        return ""
    return "".join(re.findall(r'"(.*?)"', multiline.group(1)))


def scan_po_file(po: Path, locale_root: Path) -> dict[str, list[str]]:
    text = po.read_text(encoding="utf-8")
    rel = str(po.relative_to(locale_root))
    issues: dict[str, list[str]] = {
        "header_fuzzy": [],
        "meta_prefix": [],
        "empty_translation": [],
        "identical": [],
    }

    if re.search(r"^#\s*,\s*fuzzy\s*$", text, re.M):
        issues["header_fuzzy"].append(rel)

    for match in META_PATTERNS.finditer(text):
        line = text[: match.start()].count("\n") + 1
        issues["meta_prefix"].append(f"{rel}:{line}")

    for block in re.split(r"\n\n+", text):
        if not block.startswith("msgid"):
            continue
        msgid = _parse_msg(block, "msgid")
        if not msgid:
            continue
        msgstr = _parse_msg(block, "msgstr")
        if msgstr == "":
            issues["empty_translation"].append(f"{rel} :: {msgid[:80]}")
        elif msgstr == msgid and len(msgid) > 20:
            issues["identical"].append(f"{rel} :: {msgid[:80]}")
    return issues


def scan_locale(locale: str) -> dict[str, list[str]]:
    root = LOCALE_DIR / locale
    combined: dict[str, list[str]] = {
        "header_fuzzy": [],
        "meta_prefix": [],
        "empty_translation": [],
        "identical": [],
    }
    if not root.is_dir():
        print(f"locale not found: {root}", file=sys.stderr)
        return combined
    for po in sorted(root.rglob("*.po")):
        file_issues = scan_po_file(po, root)
        for key, values in file_issues.items():
            combined[key].extend(values)
    return combined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locale", help="Locale folder name (e.g. it, ja)")
    args = parser.parse_args(argv)

    issues = scan_locale(args.locale)
    total = 0
    for key, values in issues.items():
        print(f"=== {key}: {len(values)} ===")
        for value in values[:20]:
            print(f"  {value}")
        if len(values) > 20:
            print(f"  ... and {len(values) - 20} more")
        total += len(values)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
