#!/usr/bin/env python3
# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scaffold documentation locale .po files from the zh_CN template."""

from __future__ import annotations

import argparse
from pathlib import Path

from babel.messages import pofile

LOCALE_ROOT = Path(__file__).resolve().parent / "locale"
SOURCE_LOCALE = "zh_CN"
TARGET_LOCALES = ("zh_TW", "ja", "ko", "de", "fr", "es", "it", "pt_BR")


def _clear_message_string(message) -> None:
    if isinstance(message.string, (list, tuple)):
        message.string = type(message.string)("" for _ in message.string)
    else:
        message.string = ""


def _clear_translations(catalog, language: str) -> None:
    catalog.locale = language
    catalog.language_team = f"{language} <LL@li.org>"
    for message in catalog:
        message.flags.discard("fuzzy")
        if not message.id:
            continue
        _clear_message_string(message)
    # Also clear obsolete entries so zh_CN translations don't leak into other locales
    catalog.obsolete.clear()


def scaffold_locale(source_po: Path, target_po: Path, language: str) -> None:
    with source_po.open("rb") as inpf:
        catalog = pofile.read_po(inpf, locale=language)
    _clear_translations(catalog, language)
    target_po.parent.mkdir(parents=True, exist_ok=True)
    with target_po.open("wb") as outf:
        pofile.write_po(outf, catalog, width=76)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create empty .po scaffolds for additional doc locales."
    )
    parser.add_argument(
        "--source-locale",
        default=SOURCE_LOCALE,
        help=f"Template locale directory name (default: {SOURCE_LOCALE})",
    )
    parser.add_argument(
        "--locales",
        nargs="+",
        default=list(TARGET_LOCALES),
        help="Target locale directory names to scaffold.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target .po files.",
    )
    args = parser.parse_args()

    source_root = LOCALE_ROOT / args.source_locale / "LC_MESSAGES"
    if not source_root.is_dir():
        raise SystemExit(f"Source locale not found: {source_root}")

    source_files = sorted(source_root.rglob("*.po"))
    if not source_files:
        raise SystemExit(f"No .po files found under {source_root}")

    created = 0
    skipped = 0
    for language in args.locales:
        if language == args.source_locale:
            continue
        for source_po in source_files:
            rel_path = source_po.relative_to(source_root)
            target_po = LOCALE_ROOT / language / "LC_MESSAGES" / rel_path
            if target_po.exists() and not args.force:
                skipped += 1
                continue
            scaffold_locale(source_po, target_po, language)
            created += 1

    print(
        f"Scaffolded {created} .po file(s) for {len(args.locales)} locale(s); "
        f"skipped {skipped} existing file(s)."
    )


if __name__ == "__main__":
    main()
