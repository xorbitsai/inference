#!/usr/bin/env python3
"""Compile .po catalogs to .mo before Sphinx html build (RTD + local)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from i18n_locales import KNOWN_LOCALES, resolve_sphinx_language

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"


def _messages_dir(locale: str) -> Path:
    return LOCALE_DIR / locale / "LC_MESSAGES"


def build_mo(locale: str) -> bool:
    po_root = _messages_dir(locale)
    if not po_root.is_dir():
        print(f"[build_i18n] skip {locale}: missing {po_root}")
        return False
    print(f"[build_i18n] compiling locale/{locale} ...")
    subprocess.run(
        ["sphinx-intl", "build", "-l", locale, "-d", str(LOCALE_DIR)],
        cwd=DOC_DIR,
        check=True,
    )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "locale",
        nargs="?",
        help="Sphinx locale folder name (e.g. de, zh_CN). Defaults to READTHEDOCS_LANGUAGE.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compile every known locale directory that exists under source/locale/.",
    )
    args = parser.parse_args(argv)

    if args.all:
        built = [loc for loc in KNOWN_LOCALES if build_mo(loc)]
        if not built:
            print("[build_i18n] no locale directories found")
        return 0

    locale = resolve_sphinx_language(explicit=args.locale)
    if not locale:
        print("[build_i18n] English build; skipping mo compilation")
        return 0

    build_mo(locale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
