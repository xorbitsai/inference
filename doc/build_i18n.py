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


def _msgfmt_available() -> bool:
    from shutil import which

    return which("msgfmt") is not None


def build_mo(locale: str) -> bool:
    po_root = _messages_dir(locale)
    if not po_root.is_dir():
        print(f"[build_i18n] skip {locale}: missing {po_root}", flush=True)
        return False
    po_files = sorted(po_root.rglob("*.po"))
    if not po_files:
        print(f"[build_i18n] skip {locale}: no .po files under {po_root}", flush=True)
        return False

    if not _msgfmt_available():
        print(f"[build_i18n] compiling locale/{locale} via sphinx-intl ...", flush=True)
        subprocess.run(
            ["sphinx-intl", "build", "-l", locale, "-d", str(LOCALE_DIR)],
            cwd=DOC_DIR,
            check=True,
        )
        print(f"[build_i18n] done locale/{locale}", flush=True)
        return True

    total = len(po_files)
    print(f"[build_i18n] compiling {total} catalogs for locale/{locale} ...", flush=True)
    for index, po in enumerate(po_files, start=1):
        mo = po.with_suffix(".mo")
        rel = po.relative_to(po_root)
        print(f"[build_i18n] [{index}/{total}] {rel}", flush=True)
        subprocess.run(["msgfmt", "-o", str(mo), str(po)], check=True)
    print(f"[build_i18n] done locale/{locale}", flush=True)
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
