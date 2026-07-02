#!/usr/bin/env python3
"""Compile .po catalogs to .mo before Sphinx html build (RTD + local)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from i18n_locales import KNOWN_LOCALES, resolve_sphinx_language

try:
    import babel  # noqa: F401

    HAS_BABEL = True
except ImportError:
    HAS_BABEL = False

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"


def _messages_dir(locale: str) -> Path:
    return LOCALE_DIR / locale / "LC_MESSAGES"


def _compile_po_file(po: Path, locale: str) -> None:
    from babel.messages.mofile import write_mo
    from babel.messages.pofile import read_po

    with po.open("rb") as f:
        catalog = read_po(f, locale=locale)
    mo = po.with_suffix(".mo")
    with mo.open("wb") as f:
        write_mo(f, catalog)


def build_mo(locale: str) -> bool:
    po_root = _messages_dir(locale)
    if not po_root.is_dir():
        print(f"[build_i18n] skip {locale}: missing {po_root}", flush=True)
        return False
    po_files = sorted(po_root.rglob("*.po"))
    if not po_files:
        print(f"[build_i18n] skip {locale}: no .po files under {po_root}", flush=True)
        return False

    total = len(po_files)
    print(f"[build_i18n] compiling {total} catalogs for locale/{locale} ...", flush=True)
    for index, po in enumerate(po_files, start=1):
        rel = po.relative_to(po_root)
        print(f"[build_i18n] [{index}/{total}] {rel}", flush=True)
        try:
            _compile_po_file(po, locale)
        except Exception as exc:
            print(f"[build_i18n] failed on {rel}: {exc}", flush=True)
            raise
    print(f"[build_i18n] done locale/{locale}", flush=True)
    return True


def _build_mo_with_sphinx_intl(locale: str) -> bool:
    print(f"[build_i18n] compiling locale/{locale} via sphinx-intl ...", flush=True)
    subprocess.run(
        ["sphinx-intl", "build", "-l", locale, "-d", str(LOCALE_DIR)],
        cwd=DOC_DIR,
        check=True,
    )
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
        if HAS_BABEL:
            built = [loc for loc in KNOWN_LOCALES if build_mo(loc)]
            if not built:
                print("[build_i18n] no locale directories found")
        else:
            print(
                "[build_i18n] babel not installed, falling back to sphinx-intl "
                "for all locales...",
                flush=True,
            )
            for loc in KNOWN_LOCALES:
                if _messages_dir(loc).is_dir():
                    _build_mo_with_sphinx_intl(loc)
        return 0

    locale = resolve_sphinx_language(explicit=args.locale)
    if not locale:
        print("[build_i18n] English build; skipping mo compilation")
        return 0

    if HAS_BABEL:
        build_mo(locale)
    else:
        _build_mo_with_sphinx_intl(locale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
