#!/usr/bin/env python3
"""Fix reST literal-block markers (::) broken in translated PO msgstr values."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"


def _parse_pairs(text: str) -> list[tuple[str, str, int, int]]:
    """Return (msgid, msgstr, msgstr_start, msgstr_end) using line indices."""
    pairs: list[tuple[str, str, int, int]] = []
    lines = text.splitlines(keepends=True)
    msgid: str | None = None
    msgstr: str | None = None
    msgstr_start = msgstr_end = -1
    in_msgid = in_msgstr = False

    def flush() -> None:
        nonlocal msgid, msgstr, msgstr_start, msgstr_end, in_msgid, in_msgstr
        if msgid is not None:
            pairs.append((msgid, msgstr or "", msgstr_start, msgstr_end))
        msgid = msgstr = None
        msgstr_start = msgstr_end = -1
        in_msgid = in_msgstr = False

    for index, line in enumerate(lines):
        if line.startswith("msgid "):
            flush()
            msgid = line[6:].strip().strip('"')
            in_msgid, in_msgstr = True, False
        elif line.startswith("msgstr "):
            msgstr = line[7:].strip().strip('"')
            msgstr_start = index
            msgstr_end = index
            in_msgid, in_msgstr = False, True
        elif line.startswith('"') and (in_msgid or in_msgstr):
            chunk = line.strip().strip('"')
            if in_msgid and msgid is not None:
                msgid += chunk
            elif in_msgstr and msgstr is not None:
                msgstr += chunk
                msgstr_end = index
        elif not line.startswith("#"):
            flush()
    flush()
    return pairs, lines


def _fix_msgstr(msgid: str, msgstr: str) -> str | None:
    if not msgstr or not msgid.rstrip().endswith("::"):
        return None
    if "::" in msgstr:
        fixed = re.sub(r"([^:]) ::", r"\1::", msgstr)
        fixed = re.sub(r"([^:])：：", r"\1::", fixed)
        return fixed if fixed != msgstr else None
    if msgstr.endswith(":::"):
        return msgstr[:-1]
    for suffix in ("：", ":", "。", ".", "！", "!", "？", "?"):
        if msgstr.endswith(suffix):
            return msgstr[: -len(suffix)] + "::"
    return msgstr + "::"


def _write_msgstr(lines: list[str], start: int, end: int, msgstr: str) -> list[str]:
    old_chunks: list[str] = []
    for line in lines[start : end + 1]:
        if line.startswith("msgstr "):
            old_chunks.append(line[7:].strip().strip('"'))
        elif line.startswith('"'):
            old_chunks.append(line.strip().strip('"'))
    old_msgstr = "".join(old_chunks)

    if old_msgstr and old_msgstr != msgstr:
        if old_msgstr.endswith("::") and msgstr.endswith("::"):
            suffix_len = len(old_msgstr) - len(old_msgstr.rstrip(":").rstrip("："))
        else:
            suffix_len = 0
        if suffix_len and msgstr.endswith("::"):
            prefix = msgstr[: -len("::")]
            old_prefix = old_msgstr[: -suffix_len] if suffix_len else old_msgstr
            if prefix == old_prefix or prefix.startswith(old_prefix.rstrip("：:")):
                msgstr_lines = lines[start : end + 1]
                last = msgstr_lines[-1]
                if last.startswith("msgstr "):
                    msgstr_lines[-1] = f'msgstr "{prefix}::"\n'
                else:
                    msgstr_lines[-1] = f'"{prefix}::"\n'
                return lines[:start] + msgstr_lines + lines[end + 1 :]

    wrapped: list[str] = []
    chunk_size = 70
    parts = [msgstr[i : i + chunk_size] for i in range(0, len(msgstr), chunk_size)] or [""]
    wrapped.append(f'msgstr "{parts[0]}"\n')
    for part in parts[1:]:
        wrapped.append(f'"{part}"\n')
    return lines[:start] + wrapped + lines[end + 1 :]


def fix_locale(locale: str, *, dry_run: bool = False) -> int:
    root = LOCALE_DIR / locale
    fixed_count = 0
    for po in sorted(root.rglob("*.po")):
        text = po.read_text(encoding="utf-8")
        pairs, lines = _parse_pairs(text)
        updated = False
        for msgid, msgstr, start, end in reversed(pairs):
            fixed = _fix_msgstr(msgid, msgstr)
            if fixed is None:
                continue
            rel = po.relative_to(root)
            print(f"[fix] {rel}: {msgstr[-40:]!r} -> {fixed[-40:]!r}")
            lines = _write_msgstr(lines, start, end, fixed)
            updated = True
            fixed_count += 1
        if updated and not dry_run:
            po.write_text("".join(lines), encoding="utf-8")
    return fixed_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locales", nargs="*", default=["zh_CN"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    total = 0
    for locale in args.locales:
        total += fix_locale(locale, dry_run=args.dry_run)
    print(f"fixed {total} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
