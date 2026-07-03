#!/usr/bin/env python3
"""Review and optionally fix PO translations via DeepSeek API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DOC_DIR = Path(__file__).resolve().parent
LOCALE_DIR = DOC_DIR / "source" / "locale"
ENV_FILE = DOC_DIR.parent / ".env"

LANGUAGE_NAMES = {
    "zh_CN": "Simplified Chinese",
    "zh_TW": "Traditional Chinese",
    "es": "Spanish",
    "ko": "Korean",
    "ja": "Japanese",
    "it": "Italian",
    "pt_BR": "Portuguese (Brazil)",
    "de": "German",
    "fr": "French",
}

PROMPT_PREFIXES = (
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


def _load_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if ENV_FILE.is_file():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("DEEPSEEK_API_KEY is not set (.env or environment)")


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


def _needs_review(msgid: str, msgstr: str) -> bool:
    if not msgid or not msgstr:
        return False
    lower = msgstr.lower()
    if any(prefix in lower for prefix in PROMPT_PREFIXES):
        return True
    if msgid.rstrip().endswith("::") and "::" not in msgstr:
        return True
    if re.search(r"[^:] ::", msgstr):
        return True
    if len(msgid) > 20 and len(msgstr) < len(msgid) * 0.35:
        return True
    return False


def _deepseek_chat(api_key: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def _review_entry(
    api_key: str,
    *,
    locale: str,
    msgid: str,
    msgstr: str,
) -> dict[str, str]:
    target = LANGUAGE_NAMES.get(locale, locale)
    system = (
        "You review Sphinx documentation gettext translations. "
        "Preserve reStructuredText markers exactly, including trailing :: for "
        "literal blocks, inline ``code``, links, roles, and placeholders. "
        "Never add translator prefaces or meta commentary. "
        "Respond with JSON only: "
        '{"ok": true|false, "issue": "...", "fixed": "..."}. '
        "If ok is true, fixed must equal the original msgstr."
    )
    user = (
        f"Target language: {target}\n"
        f"msgid:\n{msgid}\n\n"
        f"msgstr:\n{msgstr}\n"
    )
    raw = _deepseek_chat(
        api_key,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"Non-JSON response: {raw[:200]}")
    result = json.loads(match.group(0))
    return {
        "ok": str(result.get("ok", False)).lower(),
        "issue": str(result.get("issue", "")),
        "fixed": str(result.get("fixed", msgstr)),
    }


def review_locale(locale: str, *, limit: int, apply: bool) -> int:
    api_key = _load_api_key()
    root = LOCALE_DIR / locale
    if not root.is_dir():
        raise SystemExit(f"missing locale/{locale}")

    reviewed = 0
    changed = 0
    for po in sorted(root.rglob("*.po")):
        text = po.read_text(encoding="utf-8")
        pairs = _parse_pairs(text)
        file_changed = False
        for msgid, msgstr in pairs:
            if not _needs_review(msgid, msgstr):
                continue
            if reviewed >= limit:
                break
            reviewed += 1
            rel = po.relative_to(root)
            try:
                result = _review_entry(
                    api_key, locale=locale, msgid=msgid, msgstr=msgstr
                )
            except (urllib.error.URLError, ValueError, KeyError) as exc:
                print(f"[error] {rel}: {exc}")
                continue
            if result["ok"] == "true":
                print(f"[ok] {rel}")
                continue
            print(f"[issue] {rel}: {result['issue']}")
            print(f"  was: {msgstr[:120]}")
            print(f"  fix: {result['fixed'][:120]}")
            if apply and result["fixed"] and result["fixed"] != msgstr:
                text = text.replace(
                    f'msgstr "{msgstr}"',
                    f'msgstr "{result["fixed"]}"',
                    1,
                )
                file_changed = True
                changed += 1
        if apply and file_changed:
            po.write_text(text, encoding="utf-8")
        if reviewed >= limit:
            break
    print(f"reviewed={reviewed} changed={changed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("locale", help="Locale directory name, e.g. zh_CN")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return review_locale(args.locale, limit=args.limit, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
