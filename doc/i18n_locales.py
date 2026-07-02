"""Shared RTD / Sphinx locale mapping for documentation i18n."""

from __future__ import annotations

import os

# RTD READTHEDOCS_LANGUAGE slug (lower case) -> doc/source/locale/ directory name
RTD_TO_SPHINX_LOCALE: dict[str, str] = {
    "de": "de",
    "es": "es",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "pt-br": "pt_BR",
    "pt_br": "pt_BR",
    "zh-cn": "zh_CN",
    "zh_cn": "zh_CN",
    "zh-tw": "zh_TW",
    "zh_tw": "zh_TW",
}

KNOWN_LOCALES: tuple[str, ...] = tuple(sorted(set(RTD_TO_SPHINX_LOCALE.values())))


def resolve_sphinx_language(
    rtd_language: str | None = None,
    *,
    explicit: str | None = None,
) -> str | None:
    """Return Sphinx ``language`` config value, or ``None`` for English."""
    slug = (
        explicit or rtd_language or os.environ.get("READTHEDOCS_LANGUAGE") or ""
    ).strip().lower()
    if not slug or slug == "en" or slug.startswith(("en-", "en_")):
        return None
    return RTD_TO_SPHINX_LOCALE.get(slug, slug.replace("-", "_"))
