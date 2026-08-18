"""System-prompt templates for LLM-based prompt enhancement.

Templates live as ``.md`` files under ``sensenova_u1/prompt_enhance/templates/``
(makes them easy to edit, diff and swap out without touching Python). They are
loaded once at import time via ``importlib.resources``.

Adding a new style is a two-step change:

1. Drop a new ``*.md`` under ``templates/``.
2. Add its stem to :data:`AVAILABLE_STYLES` below.
"""

from __future__ import annotations

from importlib import resources

AVAILABLE_STYLES = ("infographic",)
"""Styles currently shipped with the package."""

_TEMPLATES_PACKAGE = "sensenova_u1.prompt_enhance.templates"


def load_system_prompt(style: str) -> str:
    """Load the system-prompt ``.md`` for ``style`` from package data.

    Raises:
        ValueError: If ``style`` is not in :data:`AVAILABLE_STYLES`.
    """
    if style not in AVAILABLE_STYLES:
        raise ValueError(f"Unknown enhance style {style!r}; supported: {AVAILABLE_STYLES}")
    return resources.files(_TEMPLATES_PACKAGE).joinpath(f"{style}_system.md").read_text(encoding="utf-8")
