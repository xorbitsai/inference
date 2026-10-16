from __future__ import annotations

import asyncio
import os

from ._templates import AVAILABLE_STYLES, load_system_prompt
from .adapters import AnthropicVlmAdapter, ChatCompletionsVlmAdapter, VlmAdapter

DEFAULT_STYLE = "infographic"
DEFAULT_BACKEND = "chat_completions"
DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
DEFAULT_MODEL = "gemini-3.1-pro"

_ENV_PREFIX = "U1_ENHANCE_"
_SUPPORTED_BACKENDS = ("chat_completions", "anthropic")


def make_adapter_from_env(
    *,
    backend: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> VlmAdapter:
    """Construct a :class:`VlmAdapter` from env + explicit overrides.

    Resolution order (highest priority first):

    1. Explicit kwargs passed to this function.
    2. ``U1_ENHANCE_BACKEND`` / ``U1_ENHANCE_ENDPOINT`` / ``U1_ENHANCE_API_KEY``
       / ``U1_ENHANCE_MODEL`` environment variables.
    3. Defaults (Gemini 3.1 Pro via its OpenAI-compatible endpoint).

    Args:
        backend: ``'chat_completions'`` (any OpenAI-compatible, Gemini,
            Kimi etc.) or ``'anthropic'``.
        endpoint: Full URL of the ``/chat/completions`` or ``/v1/messages``
            endpoint.
        api_key: Bearer token.
        model: Model name string sent in the request body.

    Raises:
        RuntimeError: If no API key can be resolved.
        ValueError: If ``backend`` is unsupported.
    """
    backend = (backend or os.environ.get(f"{_ENV_PREFIX}BACKEND") or DEFAULT_BACKEND).lower()
    endpoint = endpoint or os.environ.get(f"{_ENV_PREFIX}ENDPOINT") or DEFAULT_ENDPOINT
    model = model or os.environ.get(f"{_ENV_PREFIX}MODEL") or DEFAULT_MODEL
    api_key = api_key or os.environ.get(f"{_ENV_PREFIX}API_KEY")
    if not api_key:
        raise RuntimeError(
            f"Prompt enhancement requires an API key. Set {_ENV_PREFIX}API_KEY or pass api_key= explicitly."
        )

    if backend == "chat_completions":
        return ChatCompletionsVlmAdapter(endpoint_url=endpoint, api_key=api_key, model=model)
    if backend == "anthropic":
        return AnthropicVlmAdapter(endpoint_url=endpoint, api_key=api_key, model=model)
    raise ValueError(f"Unsupported enhance backend {backend!r}; supported: {_SUPPORTED_BACKENDS}")


class PromptEnhancer:
    """Thin facade that turns a :class:`VlmAdapter` into a one-shot enhancer.

    Both entry points call the adapter with ``images=[]`` so this works with
    any text-only chat-style LLM; vision-capable backends simply ignore the
    empty image list.
    The enhancer does not own the adapter's HTTP client's lifecycle –
    call :meth:`aclose` explicitly (or rely on the sync :meth:`enhance` path
    which spins up / tears down a fresh event loop and closes the client for you).
    """

    def __init__(self, adapter: VlmAdapter, *, style: str = DEFAULT_STYLE) -> None:
        if style not in AVAILABLE_STYLES:
            raise ValueError(f"Unknown enhance style {style!r}; supported: {AVAILABLE_STYLES}")
        self._adapter = adapter
        self._style = style
        self._system_prompt = load_system_prompt(style)

    @classmethod
    def from_env(
        cls,
        *,
        style: str = DEFAULT_STYLE,
        backend: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> PromptEnhancer:
        """Convenience constructor that reads the ``U1_ENHANCE_*`` env vars."""
        adapter = make_adapter_from_env(backend=backend, endpoint=endpoint, api_key=api_key, model=model)
        return cls(adapter, style=style)

    @property
    def style(self) -> str:
        return self._style

    async def aenhance(self, user_prompt: str) -> str:
        """Async entry point: expand ``user_prompt`` into a long T2I prompt."""
        return await self._adapter.vision_completion(
            user_prompt=user_prompt,
            images=[],
            system_prompt=self._system_prompt,
        )

    def enhance(self, user_prompt: str) -> str:
        """Sync wrapper around :meth:`aenhance`.

        Creates and tears down its own event loop on every call – fine for a
        CLI that enhances a handful of prompts, but do not use inside an
        already-running event loop (call :meth:`aenhance` directly there).
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "PromptEnhancer.enhance() is sync; you're already inside an asyncio loop. "
                "Use `await enhancer.aenhance(...)` instead."
            )

        async def _once() -> str:
            try:
                return await self.aenhance(user_prompt)
            finally:
                await self._adapter.aclose()

        return asyncio.run(_once())

    async def aclose(self) -> None:
        """Release HTTP resources owned by the underlying adapter."""
        await self._adapter.aclose()
