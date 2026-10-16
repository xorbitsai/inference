# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""SSE helpers shared by the Token Router data plane."""

from __future__ import annotations

from typing import AsyncIterator

import httpx


async def relay_sse(response: httpx.Response) -> AsyncIterator[bytes]:
    """Relay raw SSE bytes without parsing or re-encoding event payloads."""
    async for chunk in response.aiter_raw():
        yield chunk
