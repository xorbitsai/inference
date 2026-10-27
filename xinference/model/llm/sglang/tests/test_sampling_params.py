# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# CI: linux
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..core import SGLANGChatModel


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [7, None])
async def test_openai_seed_reaches_sglang_as_sampling_seed(monkeypatch, seed):
    instance = object.__new__(SGLANGChatModel)
    instance.model_uid = "test"
    instance.reasoning_parser = None
    instance._model_config = {}
    instance._active_request_ids = set()
    instance._engine = SimpleNamespace(generate_url="http://unused/generate")
    instance._prepare_chat_inputs = AsyncMock(
        side_effect=lambda messages, config: ("prompt", None, None, {}, config)
    )
    payloads = []
    response = MagicMock()
    response.__aenter__.return_value = response
    response.json = AsyncMock(
        return_value={
            "text": "hi",
            "meta_info": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": {"type": "stop"},
            },
        }
    )
    session = MagicMock()
    session.__aenter__.return_value = session

    def post(url, json, headers):
        payloads.append(json)
        return response

    session.post.side_effect = post
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: session)

    config = {"max_tokens": 8} if seed is None else {"max_tokens": 8, "seed": seed}
    await instance.async_chat([{"role": "user", "content": "hi"}], config)

    # SGLang's msgspec SamplingParams rejects unknown keys; its seed field is
    # ``sampling_seed`` (sglang/srt/sampling/sampling_params.py).
    sampling_params = payloads[0]["sampling_params"]
    assert "seed" not in sampling_params
    assert sampling_params.get("sampling_seed") == seed
