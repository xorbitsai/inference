# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The OpenAI API accepts ``null`` for optional sampling parameters."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from .. import restful_api
from .test_llm_request_abort import _new_api, _Request

NULLABLE = {
    "temperature": None,
    "top_p": None,
    "stream": None,
    "presence_penalty": None,
    "frequency_penalty": None,
}


def _api(monkeypatch, **model_methods):
    model = SimpleNamespace(
        uid=b"m-rep0",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        **model_methods,
    )
    supervisor = SimpleNamespace(
        get_model=AsyncMock(return_value=model),
        describe_model=AsyncMock(return_value={"model_family": "test-family"}),
    )
    api = _new_api()
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    return api, model


@pytest.mark.asyncio
async def test_chat_accepts_null_sampling_params(monkeypatch):
    api, model = _api(monkeypatch, chat=AsyncMock(return_value=b'{"choices": []}'))
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        **NULLABLE,
    }

    response = await api.create_chat_completion(_Request(body))

    assert response.body == b'{"choices": []}'
    generate_config = model.chat.await_args.args[1]
    for name in NULLABLE:
        assert name not in generate_config
        assert name not in model.chat.await_args.kwargs["raw_params"]


@pytest.mark.asyncio
async def test_completion_accepts_null_sampling_params(monkeypatch):
    api, model = _api(monkeypatch, generate=AsyncMock(return_value=b'{"choices": []}'))
    body = {"model": "m", "prompt": "hello", "echo": None, **NULLABLE}

    response = await api.create_completion(_Request(body))

    assert response.body == b'{"choices": []}'
    generate_config = model.generate.await_args.args[1]
    for name in (*NULLABLE, "echo"):
        assert name not in generate_config


@pytest.mark.asyncio
async def test_required_fields_still_reject_null(monkeypatch):
    api, _ = _api(monkeypatch, chat=AsyncMock())

    with pytest.raises(HTTPException) as exc:
        await api.create_chat_completion(_Request({"model": "m", "messages": None}))

    assert exc.value.status_code == 422
