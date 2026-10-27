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

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from ...model.llm.llm_family import BUILTIN_LLM_FAMILIES
from ...model.llm.tool_parsers import TOOL_PARSERS
from ...model.llm.utils import MINICPM5_TOOL_CALL_FAMILY
from .. import restful_api
from .test_llm_request_abort import _new_api, _Request

TOOL_PARSER_FAMILIES = sorted(
    family.model_name
    for family in BUILTIN_LLM_FAMILIES
    if "tools" in family.model_ability and family.tool_parser in TOOL_PARSERS
)
TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]
# The two requests the route refuses for a family outside the tool-call families.
TOOL_REQUESTS = {
    "tools": {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": TOOLS,
    },
    "tool_result": {
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
        ]
    },
}


def _chat_api(monkeypatch, model_family):
    model = SimpleNamespace(
        uid=b"model",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        chat=AsyncMock(return_value=b"{}"),
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    api = _new_api()
    supervisor = SimpleNamespace(
        describe_model=AsyncMock(return_value={"model_family": model_family})
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    return api, model


def _tool_request(request_kind):
    return _Request({"model": "model", **copy.deepcopy(TOOL_REQUESTS[request_kind])})


@pytest.mark.asyncio
@pytest.mark.parametrize("model_family", TOOL_PARSER_FAMILIES)
async def test_chat_completion_accepts_tools_for_tool_parser_families(
    monkeypatch, model_family
):
    api, model = _chat_api(monkeypatch, model_family)

    await api.create_chat_completion(_tool_request("tools"))

    assert list(model.chat.await_args.args[1]["tools"]) == TOOLS


def test_minicpm5_families_use_minicpm5_tool_parser():
    for name in ("minicpm5-1b", "minicpm5-2b"):
        family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
        assert "tools" in family.model_ability
        assert family.tool_parser == "minicpm5"
        assert name in MINICPM5_TOOL_CALL_FAMILY


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", TOOL_REQUESTS)
@pytest.mark.parametrize("model_family", ["Baichuan-M2", "QwQ-32B"])
async def test_chat_completion_accepts_tool_requests_for_families_listing_tools(
    monkeypatch, model_family, request_kind
):
    # Baichuan-M2 lists `tools` and, without a tool parser, was refused. QwQ-32B
    # is a Qwen2 reasoning family that already had the qwen parser.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_family)
    assert "tools" in family.model_ability
    api, model = _chat_api(monkeypatch, model_family)

    await api.create_chat_completion(_tool_request(request_kind))

    model.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", TOOL_REQUESTS)
async def test_chat_completion_refuses_tool_requests_without_the_tools_ability(
    monkeypatch, request_kind
):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "llama-2-chat")
    assert "tools" not in family.model_ability
    api, model = _chat_api(monkeypatch, family.model_name)

    with pytest.raises(HTTPException) as error:
        await api.create_chat_completion(_tool_request(request_kind))

    assert error.value.status_code == 400
    model.chat.assert_not_awaited()
