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

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_family", TOOL_PARSER_FAMILIES)
async def test_chat_completion_accepts_tools_for_tool_parser_families(
    monkeypatch, model_family
):
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
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    await api.create_chat_completion(
        _Request(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": tools,
            }
        )
    )

    assert list(model.chat.await_args.args[1]["tools"]) == tools


def test_minicpm5_families_use_minicpm5_tool_parser():
    for name in ("minicpm5-1b", "minicpm5-2b"):
        family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
        assert "tools" in family.model_ability
        assert family.tool_parser == "minicpm5"
        assert name in MINICPM5_TOOL_CALL_FAMILY
