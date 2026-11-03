# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import json

import pytest

from ...model.llm.llm_family import BUILTIN_LLM_FAMILIES
from .test_chat_tool_families import _chat_api
from .test_llm_request_abort import _Request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["DeepSeek-V4.1-Flash", "DeepSeek-V4.1-Flash-1017"]
)
@pytest.mark.parametrize("location", ["top", "template", "template_json"])
async def test_chat_preserves_deepseek_v41_none_effort(
    monkeypatch, model_name, location
):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    api, model = _chat_api(monkeypatch, model_name)
    supervisor = await api._get_supervisor_ref()
    supervisor.describe_model.return_value = family.to_description()
    request = {
        "model": "model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    if location == "top":
        request["reasoning_effort"] = "none"
    else:
        context = {"reasoning_effort": "none"}
        request["chat_template_kwargs"] = (
            json.dumps(context) if location == "template_json" else context
        )

    response = await api.create_chat_completion(_Request(request))

    assert response.status_code == 200
    config = model.chat.await_args.args[1]
    raw_config = model.chat.await_args.kwargs["raw_params"]
    assert config["chat_template_kwargs"]["reasoning_effort"] == "none"
    assert raw_config["chat_template_kwargs"]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in config
