# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from ...model.llm.core import LLM
from ...model.llm.llm_family import BUILTIN_LLM_FAMILIES
from .test_chat_tool_families import _chat_api
from .test_llm_request_abort import _Request

# The documented Seed-OSS request. Its chat template reads thinking_budget.
ZERO_BUDGET = {"thinking_budget": 0, "reasoning_content": True}
ENGINE_BUDGETS = {
    "thinking_token_budget",
    "max_thinking_tokens",
    "_mlx_thinking_budget",
}


def _family(name):
    return next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)


def _zero_budget_request(stream):
    return _Request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": dict(ZERO_BUDGET),
            "stream": stream,
        }
    )


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


@pytest.mark.asyncio
async def test_chat_residency_routing_renders_generation_reasoning_effort(monkeypatch):
    from ...model.llm.vllm.core import VLLMMultiModel

    family = _family("qwen3.8")
    api, actor = _chat_api(monkeypatch, "qwen3.8")
    supervisor = await api._get_supervisor_ref()
    supervisor.describe_model.return_value = family.to_description()

    model = object.__new__(VLLMMultiModel)
    model.model_family = family
    model.reasoning_parser = None
    model._engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector="XavierHybridGPUConnector")
        )
    )
    tokenizer = Mock(
        side_effect=lambda prompt, **_: SimpleNamespace(input_ids=list(prompt.encode()))
    )
    model._get_tokenizer = AsyncMock(return_value=tokenizer)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(TokensPrompt=dict))
    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        SimpleNamespace(
            process_audio_info=None,
            process_mm_info=None,
            process_vision_info=lambda *_args, **_kwargs: (None, None, {}),
        ),
    )

    async def route(_uid, _key, _reservation, *, routing_request):
        nonlocal routing_tokens
        routing_tokens = await model.get_chat_routing_tokens(
            routing_request["messages"], routing_request["config"]
        )
        return actor, False

    routing_tokens = None
    supervisor.get_chat_model = AsyncMock(side_effect=route)
    response = await api.create_chat_completion(
        _Request(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "Hello"}],
                "routing_key": "session",
                "reasoning_effort": "low",
                "chat_template_kwargs": {"enable_thinking": True},
                "stop_token_ids": [1],
            }
        )
    )
    assert response.status_code == 200
    messages, config = actor.chat.await_args.args
    inputs, _, _, _ = await model._prepare_chat_inputs(messages, dict(config))
    tokens = await model._gen_tokens_prompt(tokenizer, inputs, {})
    assert routing_tokens == tokens["prompt_token_ids"]
    assert "Reasoning effort is set to low." in inputs["prompt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_seed_oss_zero_budget_reaches_generation(monkeypatch, stream):
    seed = _family("seed-oss")
    api, model = _chat_api(monkeypatch, "seed-oss")
    supervisor = await api._get_supervisor_ref()
    supervisor.describe_model.return_value = seed.to_description()
    # The model actor runs the real LLM preflight of a Seed-OSS model.
    llm = SimpleNamespace(
        model_family=seed,
        _reasoning_effort_capability=lambda: None,
        _apply_thinking_budget=Mock(),
    )
    LLM.prepare_parse_reasoning_content(llm, True)
    model.prepare_reasoning_request = AsyncMock(
        side_effect=lambda config, **_: LLM.prepare_reasoning_request(llm, config)
    )
    if stream:

        async def chunks():
            yield b'{"choices": []}'

        model.chat.return_value = chunks()

    response = await api.create_chat_completion(_zero_budget_request(stream))

    assert response.status_code == 200
    if stream:
        assert [item async for item in response.body_iterator]
    model.prepare_reasoning_request.assert_awaited_once()
    llm._apply_thinking_budget.assert_not_called()
    config = model.chat.await_args.args[1]
    raw_config = model.chat.await_args.kwargs["raw_params"]
    assert config["chat_template_kwargs"] == ZERO_BUDGET
    assert raw_config["chat_template_kwargs"] == ZERO_BUDGET
    assert not ENGINE_BUDGETS & config.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_zero_budget_is_rejected_for_other_models(monkeypatch, stream):
    api, model = _chat_api(monkeypatch, "qwen3")
    supervisor = await api._get_supervisor_ref()
    supervisor.describe_model.return_value = _family("qwen3").to_description()

    with pytest.raises(HTTPException) as error:
        await api.create_chat_completion(_zero_budget_request(stream))

    assert error.value.status_code == 400
    assert error.value.detail == "thinking_budget must be a positive integer"
    model.chat.assert_not_awaited()
