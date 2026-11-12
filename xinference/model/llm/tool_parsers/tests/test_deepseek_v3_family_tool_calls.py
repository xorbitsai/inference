# CI: linux
import json

import pytest

from ....llm.core import LLM
from ...llm_family import BUILTIN_LLM_FAMILIES
from ...utils import ChatModelMixin
from .. import TOOL_PARSERS

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def test_every_builtin_tool_parser_is_registered():
    # An unknown name leaves the model without a tool parser, silently.
    assert {
        family.model_name: family.tool_parser
        for family in BUILTIN_LLM_FAMILIES
        if family.tool_parser and family.tool_parser not in TOOL_PARSERS
    } == {}


def _model():
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "deepseek-v3")
    model = ChatModelMixin()
    model.model_family = family
    model.model_uid = "deepseek-v3"
    model.reasoning_parser = None
    model.tool_parser = None
    LLM.prepare_parse_tool_calls(model)
    return model


def _rendered_call(model):
    """A tool call as DeepSeek-V3's chat template writes it."""
    arguments = {"city": "Beijing"}
    prompt = model.get_full_context(
        [
            {"role": "user", "content": "Weather in Beijing?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ],
        model.model_family.chat_template,
        tools=TOOLS,
    )
    # The tool instructions in the system prompt show the format too; take the
    # call from the assistant turn.
    start = prompt.index("<｜tool▁calls▁begin｜>", prompt.index("<｜Assistant｜>"))
    end = prompt.index("<｜tool▁calls▁end｜>", start) + len("<｜tool▁calls▁end｜>")
    return prompt[start:end], arguments


def _completion(text, finish_reason):
    return {
        "id": "test",
        "object": "text_completion",
        "created": 0,
        "model": "deepseek-v3",
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
        "usage": None,
    }


def test_deepseek_v3_tool_call_reaches_the_chat_response():
    model = _model()
    call, arguments = _rendered_call(model)
    choice = model._post_process_completion(
        model.model_family, model.model_uid, _completion(call, "stop"), TOOLS
    )["choices"][0]

    assert [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in choice["message"]["tool_calls"]
    ] == [("get_weather", arguments)]
    assert choice["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_deepseek_v3_tool_call_reaches_the_chat_stream():
    model = _model()
    call, arguments = _rendered_call(model)
    text = "Checking." + call

    async def chunks():
        for i in range(0, len(text), 4):
            yield _completion(text[i : i + 4], None)
        yield _completion("", "stop")

    content, names, argument_text, finish_reason = "", [], "", None
    async for chunk in model._async_to_tool_completion_chunks(chunks()):
        for choice in chunk["choices"]:
            content += choice["delta"].get("content") or ""
            for tool_call in choice["delta"].get("tool_calls") or []:
                if tool_call["function"].get("name"):
                    names.append(tool_call["function"]["name"])
                argument_text += tool_call["function"].get("arguments") or ""
            finish_reason = choice["finish_reason"] or finish_reason

    assert content == "Checking."
    assert names == ["get_weather"]
    assert json.loads(argument_text) == arguments
    assert finish_reason == "tool_calls"


def test_deepseek_v3_template_renders_the_tool_definitions():
    model = _model()
    messages = [{"role": "user", "content": "Weather in Beijing?"}]
    with_tools = model.get_full_context(
        messages, model.model_family.chat_template, tools=TOOLS
    )
    without_tools = model.get_full_context(messages, model.model_family.chat_template)

    assert with_tools != without_tools
    assert json.dumps(TOOLS[0]) in with_tools
    assert "<｜tool▁call▁begin｜>function<｜tool▁sep｜>" in with_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("request_kind", ["tools", "tool_result"])
async def test_deepseek_v3_tool_request_reaches_the_model(
    monkeypatch, request_kind, stream
):
    from .....api.tests.test_chat_tool_families import TOOL_REQUESTS, _chat_api
    from .....api.tests.test_llm_request_abort import _Request

    api, model = _chat_api(monkeypatch, "deepseek-v3")
    if stream:

        async def chunks():
            yield b'{"choices": []}'

        model.chat.return_value = chunks()
    body = {"model": "model", "stream": stream, **TOOL_REQUESTS[request_kind]}

    response = await api.create_chat_completion(_Request(body))

    assert response.status_code == 200
    if stream:
        assert [item async for item in response.body_iterator]
    model.chat.assert_awaited_once()
