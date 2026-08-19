import json

import pytest

from xinference.api.protocols import (
    AnthropicProtocolError,
    anthropic_error_response,
    anthropic_stream_events,
    openai_to_anthropic,
    parse_anthropic_request,
)


def _base_request(**overrides):
    body = {
        "model": "virtual-model",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return body


def test_parse_request_maps_system_stop_tools_and_thinking():
    request = parse_anthropic_request(
        _base_request(
            system=[
                {"type": "text", "text": "x-anthropic-billing-header: ignored"},
                {"type": "text", "text": "Be concise."},
            ],
            messages=[
                {"role": "system", "content": "Use metric units."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "weather",
                            "input": {"city": "Shanghai"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "text", "text": "sunny"}],
                        }
                    ],
                },
            ],
            stop_sequences=["END"],
            temperature=0.2,
            top_p=0.8,
            top_k=20,
            tools=[
                {
                    "name": "weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "weather"},
            thinking={"type": "enabled", "budget_tokens": 128},
            stream=True,
        )
    )

    body = request.to_openai_body()
    assert body["model"] == "virtual-model"
    assert body["messages"] == [
        {"role": "system", "content": "Be concise.\nUse metric units."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Shanghai"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
    ]
    assert body["stop"] == ["END"]
    assert body["tools"][0]["function"]["parameters"]["type"] == "object"
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": "weather"},
    }
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking": True,
        "thinking_budget": 128,
    }
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"messages": [{"role": "user", "content": [{"type": "image"}]}]}, "image"),
        ({"stream": "true"}, "stream"),
        ({"temperature": 1.1}, "temperature"),
        ({"top_p": -0.1}, "top_p"),
        ({"top_k": 0}, "top_k"),
        (
            {"thinking": {"type": "enabled", "budget_tokens": 512}},
            "budget_tokens",
        ),
    ],
)
def test_parse_request_rejects_invalid_or_unsupported_fields(overrides, message):
    with pytest.raises(AnthropicProtocolError) as exc_info:
        parse_anthropic_request(_base_request(**overrides))
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_type == "invalid_request_error"
    assert message in exc_info.value.message


def test_openai_response_maps_text_thinking_tools_stop_and_usage():
    response = openai_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "reasoning",
                        "content": "answer",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":"Beijing"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
        "virtual-model",
    )

    assert response["id"].startswith("msg_")
    assert response["model"] == "virtual-model"
    assert response["stop_reason"] == "tool_use"
    assert response["content"] == [
        {"type": "thinking", "thinking": "reasoning"},
        {"type": "text", "text": "answer"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "weather",
            "input": {"city": "Beijing"},
        },
    ]
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 7}


@pytest.mark.parametrize(
    "arguments",
    ["{invalid", "[]", '"value"', "1", "null", ["already-decoded"]],
)
def test_openai_response_defaults_invalid_tool_arguments_to_empty_object(arguments):
    response = openai_to_anthropic(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "weather",
                                    "arguments": arguments,
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        "model",
    )

    assert response["content"] == [
        {
            "type": "tool_use",
            "id": response["content"][0]["id"],
            "name": "weather",
            "input": {},
        }
    ]


@pytest.mark.asyncio
async def test_stream_maps_thinking_text_tool_and_usage_in_order():
    async def chunks():
        yield {
            "choices": [
                {"delta": {"reasoning_content": "think"}, "finish_reason": None}
            ]
        }
        yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"Paris"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        yield {
            "choices": [],
            "usage": {"prompt_tokens": 9, "completion_tokens": 5},
        }

    events = [
        event
        async for event in anthropic_stream_events(chunks(), "virtual-model", "req_1")
    ]
    names = [event["event"] for event in events]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    payloads = [json.loads(event["data"]) for event in events]
    assert payloads[0]["message"]["id"].startswith("msg_")
    tool_start = next(
        payload
        for payload in payloads
        if payload.get("type") == "content_block_start"
        and payload["content_block"]["type"] == "tool_use"
    )
    assert tool_start["content_block"]["id"] == "call_1"
    tool_deltas = [
        payload["delta"]["partial_json"]
        for payload in payloads
        if payload.get("type") == "content_block_delta"
        and payload["delta"]["type"] == "input_json_delta"
    ]
    assert tool_deltas == ['{"city":', '"Paris"}']
    assert payloads[-2]["delta"]["stop_reason"] == "tool_use"
    assert payloads[-2]["usage"] == {"output_tokens": 5}

    active_index = None
    for payload in payloads:
        if payload.get("type") == "content_block_start":
            assert active_index is None
            active_index = payload["index"]
        elif payload.get("type") == "content_block_stop":
            assert payload["index"] == active_index
            active_index = None
    assert active_index is None


@pytest.mark.asyncio
async def test_stream_buffers_multiple_tool_calls_by_index():
    async def chunks():
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_2",
                                "function": {"name": "time", "arguments": '{"tz":'},
                            },
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city":',
                                },
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": '"UTC"}'}},
                            {
                                "index": 0,
                                "function": {"arguments": '"Paris"}'},
                            },
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    payloads = [
        json.loads(event["data"])
        async for event in anthropic_stream_events(chunks(), "model", "req_tools")
    ]
    starts = [
        payload["content_block"]
        for payload in payloads
        if payload.get("type") == "content_block_start"
    ]
    assert [block["id"] for block in starts] == ["call_1", "call_2"]
    deltas = [
        payload["delta"]["partial_json"]
        for payload in payloads
        if payload.get("type") == "content_block_delta"
    ]
    assert deltas == ['{"city":', '"Paris"}', '{"tz":', '"UTC"}']


@pytest.mark.asyncio
async def test_stream_error_terminates_without_message_stop():
    async def chunks():
        yield {"error": {"type": "rate_limit_error", "message": "slow down"}}
        yield {"choices": [{"delta": {"content": "not emitted"}}]}

    events = [
        event async for event in anthropic_stream_events(chunks(), "model", "req_2")
    ]
    assert [event["event"] for event in events] == ["message_start", "error"]
    assert json.loads(events[-1]["data"]) == anthropic_error_response(
        "rate_limit_error", "slow down", "req_2"
    )


@pytest.mark.asyncio
async def test_stream_closes_inner_iterator_when_consumer_stops_early():
    closed = False

    async def chunks():
        nonlocal closed
        try:
            yield {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        finally:
            closed = True

    events = anthropic_stream_events(chunks(), "model", "req_disconnect")
    assert (await anext(events))["event"] == "message_start"
    assert (await anext(events))["event"] == "content_block_start"
    assert closed is False

    await events.aclose()

    assert closed is True
