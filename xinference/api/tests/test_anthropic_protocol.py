import json

import pytest
from anthropic import NOT_GIVEN
from anthropic.lib.streaming import MessageStream
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)

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


def test_context_management_clears_old_verified_thinking(monkeypatch):
    monkeypatch.setattr(
        "xinference.api.protocols.anthropic.verify_thinking",
        lambda text, model, signature: True,
    )
    request = parse_anthropic_request(
        _base_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "old reasoning",
                            "signature": "xinference_v1:old",
                        },
                        {"type": "text", "text": "old answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "new reasoning",
                            "signature": "xinference_v1:new",
                        },
                        {"type": "text", "text": "new answer"},
                    ],
                },
                {"role": "user", "content": "continue again"},
            ],
            context_management={
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 1},
                    }
                ]
            },
        )
    )

    assistants = [
        message for message in request.messages if message["role"] == "assistant"
    ]
    assert "reasoning_content" not in assistants[0]
    assert assistants[0]["content"] == "old answer"
    assert assistants[1]["reasoning_content"] == "new reasoning"


def test_context_management_keep_all_is_a_noop(monkeypatch):
    monkeypatch.setattr(
        "xinference.api.protocols.anthropic.verify_thinking",
        lambda text, model, signature: True,
    )
    request = parse_anthropic_request(
        _base_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "reasoning",
                            "signature": "xinference_v1:signed",
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            context_management={
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
        )
    )

    assert request.messages[0]["reasoning_content"] == "reasoning"


@pytest.mark.parametrize(
    "context_management,message",
    [
        ("invalid", "context_management must be an object"),
        ({"edits": "invalid"}, "context_management.edits must be an array"),
        (
            {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "Unsupported Anthropic context management edit type",
        ),
        (
            {"edits": [{"type": "clear_thinking_20251015", "keep": {"value": 1}}]},
            "clear_thinking_20251015.keep",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": True},
                    }
                ]
            },
            "clear_thinking_20251015.keep",
        ),
        (
            {
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 0},
                    }
                ]
            },
            "clear_thinking_20251015.keep",
        ),
        (
            {"edits": [{"type": "clear_thinking_20251015", "keep": []}]},
            "clear_thinking_20251015.keep",
        ),
    ],
)
def test_context_management_rejects_invalid_edits(context_management, message):
    with pytest.raises(AnthropicProtocolError, match=message):
        parse_anthropic_request(_base_request(context_management=context_management))


def test_context_management_preserves_unresolved_tool_thinking():
    from xinference.api.protocols.anthropic import (
        _apply_context_management,
        _plan_context_management,
    )

    messages = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "needed by pending tool use",
            "tool_calls": [{"id": "pending"}],
        },
        {
            "role": "assistant",
            "content": "later answer",
            "reasoning_content": "later reasoning",
        },
    ]

    plan = _plan_context_management(messages, {"keep_turns": 1})
    prepared = _apply_context_management(messages, plan)

    assert prepared[0]["reasoning_content"] == "needed by pending tool use"
    assert prepared[1]["reasoning_content"] == "later reasoning"


@pytest.mark.parametrize(
    "keep_turns,expected",
    [
        (1, [None, "tool reasoning A", "tool reasoning B"]),
        (2, ["old reasoning", "tool reasoning A", "tool reasoning B"]),
    ],
)
def test_context_management_retains_complete_assistant_turns(
    monkeypatch, keep_turns, expected
):
    monkeypatch.setattr(
        "xinference.api.protocols.anthropic.verify_thinking",
        lambda text, model, signature: True,
    )
    messages = [
        {"role": "user", "content": "old question"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "old reasoning",
                    "signature": "xinference_v1:old",
                },
                {"type": "text", "text": "old answer"},
            ],
        },
        {"role": "user", "content": "new question"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "tool reasoning A",
                    "signature": "xinference_v1:a",
                },
                {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-a", "content": "A"}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "tool reasoning B",
                    "signature": "xinference_v1:b",
                },
                {"type": "tool_use", "id": "call-b", "name": "lookup", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-b", "content": "B"}
            ],
        },
    ]

    request = parse_anthropic_request(
        _base_request(
            messages=messages,
            context_management={
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": keep_turns},
                    }
                ]
            },
        )
    )

    assistants = [
        message for message in request.messages if message["role"] == "assistant"
    ]
    assert [message.get("reasoning_content") for message in assistants] == expected


def test_context_management_reuses_full_history_plan_for_cache_prefix(monkeypatch):
    monkeypatch.setattr(
        "xinference.api.protocols.anthropic.verify_thinking",
        lambda text, model, signature: True,
    )
    request = parse_anthropic_request(
        _base_request(
            messages=[
                {"role": "user", "content": "old question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "old reasoning",
                            "signature": "xinference_v1:old",
                        },
                        {
                            "type": "text",
                            "text": "old answer",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {"role": "user", "content": "new question"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "new reasoning",
                            "signature": "xinference_v1:new",
                        },
                        {"type": "text", "text": "new answer"},
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            context_management={
                "edits": [
                    {
                        "type": "clear_thinking_20251015",
                        "keep": {"type": "thinking_turns", "value": 1},
                    }
                ]
            },
        )
    )

    assert "reasoning_content" not in request.messages[1]
    assert "reasoning_content" not in request.cache_control["messages"][1]
    assert request.cache_control["messages"] == request.messages[:2]


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
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
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
        "reasoning_content": True,
        "thinking_budget": 1024,
    }
    assert body["stream_options"] == {"include_usage": True}


def test_parse_request_orders_tool_results_before_follow_up_text():
    request = parse_anthropic_request(
        _base_request(
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        },
                        {"type": "text", "text": "Please summarize it."},
                    ],
                },
            ]
        )
    )

    assert request.messages == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
        {"role": "user", "content": "Please summarize it."},
    ]


def test_parse_request_enforces_minimum_thinking_budget():
    with pytest.raises(AnthropicProtocolError) as exc_info:
        parse_anthropic_request(
            _base_request(
                max_tokens=2048,
                thinking={"type": "enabled", "budget_tokens": 1023},
            )
        )
    assert (
        exc_info.value.message
        == "thinking.budget_tokens must be greater than or equal to 1024"
    )

    request = parse_anthropic_request(
        _base_request(
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
    )
    assert request.thinking_budget_tokens == 1024


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


def test_openai_response_omits_unsigned_thinking_and_maps_text_tools():
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
        {"type": "text", "text": "answer"},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "weather",
            "input": {"city": "Beijing"},
        },
    ]
    assert response["usage"] == {"input_tokens": 12, "output_tokens": 7}
    sdk_message = Message(**response)
    assert [block.type for block in sdk_message.content] == ["text", "tool_use"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "finish_reason,expected",
    [("stop", "stop_sequence"), ("length", "max_tokens"), ("tool_calls", "tool_use")],
)
async def test_stop_sequence_preserves_length_and_tool_finish_reasons(
    stream, finish_reason, expected
):
    choice = {
        "message": {"content": "hello"},
        "delta": {"content": "hello"},
        "finish_reason": finish_reason,
        "stop_sequence": "HALT",
    }
    response = {"choices": [choice], "usage": {}}
    if stream:

        async def chunks():
            yield response

        events = [
            json.loads(e["data"])
            async for e in anthropic_stream_events(
                chunks(), "test", "req_test", stop_sequences=["HALT"]
            )
        ]
        result = next(e["delta"] for e in events if e["type"] == "message_delta")
    else:
        result = openai_to_anthropic(response, "test", stop_sequences=["HALT"])
    assert result["stop_reason"] == expected
    assert result["stop_sequence"] == ("HALT" if finish_reason == "stop" else None)


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
async def test_stream_omits_unsigned_thinking_and_maps_text_tool_usage():
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
    assert payloads[-2]["usage"] == {"input_tokens": 9, "output_tokens": 5}

    active_index = None
    for payload in payloads:
        if payload.get("type") == "content_block_start":
            assert active_index is None
            active_index = payload["index"]
        elif payload.get("type") == "content_block_stop":
            assert payload["index"] == active_index
            active_index = None
    assert active_index is None

    event_types = {
        "message_start": RawMessageStartEvent,
        "content_block_start": RawContentBlockStartEvent,
        "content_block_delta": RawContentBlockDeltaEvent,
        "content_block_stop": RawContentBlockStopEvent,
        "message_delta": RawMessageDeltaEvent,
        "message_stop": RawMessageStopEvent,
    }
    sdk_events = [event_types[payload["type"]](**payload) for payload in payloads]

    class RawStream:
        def __iter__(self):
            return iter(sdk_events)

        def close(self):
            pass

    sdk_message = MessageStream(
        RawStream(), output_format=NOT_GIVEN
    ).get_final_message()
    assert [block.type for block in sdk_message.content] == ["text", "tool_use"]
    assert sdk_message.content[0].text == "hello"


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


def test_cache_breakpoints_retain_individual_lifetimes():
    body = _base_request(
        system=[
            {
                "type": "text",
                "text": "stable system",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "stable user",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "uncached tail"},
                ],
            }
        ],
    )
    request = parse_anthropic_request(body)
    checkpoints = request.cache_control["breakpoints"]
    assert [item["ttl"] for item in checkpoints] == [3600, 300]
    assert checkpoints[0]["messages"] == [
        {"role": "system", "content": "stable system"}
    ]
    assert checkpoints[0]["tools"] == []
    assert checkpoints[1]["messages"][-1] == {"role": "user", "content": "stable user"}
    assert request.messages[-1]["content"] == "stable user\nuncached tail"
    body["system"][0]["cache_control"]["ttl"] = "5m"
    body["messages"][0]["content"][0]["cache_control"]["ttl"] = "1h"
    with pytest.raises(AnthropicProtocolError, match="Longer cache lifetimes"):
        parse_anthropic_request(body)
