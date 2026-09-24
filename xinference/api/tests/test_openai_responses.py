import json
from types import MethodType

import httpx
import pytest
from fastapi import FastAPI

from xinference.api import restful_api as restful_api_module
from xinference.api.protocols.openai_responses import (
    ResponsesProtocolError,
    chat_to_response,
    parse_responses_request,
    responses_stream_events,
)
from xinference.api.restful_api import RESTfulAPI

CODEX_TOOLS = [
    {
        "type": "function",
        "name": "shell",
        "description": "Run a command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "array", "items": {"type": "string"}}},
        },
    },
    {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch",
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
    },
    {"type": "web_search"},
]


def _events(items):
    return [json.loads(item["data"]) for item in items]


async def _collect(chunks, req):
    async def source():
        for chunk in chunks:
            yield chunk

    return _events([item async for item in responses_stream_events(source(), req)])


def test_string_input_with_instructions():
    req = parse_responses_request(
        {
            "model": "m",
            "instructions": "Be brief.",
            "input": "hi",
            "max_output_tokens": 64,
            "temperature": 0.2,
            "stream": True,
        }
    )
    assert req.chat_body == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
        ],
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.2,
        "stream_options": {"include_usage": True},
    }


def test_codex_style_history_keeps_call_order_and_reasoning():
    req = parse_responses_request(
        {
            "model": "m",
            "instructions": "sys",
            "tools": CODEX_TOOLS,
            "tool_choice": "auto",
            "input": [
                {"type": "message", "role": "developer", "content": "perms"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "ls"}],
                },
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "think"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "shell",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
                {
                    "type": "custom_tool_call",
                    "call_id": "c2",
                    "name": "apply_patch",
                    "input": "P",
                },
                {"type": "custom_tool_call_output", "call_id": "c2", "output": "ok"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ],
        }
    )
    messages = req.chat_body["messages"]
    assert messages[0] == {"role": "system", "content": "sys\n\nperms"}
    assert messages[1] == {"role": "user", "content": "ls"}
    # Serial calls stay serial: each call is followed by its own output.
    assert messages[2]["tool_calls"][0]["id"] == "c1"
    assert messages[2]["reasoning_content"] == "think"
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt"}
    assert messages[4]["tool_calls"][0]["function"] == {
        "name": "apply_patch",
        "arguments": json.dumps({"input": "P"}),
    }
    assert "reasoning_content" not in messages[4]
    assert messages[5]["role"] == "tool"
    assert messages[6] == {"role": "assistant", "content": "done"}

    names = [tool["function"]["name"] for tool in req.chat_body["tools"]]
    assert names == ["shell", "apply_patch"], "hosted tools are dropped"
    assert req.chat_body["tools"][1]["function"]["parameters"]["required"] == ["input"]
    assert "start: /.+/" in req.chat_body["tools"][1]["function"]["description"]
    assert req.custom_tools == {"apply_patch"}


def test_parallel_calls_share_one_assistant_message():
    req = parse_responses_request(
        {
            "model": "m",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "a",
                    "name": "f",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "b",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "a", "output": "1"},
                {"type": "function_call_output", "call_id": "b", "output": "2"},
            ],
        }
    )
    messages = req.chat_body["messages"]
    assert [c["id"] for c in messages[0]["tool_calls"]] == ["a", "b"]
    assert [m["tool_call_id"] for m in messages[1:]] == ["a", "b"]


def test_json_schema_format_and_unsupported_fields():
    req = parse_responses_request(
        {
            "model": "m",
            "input": "x",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "r",
                    "schema": {"type": "object"},
                }
            },
        }
    )
    assert req.chat_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "r", "schema": {"type": "object"}},
    }
    with pytest.raises(ResponsesProtocolError) as exc:
        parse_responses_request(
            {"model": "m", "input": "x", "previous_response_id": "r"}
        )
    assert exc.value.param == "previous_response_id"


def test_chat_to_response_maps_reasoning_text_and_calls():
    req = parse_responses_request({"model": "m", "input": "x", "tools": CODEX_TOOLS})
    response = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "hello",
                        "reasoning_content": "hmm",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {"name": "shell", "arguments": "{}"},
                            },
                            {
                                "id": "c2",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"input": "PATCH"}),
                                },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        },
        req,
    )
    assert response["status"] == "completed"
    assert [item["type"] for item in response["output"]] == [
        "reasoning",
        "message",
        "function_call",
        "custom_tool_call",
    ]
    assert response["output"][0]["content"] == [
        {"type": "reasoning_text", "text": "hmm"}
    ]
    assert response["output"][3]["input"] == "PATCH"
    assert response["usage"]["input_tokens"] == 3
    assert response["usage"]["output_tokens"] == 5


def test_truncated_chat_is_incomplete():
    req = parse_responses_request({"model": "m", "input": "x"})
    response = chat_to_response(
        {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]}, req
    )
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}


@pytest.mark.asyncio
async def test_stream_reasoning_then_text_then_tool_call():
    req = parse_responses_request({"model": "m", "input": "x", "stream": True})
    events = await _collect(
        [
            {"choices": [{"delta": {"reasoning_content": "th"}}]},
            {"choices": [{"delta": {"reasoning_content": "ink"}}]},
            {"choices": [{"delta": {"content": "Bon"}}]},
            {"choices": [{"delta": {"content": "jour"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "shell", "arguments": '{"a"'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ":1}"}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 7}},
        ],
        req,
    )
    types = [e["type"] for e in events]
    assert types[:2] == ["response.created", "response.in_progress"]
    assert types[-1] == "response.completed"
    assert [e["sequence_number"] for e in events] == list(range(len(events)))

    # Every delta must follow the output_item.added of its own item.
    added = set()
    for e in events:
        if e["type"] == "response.output_item.added":
            added.add(e["item"]["id"])
        elif e["type"].endswith(".delta"):
            assert e["item_id"] in added, e

    final = events[-1]["response"]
    assert [item["type"] for item in final["output"]] == [
        "reasoning",
        "message",
        "function_call",
    ]
    assert final["output"][0]["content"][0]["text"] == "think"
    assert final["output"][1]["content"][0]["text"] == "Bonjour"
    assert final["output"][2]["arguments"] == '{"a":1}'
    assert final["output"][2]["call_id"] == "c1"
    assert final["usage"]["output_tokens"] == 7
    done_items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    assert done_items == final["output"], "Codex rebuilds history from output_item.done"


@pytest.mark.asyncio
async def test_stream_error_and_truncation():
    req = parse_responses_request({"model": "m", "input": "x", "stream": True})
    events = await _collect(
        [
            {"choices": [{"delta": {"content": "a"}}]},
            {"error": {"message": "This model's maximum context length is 8 tokens"}},
        ],
        req,
    )
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["error"]["code"] == "context_length_exceeded"

    events = await _collect(
        [{"choices": [{"delta": {"content": "a"}, "finish_reason": "length"}]}], req
    )
    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"
    }


async def _aiter(items):
    for item in items:
        yield item


def _make_app(monkeypatch, model):
    api = RESTfulAPI.__new__(RESTfulAPI)
    api._advanced_auth_service = None
    api._uid_to_model_name = {}

    class FakeSupervisor:
        async def resolve_token_router_runtime(self, model_uid):
            return None

    async def get_supervisor_ref(_self):
        return FakeSupervisor()

    async def fake_require_model(*args, **kwargs):
        return model

    api._get_supervisor_ref = MethodType(get_supervisor_ref, api)
    monkeypatch.setattr(restful_api_module, "require_model", fake_require_model)
    app = FastAPI()
    app.add_api_route("/v1/responses", api.create_response, methods=["POST"])
    return app


class FakeModel:
    uid = "m"

    def __init__(self):
        self.calls = []
        self.released = 0

    async def chat(self, messages, kwargs, raw_params=None):
        self.calls.append((messages, kwargs))
        if not kwargs.get("stream"):
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": "pong"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                }
            ).encode()

        async def chunks():
            yield 'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"po"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{"content":"ng"},"finish_reason":"stop"}]}\n\n'
            yield 'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            yield "data: [DONE]\n\n"

        return chunks()

    async def decrease_serve_count(self):
        self.released += 1


@pytest.mark.asyncio
async def test_endpoint_non_stream_and_stream(monkeypatch):
    model = FakeModel()
    app = _make_app(monkeypatch, model)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/responses", json={"model": "m", "input": "ping"}
        )
        assert response.status_code == 200
        assert response.json()["output"][0]["content"][0]["text"] == "pong"

        response = await client.post(
            "/v1/responses", json={"model": "m", "input": "ping", "stream": True}
        )
        assert response.status_code == 200
        assert "event: response.completed" in response.text

        response = await client.post(
            "/v1/responses",
            json={"model": "m", "input": "x", "previous_response_id": "r"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "previous_response_id"

    assert model.calls[0][0] == [{"role": "user", "content": "ping"}]
    assert model.released == 1


@pytest.mark.asyncio
async def test_openai_sdk_parses_both_modes(monkeypatch):
    openai = pytest.importorskip("openai")
    if not hasattr(openai.AsyncOpenAI, "responses"):
        pytest.skip("openai SDK without the Responses API")
    app = _make_app(monkeypatch, FakeModel())
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    client = openai.AsyncOpenAI(
        api_key="x", base_url="http://test/v1", http_client=http_client
    )

    response = await client.responses.create(model="m", input="ping")
    assert response.output_text == "pong"
    assert response.usage.input_tokens == 3

    # The stream helper rebuilds a snapshot from events and rejects
    # out-of-order ones, so this checks the event sequence too.
    async with client.responses.stream(model="m", input="ping") as stream:
        deltas = [
            event.delta
            async for event in stream
            if event.type == "response.output_text.delta"
        ]
        final = await stream.get_final_response()
    assert deltas == ["po", "ng"]
    assert final.output_text == "pong"
    assert final.output[0].type == "reasoning"
    await http_client.aclose()


def test_text_then_calls_in_one_turn_stay_one_assistant_message():
    # Codex's prompt makes the model say something before calling a tool.
    req = parse_responses_request(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "go"},
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "r"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Checking."}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "f",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "x"},
                {"type": "message", "role": "developer", "content": "<model_switch>"},
            ],
        }
    )
    assert req.chat_body["messages"] == [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Checking.",
            "reasoning_content": "r",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x"},
        {"role": "user", "content": "<model_switch>"},
    ]


def test_namespaced_tools_round_trip():
    tools = [
        {
            "type": "namespace",
            "name": "mcp__repl__",
            "description": "d",
            "tools": [
                {"type": "function", "name": "js", "parameters": {"type": "object"}}
            ],
        }
    ]
    req = parse_responses_request(
        {
            "model": "m",
            "tools": tools,
            "input": [
                {
                    "type": "function_call",
                    "call_id": "c",
                    "name": "js",
                    "namespace": "mcp__repl__",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c", "output": "1"},
            ],
        }
    )
    assert req.chat_body["tools"][0]["function"]["name"] == "mcp__repl__js"
    assert (
        req.chat_body["messages"][0]["tool_calls"][0]["function"]["name"]
        == "mcp__repl__js"
    )
    response = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c2",
                                "function": {
                                    "name": "mcp__repl__js",
                                    "arguments": "{}",
                                },
                            }
                        ]
                    }
                }
            ]
        },
        req,
    )
    call = response["output"][0]
    assert (call["type"], call["name"], call["namespace"]) == (
        "function_call",
        "js",
        "mcp__repl__",
    )


def test_custom_input_given_as_json_string_and_item_reference():
    req = parse_responses_request({"model": "m", "input": "x", "tools": CODEX_TOOLS})
    response = chat_to_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": '"PATCH"',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        req,
    )
    assert response["output"][0]["input"] == "PATCH"
    with pytest.raises(ResponsesProtocolError):
        parse_responses_request(
            {"model": "m", "input": [{"type": "item_reference", "id": "x"}]}
        )


@pytest.mark.asyncio
async def test_stream_parallel_calls_sharing_index_zero():
    req = parse_responses_request({"model": "m", "input": "x", "stream": True})
    fragment = lambda cid, args: {  # noqa: E731
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": cid,
                            "function": {"name": "f", "arguments": args},
                        }
                    ]
                }
            }
        ]
    }
    events = await _collect([fragment("a", '{"q":1}'), fragment("b", '{"q":2}')], req)
    final = events[-1]["response"]
    assert [(i["call_id"], i["arguments"]) for i in final["output"]] == [
        ("a", '{"q":1}'),
        ("b", '{"q":2}'),
    ]


@pytest.mark.asyncio
async def test_stream_closes_upstream_on_error():
    req = parse_responses_request({"model": "m", "input": "x", "stream": True})
    closed = []

    async def source():
        try:
            yield {"error": {"message": "boom"}}
            yield {"choices": []}
        finally:
            closed.append(True)

    events = _events([item async for item in responses_stream_events(source(), req)])
    assert events[-1]["type"] == "response.failed"
    assert closed == [True], "serve count is released when the stream ends, not at GC"


@pytest.mark.asyncio
async def test_context_length_error_is_400_with_code(monkeypatch):
    class TooLong(FakeModel):
        async def chat(self, messages, kwargs, raw_params=None):
            raise RuntimeError("This model's maximum context length is 8 tokens")

    api_app = _make_app(monkeypatch, TooLong())

    async def no_last_error(self, uid, exc):
        return exc

    monkeypatch.setattr(RESTfulAPI, "_get_model_last_error", no_last_error)
    monkeypatch.setattr(RESTfulAPI, "_report_error_event", lambda *a, **k: _noop())
    transport = httpx.ASGITransport(app=api_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/responses", json={"model": "m", "input": "x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


async def _noop():
    return None


@pytest.mark.asyncio
async def test_stream_call_continued_with_empty_ids_stays_one_call():
    # Some backends repeat the call with id="" and name=None on later fragments.
    req = parse_responses_request({"model": "m", "input": "x", "stream": True})
    first = {
        "index": 0,
        "id": "call_8f",
        "function": {"name": "weather", "arguments": '{"city":'},
    }
    rest = {
        "index": 0,
        "id": "",
        "function": {"name": None, "arguments": ' "Hangzhou"}'},
    }
    events = await _collect(
        [
            {"choices": [{"delta": {"tool_calls": [first]}}]},
            {"choices": [{"delta": {"tool_calls": [rest]}}]},
        ],
        req,
    )
    output = events[-1]["response"]["output"]
    assert [(i["call_id"], i["name"], i["arguments"]) for i in output] == [
        ("call_8f", "weather", '{"city": "Hangzhou"}')
    ]


def test_cached_tokens_passed_through():
    req = parse_responses_request({"model": "m", "input": "x"})
    reported = chat_to_response(
        {
            "choices": [{"message": {"content": "a"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 8},
            },
        },
        req,
    )
    assert reported["usage"]["input_tokens_details"] == {"cached_tokens": 8}
    unreported = chat_to_response(
        {
            "choices": [{"message": {"content": "a"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        },
        req,
    )
    assert unreported["usage"]["input_tokens_details"] == {"cached_tokens": 0}
