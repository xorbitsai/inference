import base64
import io
import json

import pytest

from xinference.api.protocols.anthropic import (
    AnthropicProtocolError,
    anthropic_stream_events,
    openai_to_anthropic,
    parse_anthropic_request,
)


def request(**kwargs):
    return parse_anthropic_request(
        {
            "model": "local",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "hello"}],
            **kwargs,
        }
    )


def image():
    from PIL import Image

    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(stream, "PNG")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(stream.getvalue()).decode(),
        },
    }


def test_image_order_and_tool_result_images():
    block = image()
    result = request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    block,
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    )
    content = result.messages[0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].endswith(block["source"]["data"])
    result = request(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [block],
                        "is_error": True,
                    }
                ],
            }
        ]
    )
    assert result.messages[0]["role"] == "tool"
    assert result.messages[0]["content"][0]["text"] == "Tool error:"
    assert result.messages[0]["content"][1]["type"] == "image_url"


def test_adaptive_effort_and_reasoning_history():
    result = request(
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "previous reasoning",
                        "signature": "opaque",
                    },
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "answer"},
                ],
            },
            {"role": "user", "content": "continue"},
        ],
    )
    assert "reasoning_content" not in result.messages[0]
    body = result.to_openai_body()
    assert body["reasoning_effort"] == "high"
    assert body["chat_template_kwargs"] == {
        "enable_thinking": True,
        "thinking": True,
        "reasoning_content": True,
    }
    assert result.thinking_budget_tokens is None


def test_tool_references_resolve_against_definitions():
    result = request(
        tools=[{"name": "Read", "input_schema": {"type": "object"}}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "search",
                        "content": [{"type": "tool_reference", "tool_name": "Read"}],
                    }
                ],
            }
        ],
    )
    assert result.messages[0]["content"] == "Tool available: Read"
    with pytest.raises(AnthropicProtocolError, match="Unknown tool reference"):
        request(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_reference", "tool_name": "missing"}],
                }
            ]
        )


def test_cache_boundary_and_ttl_are_retained():
    result = request(
        system="system",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "stable",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    },
                    {"type": "text", "text": "variable"},
                ],
            }
        ],
    )
    assert result.cache_control == {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "stable"},
        ],
        "ttl": 3600,
    }
    assert result.to_openai_body()["_anthropic_cache_control"] == result.cache_control
    with pytest.raises(AnthropicProtocolError, match="ttl"):
        request(cache_control={"type": "ephemeral", "ttl": "forever"})


def test_schema_and_parallel_controls_are_preserved():
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    body = request(
        output_config={"format": {"type": "json_schema", "schema": schema}},
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    ).to_openai_body()
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["parallel_tool_calls"] is False
    with pytest.raises(AnthropicProtocolError, match="boolean"):
        request(tool_choice={"type": "auto", "disable_parallel_tool_use": "yes"})


def test_cache_system_boundary_excludes_later_blocks_and_tools():
    result = request(
        tools=[{"name": "Read", "input_schema": {"type": "object"}}],
        system=[
            {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "variable"},
        ],
    )
    assert result.cache_control == {
        "messages": [{"role": "system", "content": "stable"}],
        "tools": [],
        "ttl": 300,
    }


def test_text_documents_and_pdf_pages():
    result = request(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "document text",
                        },
                        "title": "Title",
                    }
                ],
            }
        ]
    )
    assert result.messages[0]["content"] == "Title\ndocument text"
    pytest.importorskip("pypdfium2")
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(100, 100)
    from pypdf.generic import DecodedStreamObject, NameObject

    # A PDF stream must be an indirect object. Draw a visible black rectangle
    # to verify rendering preserves actual page content rather than a blank page.
    page_content = DecodedStreamObject()
    page_content.set_data(b"0 0 0 rg 10 10 50 50 re f")
    page[NameObject("/Contents")] = writer._add_object(page_content)
    stream = io.BytesIO()
    writer.write(stream)
    block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(stream.getvalue()).decode(),
        },
    }
    result = request(messages=[{"role": "user", "content": [block]}])
    assert result.messages[0]["content"][0]["type"] == "image_url"
    from PIL import Image

    data_uri = result.messages[0]["content"][0]["image_url"]["url"]
    rendered = Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1])))
    assert rendered.convert("L").getextrema() == (0, 255)


@pytest.mark.asyncio
async def test_stream_reports_initial_and_final_input_and_cache_usage():
    async def chunks():
        yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        yield {
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 60},
                "cache_creation_input_tokens": 10,
            },
        }

    events = [
        json.loads(e["data"])
        async for e in anthropic_stream_events(
            chunks(), "local", "req", input_tokens=100
        )
    ]
    assert events[0]["message"]["usage"]["input_tokens"] == 0
    assert events[-2]["usage"] == {
        "input_tokens": 30,
        "output_tokens": 2,
        "cache_read_input_tokens": 60,
        "cache_creation_input_tokens": 10,
    }
    result = openai_to_anthropic(
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 60},
                "cache_creation_input_tokens": 10,
            },
        },
        "local",
    )
    assert result["usage"] == events[-2]["usage"]


@pytest.mark.asyncio
async def test_signed_reasoning_stream_and_history_round_trip(monkeypatch):
    from xinference.api.protocols import anthropic_thinking

    monkeypatch.setattr(anthropic_thinking, "_key", lambda: b"test-key")

    async def chunks():
        yield {
            "choices": [
                {"delta": {"reasoning_content": "think "}, "finish_reason": None}
            ]
        }
        yield {
            "choices": [
                {"delta": {"reasoning_content": "carefully"}, "finish_reason": None}
            ]
        }
        yield {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]}

    events = [
        json.loads(e["data"])
        async for e in anthropic_stream_events(
            chunks(), "local", "req", include_thinking=True
        )
    ]
    signature = next(
        e["delta"]["signature"]
        for e in events
        if e.get("delta", {}).get("type") == "signature_delta"
    )
    assert anthropic_thinking.verify_thinking("think carefully", "local", signature)
    assert [
        e["content_block"]["type"] for e in events if e["type"] == "content_block_start"
    ] == ["thinking", "text"]
    body = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "think carefully",
                    "signature": signature,
                },
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    assert request(messages=body).messages[0]["reasoning_content"] == "think carefully"
    body[0]["content"][0]["thinking"] = "tampered"
    assert "reasoning_content" not in request(messages=body).messages[0]


def pdf_block(pages=1, size=100):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(size, size)
    data = io.BytesIO()
    writer.write(data)
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(data.getvalue()).decode(),
        },
    }


def test_corrupt_pdf_is_a_client_error():
    import base64

    pytest.importorskip("pypdfium2")
    block = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"not a PDF").decode(),
        },
    }
    with pytest.raises(ValueError, match="Invalid PDF document"):
        parse_anthropic_request(
            request(messages=[{"role": "user", "content": [block]}])
        )


def test_pdf_requests_share_the_ocr_lock_and_render_once(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import pypdfium2

    from xinference.api import pdf_ocr

    original = pypdfium2.PdfDocument
    calls = []

    def open_document(data):
        assert pdf_ocr._pdfium_lock.locked()
        calls.append(1)
        return original(data)

    monkeypatch.setattr(pypdfium2, "PdfDocument", open_document)
    block = pdf_block(2)
    block["cache_control"] = {"type": "ephemeral"}

    def parse(_):
        parsed = request(messages=[{"role": "user", "content": [block]}])
        assert len(parsed.messages[0]["content"]) == 2
        assert parsed.cache_control["messages"] == []

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(parse, range(32)))
    assert len(calls) == 32
    assert not pdf_ocr._pdfium_lock.locked()


def test_pdf_total_pages_and_pixels_are_bounded_per_request():
    block = pdf_block(11)
    with pytest.raises(AnthropicProtocolError, match="20 pages per request"):
        request(messages=[{"role": "user", "content": [block, dict(block)]}])
    with pytest.raises(AnthropicProtocolError, match="rendered pixels per request"):
        request(messages=[{"role": "user", "content": [pdf_block(5, 3000)]}])


def test_pdf_decoded_bytes_are_bounded_before_open(monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_MAX_BYTES", "32")
    with pytest.raises(AnthropicProtocolError, match="request byte limit"):
        request(messages=[{"role": "user", "content": [pdf_block()]}])


@pytest.mark.parametrize("citations", [None, {}, {"enabled": False}])
def test_null_document_citations_are_valid(citations):
    block = {
        "type": "document",
        "citations": citations,
        "source": {"type": "text", "media_type": "text/plain", "data": "hello"},
    }
    assert request(messages=[{"role": "user", "content": [block]}]).messages == [
        {"role": "user", "content": "hello"}
    ]


@pytest.mark.parametrize("kind", ["unsigned", "foreign", "other_model", "new_key"])
def test_unverifiable_thinking_does_not_block_answer_or_tool_history(monkeypatch, kind):
    from xinference.api.protocols import anthropic_thinking

    monkeypatch.setattr(anthropic_thinking, "_key", lambda: b"first")
    signature = anthropic_thinking.sign_thinking("old thought", "local")
    if kind == "unsigned":
        signature = ""
    if kind == "foreign":
        signature = "foreign signature"
    if kind == "other_model":
        signature = anthropic_thinking.sign_thinking("old thought", "old-model")
    if kind == "new_key":
        monkeypatch.setattr(anthropic_thinking, "_key", lambda: b"second")
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "old thought", "signature": signature},
                {"type": "text", "text": "answer"},
                {
                    "type": "tool_use",
                    "id": "call",
                    "name": "Read",
                    "input": {"path": "/tmp/a"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call", "content": "done"}
            ],
        },
    ]
    parsed = request(messages=messages)
    assert parsed.messages[0]["content"] == "answer"
    assert "reasoning_content" not in parsed.messages[0]
    assert parsed.messages[0]["tool_calls"][0]["function"]["name"] == "Read"
    assert parsed.messages[1]["content"] == "done"


@pytest.mark.asyncio
async def test_message_start_and_delta_use_the_same_uncached_token_accounting():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 60},
        "cache_creation_input_tokens": 10,
    }

    async def chunks():
        yield {
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": usage,
        }

    events = [
        json.loads(e["data"])
        async for e in anthropic_stream_events(
            chunks(), "local", "req", input_tokens=100
        )
    ]
    assert events[0]["message"]["usage"] == {
        "input_tokens": 30,
        "output_tokens": 0,
        "cache_read_input_tokens": 60,
        "cache_creation_input_tokens": 10,
    }
    assert events[-2]["usage"]["input_tokens"] == 30
