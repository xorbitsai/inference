# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Inline PDF attachments on the OpenAI Chat Completions route."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from .. import restful_api
from .test_llm_request_abort import _new_api
from .test_model_request_logging import _app

CHAT_PATH = "/v1/chat/completions"


def _pdf_data(*page_sizes):
    pdfium = pytest.importorskip("pypdfium2")
    with pdfium.PdfDocument.new() as document:
        for width, height in page_sizes:
            document.new_page(width, height).close()
        output = io.BytesIO()
        document.save(output)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _file(data):
    return {"type": "file", "file": {"filename": "report.pdf", "file_data": data}}


@pytest.fixture
async def chat_api(monkeypatch):
    api = _new_api()
    model = SimpleNamespace(
        uid=b"model-0",
        is_vllm_backend=AsyncMock(return_value=False),
        is_transformers_batching_enabled=AsyncMock(return_value=False),
        chat=AsyncMock(return_value=b'{"choices": []}'),
    )
    api._get_supervisor_ref = AsyncMock(
        return_value=SimpleNamespace(
            describe_model=AsyncMock(return_value={"model_family": "test-family"})
        )
    )
    monkeypatch.setattr(restful_api, "require_model", AsyncMock(return_value=model))
    monkeypatch.setattr(restful_api, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    transport = httpx.ASGITransport(app=_app(CHAT_PATH, api.create_chat_completion))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, model, api


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream,prefix", [(False, "data:application/pdf;base64,"), (True, "")]
)
async def test_chat_expands_pdf_pages_in_place(chat_api, stream, prefix):
    client, model, _ = chat_api

    async def chunks():
        yield {"data": '{"choices": []}'}

    if stream:
        model.chat.return_value = chunks()
    first = {"type": "text", "text": "Summarize"}
    last = {"type": "text", "text": "this report."}
    response = await client.post(
        CHAT_PATH,
        json={
            "model": "local",
            "stream": stream,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        first,
                        _file(prefix + _pdf_data((80, 100), (100, 80))),
                        last,
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    if stream:
        assert "data: [DONE]" in response.text
    else:
        assert response.json() == {"choices": []}
    parts = model.chat.await_args.args[0][0]["content"]
    assert parts[0] == first
    assert parts[3] == last
    assert [part["type"] for part in parts] == [
        "text",
        "image_url",
        "image_url",
        "text",
    ]
    sizes = []
    for part in parts[1:3]:
        prefix, data = part["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        with Image.open(io.BytesIO(base64.b64decode(data))) as page:
            sizes.append(page.size)
    assert sizes == [(120, 150), (150, 120)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role,part,detail",
    [
        ("user", _file("data:application/pdf;base64,!!"), "Invalid base64"),
        ("user", _file("data:image/png;base64,YQ=="), "application/pdf"),
        (
            "user",
            {"type": "file", "file": {"file_id": "file-123"}},
            "inline PDF",
        ),
        ("assistant", _file("YQ=="), "only in user messages"),
    ],
)
async def test_chat_rejects_invalid_pdf_parts(chat_api, role, part, detail):
    client, model, api = chat_api
    response = await client.post(
        CHAT_PATH,
        json={"model": "local", "messages": [{"role": role, "content": [part]}]},
    )
    assert response.status_code == 400
    assert detail in response.json()["detail"]
    api._get_supervisor_ref.assert_not_awaited()
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_pdf_page_limit_is_shared_across_messages(chat_api):
    client, model, _ = chat_api
    attachment = _file(_pdf_data(*([(30, 30)] * 11)))
    response = await client.post(
        CHAT_PATH,
        json={
            "model": "local",
            "messages": [
                {"role": "user", "content": [attachment]},
                {"role": "user", "content": [attachment]},
            ],
        },
    )
    assert response.status_code == 400
    assert "at most 20 pages" in response.json()["detail"]
    model.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_pdf_pages_count_toward_model_image_limit(chat_api):
    client, model, _ = chat_api
    model.is_vllm_backend.return_value = True
    model.preflight_multimodal_request = AsyncMock(
        return_value={"limits": {"image": 1}}
    )
    response = await client.post(
        CHAT_PATH,
        json={
            "model": "local",
            "messages": [
                {"role": "user", "content": [_file(_pdf_data((80, 100), (100, 80)))]}
            ],
        },
    )
    assert response.status_code == 400
    assert "received 2" in response.json()["detail"]
    model.chat.assert_not_awaited()
