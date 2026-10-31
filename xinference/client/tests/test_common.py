from contextlib import AsyncExitStack, closing
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
import requests

from ..common import async_streaming_response_iterator, streaming_response_iterator
from ..restful.async_restful_client import AsyncClient
from ..restful.restful_client import Client

_ENDPOINTS = [
    ("http://localhost:9997", "http://localhost:9997"),
    ("http://localhost:9997/", "http://localhost:9997"),
    ("http://localhost:9997//", "http://localhost:9997"),
    ("http://localhost:9997/proxy/path", "http://localhost:9997/proxy/path"),
    ("http://localhost:9997/proxy/path/", "http://localhost:9997/proxy/path"),
    ("http://localhost:9997/proxy/path//", "http://localhost:9997/proxy/path"),
]


@pytest.mark.parametrize("endpoint, base_url", _ENDPOINTS)
def test_client_normalizes_endpoint(monkeypatch, endpoint, base_url):
    session = MagicMock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.side_effect = [
        {"auth": True},
        {"model_type": "embedding"},
    ]
    session.post.return_value.status_code = 200
    monkeypatch.setattr(requests, "Session", lambda: session)
    headers = {"Authorization": "Bearer test-key"}

    with closing(Client(endpoint, api_key="test-key")) as client:
        session.get.assert_called_once_with(f"{base_url}/v1/cluster/auth")
        assert client.base_url == base_url
        assert client._cluster_authed is True
        with closing(client.get_model("model")) as handle:
            session.get.assert_called_with(
                f"{base_url}/v1/models/model", headers=headers
            )
            assert handle._base_url == base_url
            handle.create_embedding("hello")
            session.post.assert_called_once_with(
                f"{base_url}/v1/embeddings",
                json={"model": "model", "input": "hello"},
                headers=headers,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint, base_url", _ENDPOINTS)
async def test_async_client_normalizes_endpoint(monkeypatch, endpoint, base_url):
    auth_session = MagicMock()
    auth_session.get.return_value.status_code = 200
    auth_session.get.return_value.json.return_value = {"auth": True}
    monkeypatch.setattr(requests, "Session", lambda: auth_session)
    response = MagicMock(status=200)
    response.json = AsyncMock(return_value={"model_type": "embedding"})
    response.wait_for_close = AsyncMock()
    get = AsyncMock(return_value=response)
    post = AsyncMock(return_value=response)
    monkeypatch.setattr(aiohttp.ClientSession, "get", get)
    monkeypatch.setattr(aiohttp.ClientSession, "post", post)
    headers = {"Authorization": "Bearer test-key"}

    async with AsyncExitStack() as stack:
        client = AsyncClient(endpoint, api_key="test-key")
        stack.push_async_callback(client.close)
        auth_session.get.assert_called_once_with(f"{base_url}/v1/cluster/auth")
        assert client.base_url == base_url
        assert client._cluster_authed is True
        handle = await client.get_model("model")
        stack.push_async_callback(handle.close)
        get.assert_awaited_once_with(f"{base_url}/v1/models/model", headers=headers)
        assert handle._base_url == base_url
        await handle.create_embedding("hello")
        post.assert_awaited_once_with(
            f"{base_url}/v1/embeddings",
            json={"model": "model", "input": "hello"},
            headers=headers,
        )


class _FakeContent:
    def __init__(self, lines):
        self._lines = lines

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for line in self._lines:
            yield line


class _FakeResponse:
    def __init__(self, lines):
        self.content = _FakeContent(lines)
        self.released = False
        self.closed = False

    def release(self):
        self.released = True

    async def wait_for_close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_async_streaming_response_iterator_releases_response_on_close():
    response = _FakeResponse(
        [
            b'data: {"choices": [{"text": "first"}]}',
            b'data: {"choices": [{"text": "second"}]}',
        ]
    )
    stream = async_streaming_response_iterator(response)

    first = await anext(stream)
    await stream.aclose()

    assert first["choices"][0]["text"] == "first"
    assert response.released is True
    assert response.closed is True


def test_streaming_response_iterator_stops_at_done():
    lines = [
        b'data: {"choices": [{"text": "first"}]}',
        b"data: [DONE]",
        b'data: {"choices": [{"text": "after done"}]}',
    ]

    assert list(streaming_response_iterator(iter(lines))) == [
        {"choices": [{"text": "first"}]}
    ]


@pytest.mark.asyncio
async def test_async_streaming_response_iterator_stops_at_done():
    response = _FakeResponse(
        [
            b'data: {"choices": [{"text": "first"}]}',
            b"data: [DONE]",
            b'data: {"choices": [{"text": "after done"}]}',
        ]
    )

    result = [chunk async for chunk in async_streaming_response_iterator(response)]

    assert result == [{"choices": [{"text": "first"}]}]
    assert response.released is True
    assert response.closed is True
