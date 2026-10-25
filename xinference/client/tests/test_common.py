from typing import AsyncIterator

import pytest

from ..common import async_streaming_response_iterator, streaming_response_iterator


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
