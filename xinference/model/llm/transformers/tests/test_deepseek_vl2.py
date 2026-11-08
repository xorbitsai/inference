# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# CI: linux

from concurrent.futures import Future

import pytest

from ..multimodal import deepseek_vl2

URLS = ["https://example.test/a.png", "https://example.test/b.png"]


class _DeferredExecutor:
    """Run submitted calls only when the ``with`` block exits."""

    def __init__(self, *args, **kwargs):
        self._calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        for future, fn, args in self._calls:
            try:
                future.set_result(fn(*args))
            except Exception as e:
                future.set_exception(e)

    def submit(self, fn, *args):
        future = Future()
        self._calls.append((future, fn, args))
        return future


class _Response:
    def __init__(self, url, status_code=200):
        self.content = url.encode()
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _content(urls):
    return [{"type": "text", "text": "Describe"}] + [
        {"type": "image_url", "image_url": {"url": url}} for url in urls
    ]


def test_each_remote_image_is_downloaded_from_its_own_url(monkeypatch):
    monkeypatch.setattr(deepseek_vl2, "ThreadPoolExecutor", _DeferredExecutor)
    monkeypatch.setattr(
        deepseek_vl2.requests, "get", lambda url, headers=None: _Response(url)
    )

    text, images = deepseek_vl2.DeepSeekVL2ChatModel._message_content_to_deepseek(
        _content(URLS)
    )

    assert text == "Describe"
    assert [image.getvalue() for image in images] == [url.encode() for url in URLS]


def test_failed_image_download_is_raised(monkeypatch):
    monkeypatch.setattr(deepseek_vl2, "ThreadPoolExecutor", _DeferredExecutor)
    monkeypatch.setattr(
        deepseek_vl2.requests,
        "get",
        lambda url, headers=None: _Response(url, 404 if url == URLS[1] else 200),
    )

    with pytest.raises(RuntimeError, match="HTTP 404"):
        deepseek_vl2.DeepSeekVL2ChatModel._message_content_to_deepseek(_content(URLS))
