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

from unittest.mock import Mock

import pytest
import requests

from xinference.client.restful import restful_client
from xinference.client.restful.restful_client import RESTfulImageModelHandle


@pytest.mark.parametrize("missing_input", ["second_image", "mask"])
def test_image_edit_closes_files_when_a_later_open_fails(
    tmp_path, monkeypatch, missing_input
):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    missing = str(tmp_path / "missing.png")
    opened = []
    original_open = open

    def track_open(*args, **kwargs):
        file = original_open(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(restful_client, "open", track_open, raising=False)
    handle = RESTfulImageModelHandle("model", "http://localhost:9997", {})
    handle.session.post = Mock()
    try:
        with pytest.raises(FileNotFoundError):
            if missing_input == "second_image":
                handle.image_edit([str(image), missing], "edit")
            else:
                handle.image_edit(str(image), "edit", mask=missing)
        handle.session.post.assert_not_called()
        assert len(opened) == 1
        assert opened[0].closed
    finally:
        for file in opened:
            file.close()
        handle.close()


@pytest.mark.parametrize("outcome", ["success", "http_error", "transport_error"])
def test_image_edit_keeps_files_open_during_upload_and_closes_afterward(
    tmp_path, outcome
):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    uploaded = []
    handle = RESTfulImageModelHandle("model", "http://localhost:9997", {})

    def post(url, *, files, headers):
        file = next(part[1][1] for part in files if part[0] == "images")
        uploaded.append(file)
        assert not file.closed
        assert file.read() == b"image"
        if outcome == "transport_error":
            raise requests.ConnectionError("offline fixture")
        response = Mock(status_code=200 if outcome == "success" else 400)
        response.json.return_value = (
            {"data": []} if outcome == "success" else {"detail": "bad request"}
        )
        return response

    handle.session.post = post
    try:
        if outcome == "success":
            assert handle.image_edit(str(image), "edit") == {"data": []}
        elif outcome == "http_error":
            with pytest.raises(RuntimeError, match="bad request"):
                handle.image_edit(str(image), "edit")
        else:
            with pytest.raises(requests.ConnectionError, match="offline fixture"):
                handle.image_edit(str(image), "edit")
        assert len(uploaded) == 1
        assert uploaded[0].closed
    finally:
        handle.close()
