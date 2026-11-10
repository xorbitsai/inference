# CI: linux
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

from unittest.mock import AsyncMock

import httpx
import pytest

from .. import restful_api
from .test_llm_request_abort import _new_api
from .test_model_request_logging import _app

ROUTES = {
    "/v1/completions": ("create_completion", '{"model": "m", "prompt": "hi"'),
    "/v1/chat/completions": (
        "create_chat_completion",
        '{"model": "m", "messages": [{"role": "user", "content": "hi"}]',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ROUTES)
@pytest.mark.parametrize(
    "body,detail",
    [
        ("truncated", "Request body must be valid JSON"),
        ("truncated-stream", "Request body must be valid JSON"),
        ("[]", "Request body must be a JSON object"),
        ("null", "Request body must be a JSON object"),
    ],
)
async def test_llm_routes_reject_a_body_that_is_not_a_json_object(
    monkeypatch, path, body, detail
):
    handler, prefix = ROUTES[path]
    if body == "truncated":
        body = prefix
    elif body == "truncated-stream":
        body = prefix + ', "stream": true'
    require_model = AsyncMock()
    monkeypatch.setattr(restful_api, "require_model", require_model)
    api = _new_api()
    api._get_supervisor_ref = AsyncMock()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=_app(path, getattr(api, handler)), raise_app_exceptions=False
        ),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path, content=body, headers={"content-type": "application/json"}
        )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    require_model.assert_not_awaited()
    api._get_supervisor_ref.assert_not_awaited()
