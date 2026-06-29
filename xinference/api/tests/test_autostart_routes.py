# Copyright 2022-2026 XProbe Inc.
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

import inspect
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

from xinference.api.routers import models


def _register_and_capture(is_auth):
    captured = {}

    def add_api_route(path, endpoint, methods=None, **kwargs):
        captured[(path, tuple(methods or []))] = endpoint

    api = MagicMock()
    api._router.add_api_route.side_effect = add_api_route
    api._auth_service = MagicMock()
    api.is_authenticated.return_value = is_auth
    models.register_routes(api)
    return api, captured


def test_auth_off_autostart_post_handler_exposes_no_user_param():
    _, captured = _register_and_capture(is_auth=False)
    post_handler = captured[("/v1/autostart/models", ("POST",))]
    assert "user" not in inspect.signature(post_handler).parameters


def test_auth_off_autostart_get_handler_exposes_no_user_param():
    _, captured = _register_and_capture(is_auth=False)
    get_handler = captured[("/v1/autostart/models", ("GET",))]
    assert "user" not in inspect.signature(get_handler).parameters


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("is_auth", [False, True])
def test_autostart_model_handlers_return_type_is_response(is_auth, method):
    _, captured = _register_and_capture(is_auth=is_auth)
    handler = captured[("/v1/autostart/models", (method,))]

    return_type = get_type_hints(handler)["return"]

    assert issubclass(return_type, Response)


@pytest.mark.asyncio
async def test_auth_off_autostart_get_handler_forwards_none_user():
    api, captured = _register_and_capture(is_auth=False)
    api.get_autostart_config = AsyncMock(return_value={"models": []})
    get_handler = captured[("/v1/autostart/models", ("GET",))]

    response = await get_handler()

    assert response == {"models": []}
    api.get_autostart_config.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_auth_on_autostart_get_handler_forwards_user():
    api, captured = _register_and_capture(is_auth=True)
    api.get_autostart_config = AsyncMock(return_value={"models": []})
    get_handler = captured[("/v1/autostart/models", ("GET",))]

    response = await get_handler(user={"username": "alice"})

    assert response == {"models": []}
    api.get_autostart_config.assert_awaited_once_with({"username": "alice"})


@pytest.mark.asyncio
async def test_auth_off_autostart_post_handler_forwards_none_user():
    api, captured = _register_and_capture(is_auth=False)
    api.upsert_autostart_model = AsyncMock(return_value={"models": []})
    post_handler = captured[("/v1/autostart/models", ("POST",))]
    request = MagicMock()

    response = await post_handler(request=request)

    assert response == {"models": []}
    api.upsert_autostart_model.assert_awaited_once_with(request, None)


@pytest.mark.asyncio
async def test_auth_on_autostart_post_handler_forwards_user():
    api, captured = _register_and_capture(is_auth=True)
    api.upsert_autostart_model = AsyncMock(return_value={"models": []})
    post_handler = captured[("/v1/autostart/models", ("POST",))]
    request = MagicMock()

    response = await post_handler(request=request, user={"username": "alice"})

    assert response == {"models": []}
    api.upsert_autostart_model.assert_awaited_once_with(request, {"username": "alice"})


def test_autostart_full_config_put_route_is_not_registered():
    _, captured = _register_and_capture(is_auth=False)
    assert ("/v1/autostart/models", ("PUT",)) not in captured
