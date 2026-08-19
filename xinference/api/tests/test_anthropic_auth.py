from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.security import SecurityScopes

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.crypto import get_password_hash


@pytest.mark.asyncio
async def test_advanced_auth_accepts_x_api_key_as_credential(tmp_path):
    service = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    user_id = service.db.create_user(
        username="anthropic-user",
        password_hash=get_password_hash("pass"),
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=["models:read"],
    )
    token = service.create_access_token(user_id, "anthropic-user", ["models:read"])
    request = MagicMock()
    request.url.path = "/v1/messages"
    request.method = "POST"
    request.headers = {
        "x-api-key": token,
        "content-type": "application/json",
        "content-length": "76",
    }
    request.client.host = "127.0.0.1"
    request.body = AsyncMock(
        return_value=b'{"model":"model","max_tokens":8,"messages":[]}'
    )

    user = await service(
        request,
        SecurityScopes(scopes=["models:read"]),
        token=None,
    )

    assert user["username"] == "anthropic-user"


@pytest.mark.asyncio
async def test_advanced_auth_does_not_accept_x_api_key_on_other_routes(tmp_path):
    service = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    user_id = service.db.create_user(
        username="openai-user",
        password_hash=get_password_hash("pass"),
        source="local",
        enabled=1,
        must_change_password=0,
        permissions=["models:read"],
    )
    token = service.create_access_token(user_id, "openai-user", ["models:read"])
    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    request.method = "POST"
    request.headers = {
        "x-api-key": token,
        "content-type": "application/json",
        "content-length": "32",
    }
    request.client.host = "127.0.0.1"
    request.body = AsyncMock(return_value=b'{"model":"model"}')

    with pytest.raises(Exception) as exc_info:
        await service(
            request,
            SecurityScopes(scopes=["models:read"]),
            token=None,
        )

    assert getattr(exc_info.value, "status_code", None) == 401
