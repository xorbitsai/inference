"""Regression tests for rejecting expired OAuth2 JWT access tokens."""

import json
from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import SecurityScopes

from xinference.api.oauth2.auth_service import AuthService
from xinference.api.oauth2.types import AuthConfig, AuthStartupConfig, User
from xinference.api.oauth2.utils import create_access_token

_SECRET = "unit-test-secret"
_ALG = "HS256"


@pytest.fixture
def auth_service(tmp_path):
    auth_config = AuthConfig(
        algorithm=_ALG,
        secret_key=_SECRET,
        token_expire_in_minutes=30,
    )
    user = User(
        username="alice",
        password="pass",
        permissions=["models:read"],
        api_keys=["sk-123456789abcd"],
    )
    startup_config = AuthStartupConfig(auth_config=auth_config, user_config=[user])
    auth_config_file = tmp_path / "auth_config.json"
    auth_config_file.write_text(json.dumps(startup_config.dict()))
    return AuthService(str(auth_config_file))


def test_expired_token_is_rejected(auth_service):
    token = create_access_token(
        data={"sub": "alice", "scopes": ["models:read"]},
        secret_key=_SECRET,
        algorithm=_ALG,
        expires_delta=timedelta(minutes=-5),
    )
    with pytest.raises(HTTPException) as exc_info:
        auth_service(SecurityScopes(scopes=["models:read"]), token)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Could not validate credentials"


def test_valid_token_is_accepted(auth_service):
    token = create_access_token(
        data={"sub": "alice", "scopes": ["models:read"]},
        secret_key=_SECRET,
        algorithm=_ALG,
        expires_delta=timedelta(minutes=30),
    )
    user = auth_service(SecurityScopes(scopes=["models:read"]), token)

    assert user.username == "alice"
    assert user.permissions == ["models:read"]
