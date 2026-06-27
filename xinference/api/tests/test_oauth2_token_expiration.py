"""Regression test for #5058 Finding 1: JWTs issued by create_access_token
must be rejected once expired (the decode path now uses verify_exp=True).

    pytest xinference/api/tests/test_oauth2_token_expiration.py -v
"""
from datetime import timedelta

import pytest
from jose import ExpiredSignatureError, jwt

from xinference.api.oauth2.utils import create_access_token

_SECRET = "unit-test-secret"
_ALG = "HS256"


def _decode(token: str):
    # same decode contract as AuthService.__call__ (verify_exp enabled by the fix)
    return jwt.decode(token, _SECRET, algorithms=[_ALG], options={"verify_exp": True})


def test_expired_token_is_rejected():
    token = create_access_token(
        data={"sub": "helpdesk", "scopes": ["users:manage"]},
        secret_key=_SECRET,
        algorithm=_ALG,
        expires_delta=timedelta(minutes=-5),
    )
    with pytest.raises(ExpiredSignatureError):
        _decode(token)


def test_valid_token_is_accepted():
    token = create_access_token(
        data={"sub": "alice", "scopes": ["models:read"]},
        secret_key=_SECRET,
        algorithm=_ALG,
        expires_delta=timedelta(minutes=30),
    )
    payload = _decode(token)
    assert payload["sub"] == "alice"
    assert payload["scopes"] == ["models:read"]
