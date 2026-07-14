"""Regression test for #5058 Finding 4: resetting a user's password must
revoke that user's outstanding refresh tokens, so a leaked refresh token
cannot keep minting access tokens after the reset.

    pytest xinference/api/tests/test_oauth2_password_reset_revokes_tokens.py -v
"""

import asyncio

import pytest

import xinference.api.oauth2.advanced.routes as routes
from xinference.api.oauth2.advanced.database import Database


class _Auth:
    """Minimal stand-in exposing just the ``db`` attribute the route uses."""

    def __init__(self, db):
        self.db = db


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def auth(tmp_path):
    return _Auth(Database(str(tmp_path / "auth.db")))


def _patch(monkeypatch, auth, caller_scopes):
    monkeypatch.setattr(routes, "get_advanced_auth", lambda request: auth)
    # Caller is admin so the admin-target-takeover guard is a no-op and we
    # isolate the refresh-token revocation behaviour under test.
    monkeypatch.setattr(
        routes,
        "_get_current_user_from_token",
        lambda request, a: (1, "caller", caller_scopes),
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_password_reset_revokes_refresh_tokens(monkeypatch, auth):
    user_id = auth.db.create_user(
        username="alice",
        password_hash="x",
        source="local",
        permissions=["models:read"],
    )
    auth.db.create_refresh_token(user_id, "hash-1", "2999-01-01 00:00:00")
    auth.db.create_refresh_token(user_id, "hash-2", "2999-01-01 00:00:00")
    assert auth.db.get_refresh_token("hash-1") is not None
    assert auth.db.get_refresh_token("hash-2") is not None

    _patch(monkeypatch, auth, ["admin"])
    req = _Request({"new_password": "NewPass123!"})
    resp = _run(routes.change_password(user_id, req))
    assert resp.status_code == 200

    # Both outstanding refresh tokens must be gone after the reset.
    assert auth.db.get_refresh_token("hash-1") is None
    assert auth.db.get_refresh_token("hash-2") is None


def test_password_reset_does_not_touch_other_users_tokens(monkeypatch, auth):
    alice = auth.db.create_user(
        username="alice", password_hash="x", source="local", permissions=[]
    )
    bob = auth.db.create_user(
        username="bob", password_hash="x", source="local", permissions=[]
    )
    auth.db.create_refresh_token(alice, "alice-rt", "2999-01-01 00:00:00")
    auth.db.create_refresh_token(bob, "bob-rt", "2999-01-01 00:00:00")

    _patch(monkeypatch, auth, ["admin"])
    _run(routes.change_password(alice, _Request({"new_password": "NewPass123!"})))

    assert auth.db.get_refresh_token("alice-rt") is None
    # Bob's session is unaffected.
    assert auth.db.get_refresh_token("bob-rt") is not None
