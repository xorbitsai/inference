"""Regression test for #5058 Finding 4: resetting a user's password must
revoke that user's outstanding refresh tokens, so a leaked refresh token
cannot keep minting access tokens after the reset.

The revocation must also hold under concurrency: token rotation and the
password-reset revocation each run in a single ``BEGIN IMMEDIATE`` write
transaction, so a refresh in flight cannot slip a freshly-rotated token past
the revocation and survive the reset.

    pytest xinference/api/tests/test_oauth2_password_reset_revokes_tokens.py -v
"""

import asyncio
import threading

import pytest

import xinference.api.oauth2.advanced.routes as routes
from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.crypto import sha256_hex
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
    return asyncio.run(coro)


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


def test_rotate_refresh_token_is_atomic(auth):
    """A rotation only succeeds if the old token still existed, and it swaps
    old-for-new in one step (no window where both or neither exist)."""
    user_id = auth.db.create_user(
        username="alice", password_hash="x", source="local", permissions=[]
    )
    auth.db.create_refresh_token(user_id, "old", "2999-01-01 00:00:00")

    row = auth.db.rotate_refresh_token("old", "new", "2999-01-01 00:00:00")
    assert row is not None and row["user_id"] == user_id
    assert auth.db.get_refresh_token("old") is None
    assert auth.db.get_refresh_token("new") is not None

    # Rotating an already-revoked token is a no-op that reports failure and
    # does not resurrect a session.
    assert auth.db.rotate_refresh_token("old", "new2", "2999-01-01 00:00:00") is None
    assert auth.db.get_refresh_token("new2") is None


def _make_service(tmp_path):
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def test_concurrent_refresh_and_password_reset_leaves_no_live_token(tmp_path):
    """Reproduce the reported race: while a refresh rotates a user's token, an
    admin resets that user's password. After both complete, no refresh token
    for the user may survive -- otherwise the attacker keeps a live session.

    Run many rounds so the two BEGIN IMMEDIATE transactions interleave in
    different orders across the SQLite write lock.
    """
    service = _make_service(tmp_path)
    db = service.db
    user_id = db.create_user(
        username="alice",
        password_hash="x",
        source="local",
        permissions=["models:read"],
    )

    survivors = []
    for i in range(60):
        # Clear any leftover tokens and seed one fresh refresh token.
        db.delete_user_refresh_tokens(user_id)
        rt = f"seed-token-{i}"
        db.create_refresh_token(user_id, sha256_hex(rt), "2999-01-01 00:00:00")

        barrier = threading.Barrier(2)

        def do_refresh():
            barrier.wait()
            service.refresh_access_token(rt)

        def do_reset():
            barrier.wait()
            db.update_password_and_revoke_tokens(user_id, f"newhash-{i}")

        t1 = threading.Thread(target=do_refresh)
        t2 = threading.Thread(target=do_reset)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # The password reset is the last-writer-wins authority here: whatever
        # ordering happened, once the reset's revocation has committed there
        # must be no refresh token left for this user. If the non-atomic race
        # were present, the refresh's INSERT could land after the DELETE and a
        # token would survive.
        remaining = db.get_user_refresh_tokens(user_id)
        if remaining:
            survivors.append((i, remaining))

    assert not survivors, f"refresh token(s) survived a password reset: {survivors}"
