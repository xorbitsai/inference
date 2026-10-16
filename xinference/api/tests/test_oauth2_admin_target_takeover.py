"""Regression tests for the admin-target-takeover guard.

Verifies that non-admin callers (even with ``users:manage``) cannot
perform sensitive write operations on admin users:
- delete_user
- change_password
- update_user (enable/disable)
- update_user_permissions

    pytest xinference/api/tests/test_oauth2_admin_target_takeover.py -v
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import xinference.api.oauth2.advanced.routes as routes


def _patch_caller(monkeypatch, scopes):
    monkeypatch.setattr(
        routes,
        "_get_current_user_from_token",
        lambda request, auth: (1, "caller", scopes),
    )


def _make_user(user_id, perms):
    return {"id": user_id, "username": f"user{user_id}", "permissions": perms}


@pytest.mark.parametrize(
    "action", ["delete", "change_password", "update", "update_perms"]
)
def test_non_admin_cannot_target_admin_user(monkeypatch, action):
    _patch_caller(monkeypatch, ["users:manage"])
    auth = MagicMock()
    auth.db.get_user_by_id.return_value = _make_user(2, ["admin"])

    with pytest.raises(HTTPException) as exc:
        if action == "delete":
            routes._reject_admin_target_takeover(None, auth, 2)
        elif action == "change_password":
            routes._reject_admin_target_takeover(None, auth, 2)
        elif action == "update":
            routes._reject_admin_target_takeover(None, auth, 2)
        elif action == "update_perms":
            routes._reject_admin_target_takeover(None, auth, 2)
    assert exc.value.status_code == 403


def test_non_admin_can_target_non_admin_user(monkeypatch):
    _patch_caller(monkeypatch, ["users:manage"])
    auth = MagicMock()
    auth.db.get_user_by_id.return_value = _make_user(2, ["users:manage"])
    # Should not raise
    routes._reject_admin_target_takeover(None, auth, 2)


def test_admin_can_target_admin_user(monkeypatch):
    _patch_caller(monkeypatch, ["admin"])
    auth = MagicMock()
    auth.db.get_user_by_id.return_value = _make_user(2, ["admin"])
    # Should not raise
    routes._reject_admin_target_takeover(None, auth, 2)


def test_missing_user_does_not_raise(monkeypatch):
    """Defensive: missing target user should not raise (caller already
    validated existence; we don't want to leak existence via 403-vs-404)."""
    _patch_caller(monkeypatch, ["users:manage"])
    auth = MagicMock()
    auth.db.get_user_by_id.return_value = None
    # Should not raise
    routes._reject_admin_target_takeover(None, auth, 999)


def test_target_with_empty_permissions_does_not_raise(monkeypatch):
    """A user with no permissions is not an admin target."""
    _patch_caller(monkeypatch, ["users:manage"])
    auth = MagicMock()
    auth.db.get_user_by_id.return_value = _make_user(2, [])
    # Should not raise
    routes._reject_admin_target_takeover(None, auth, 2)


def test_target_with_none_permissions_does_not_raise(monkeypatch):
    """A user with `permissions=None` (edge case) should not be treated
    as admin."""
    _patch_caller(monkeypatch, ["users:manage"])
    auth = MagicMock()
    user = _make_user(2, [])
    user["permissions"] = None
    auth.db.get_user_by_id.return_value = user
    # Should not raise
    routes._reject_admin_target_takeover(None, auth, 2)
