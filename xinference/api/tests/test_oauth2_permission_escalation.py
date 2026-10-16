"""Regression test for #5058 Finding 2: a caller may only grant permissions
they themselves hold (admins may grant anything).

    pytest xinference/api/tests/test_oauth2_permission_escalation.py -v
"""

import pytest
from fastapi import HTTPException

import xinference.api.oauth2.advanced.routes as routes


def _patch_caller(monkeypatch, scopes):
    monkeypatch.setattr(
        routes,
        "_get_current_user_from_token",
        lambda request, auth: (1, "caller", scopes),
    )


def test_users_manage_cannot_grant_admin(monkeypatch):
    _patch_caller(monkeypatch, ["users:manage"])
    with pytest.raises(HTTPException) as exc:
        routes._reject_permission_escalation(None, None, ["admin", "keys:manage"])
    assert exc.value.status_code == 403


def test_cannot_grant_unheld_scope(monkeypatch):
    _patch_caller(monkeypatch, ["users:manage"])
    with pytest.raises(HTTPException):
        routes._reject_permission_escalation(None, None, ["models:write"])


def test_admin_can_grant_anything(monkeypatch):
    _patch_caller(monkeypatch, ["admin"])
    routes._reject_permission_escalation(
        None, None, ["admin", "keys:manage", "models:read"]
    )


def test_granting_held_permissions_is_allowed(monkeypatch):
    _patch_caller(monkeypatch, ["users:manage", "models:read"])
    routes._reject_permission_escalation(None, None, ["users:manage", "models:read"])


@pytest.mark.parametrize("bad", [None, "admin", 123, True, ["admin", 1]])
def test_non_list_of_strings_rejected(monkeypatch, bad):
    _patch_caller(monkeypatch, ["admin"])
    with pytest.raises(HTTPException) as exc:
        routes._reject_permission_escalation(None, None, bad)
    assert exc.value.status_code == 400


def test_none_caller_scopes_does_not_crash(monkeypatch):
    _patch_caller(monkeypatch, None)
    with pytest.raises(HTTPException) as exc:
        routes._reject_permission_escalation(None, None, ["admin"])
    assert exc.value.status_code == 403
