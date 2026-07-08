"""Regression tests for the scope-rename + alias-bridge migration.

Verifies that the `SCOPE_ALIASES` bridge table correctly maps legacy
scope names to their current equivalents so tokens carrying legacy
scopes (`models:start`, `models:stop`, `models:add`,
`models:unregister`) continue to pass authorization checks on routes
that now use the new names (`models:write`, `models:register`).

    pytest xinference/api/tests/test_oauth2_scope_alias_migration.py -v
"""

import pytest

from xinference.api.oauth2.scope_aliases import SCOPE_ALIASES, _normalize_scopes

_EXPECTED_ALIASES = {
    "models:start": "models:write",
    "models:stop": "models:write",
    "models:add": "models:register",
    "models:unregister": "models:register",
}


def test_alias_table_covers_legacy_names():
    assert SCOPE_ALIASES == _EXPECTED_ALIASES


def test_normalize_scopes_expands_legacy_to_new():
    result = _normalize_scopes(["models:start", "models:add"])
    assert "models:start" in result  # original kept
    assert "models:add" in result
    assert "models:write" in result  # alias added
    assert "models:register" in result


def test_normalize_scopes_passthrough_new_only():
    result = _normalize_scopes(["models:write", "models:register"])
    assert result == {"models:write", "models:register"}


def test_normalize_scopes_handles_empty_and_none():
    assert _normalize_scopes([]) == set()
    assert _normalize_scopes(None) == set()


def test_normalize_scopes_admin_passthrough():
    result = _normalize_scopes(["admin"])
    assert result == {"admin"}


def test_legacy_token_passes_new_scope_check():
    """A token issued with `models:start` (now legacy) passes a
    `models:write` scope check."""
    token_scopes = _normalize_scopes(["models:start"])
    assert "models:write" in token_scopes


def test_legacy_unregister_passes_new_register_check():
    token_scopes = _normalize_scopes(["models:unregister"])
    assert "models:register" in token_scopes


@pytest.mark.parametrize(
    "legacy,new",
    [
        ("models:start", "models:write"),
        ("models:stop", "models:write"),
        ("models:add", "models:register"),
        ("models:unregister", "models:register"),
    ],
)
def test_each_legacy_alias_mapping(legacy, new):
    assert _normalize_scopes([legacy]) == {legacy, new}
