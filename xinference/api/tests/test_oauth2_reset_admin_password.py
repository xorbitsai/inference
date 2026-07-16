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
"""Tests for the offline ``xinference-reset-auth-password`` command.

pytest xinference/api/tests/test_oauth2_reset_admin_password.py -v
"""

import pytest

from xinference.api.oauth2.advanced.crypto import get_password_hash, verify_password
from xinference.api.oauth2.advanced.database import Database
from xinference.api.oauth2.advanced.reset_password import reset_admin_password


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "auth.db"))
    database.create_user(
        username="admin",
        password_hash=get_password_hash("original-pass"),
        source="local",
        enabled=1,
        must_change_password=1,
        permissions=["admin", "users:manage", "models:read"],
    )
    database.create_user(
        username="alice",
        password_hash=get_password_hash("alice-pass"),
        source="local",
        permissions=["models:read"],
    )
    return database


def test_reset_sets_new_password(db):
    reset_admin_password(db, "admin", "new-secret-pass")

    admin = db.get_user_by_username("admin")
    assert verify_password("new-secret-pass", admin["password_hash"])
    assert not verify_password("original-pass", admin["password_hash"])


def test_reset_revokes_refresh_tokens(db):
    admin = db.get_user_by_username("admin")
    db.create_refresh_token(admin["id"], "tokenhash", "2099-01-01 00:00:00")
    assert db.get_refresh_token("tokenhash") is not None

    reset_admin_password(db, "admin", "new-secret-pass")

    assert db.get_refresh_token("tokenhash") is None


def test_reset_clears_must_change_password(db):
    # update_password_and_revoke_tokens sets must_change_password = 0, so the
    # reset admin can log in with the new password without a forced change.
    reset_admin_password(db, "admin", "new-secret-pass")

    admin = db.get_user_by_username("admin")
    assert admin["must_change_password"] == 0


def test_reset_rejects_non_admin(db):
    with pytest.raises(ValueError, match="not an admin"):
        reset_admin_password(db, "alice", "new-secret-pass")

    alice = db.get_user_by_username("alice")
    assert verify_password("alice-pass", alice["password_hash"])


def test_reset_rejects_unknown_user(db):
    with pytest.raises(ValueError, match="not found"):
        reset_admin_password(db, "ghost", "new-secret-pass")


def test_reset_rejects_short_password(db):
    with pytest.raises(ValueError, match="at least"):
        reset_admin_password(db, "admin", "short")

    admin = db.get_user_by_username("admin")
    assert verify_password("original-pass", admin["password_hash"])


def test_reset_bumps_token_version(db):
    admin = db.get_user_by_username("admin")
    before = db.get_user_token_version(admin["id"])

    reset_admin_password(db, "admin", "new-secret-pass")

    after = db.get_user_token_version(admin["id"])
    assert after == before + 1


def test_pre_reset_access_token_is_rejected(tmp_path):
    """Regression: an access token minted before the reset must stop working
    once the password is reset, even though the JWT itself has not expired.
    Resetting bumps the user's token_version, and verify_access_token checks
    it against the database on every call."""
    from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService

    auth = AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    user_id = auth.db.create_user(
        username="admin",
        password_hash=get_password_hash("original-pass"),
        source="local",
        enabled=1,
        permissions=["admin", "users:manage"],
    )

    user = auth.db.get_user_by_id(user_id)
    token = auth.create_access_token(
        user["id"], user["username"], user["permissions"], user["token_version"]
    )

    # Valid before the reset: scopes are honored.
    payload = auth.verify_access_token(token)
    assert payload is not None
    assert "admin" in payload["scopes"]

    reset_admin_password(auth.db, "admin", "new-secret-pass")

    # After the reset the same (unexpired) token is rejected.
    assert auth.verify_access_token(token) is None

    # A token minted after the reset works again.
    user = auth.db.get_user_by_id(user_id)
    fresh = auth.create_access_token(
        user["id"], user["username"], user["permissions"], user["token_version"]
    )
    assert auth.verify_access_token(fresh) is not None
