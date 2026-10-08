"""Tests for required, globally unique API key names and legacy migration."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from xinference.api.oauth2.advanced.auth_service import AdvancedAuthService
from xinference.api.oauth2.advanced.database import (
    API_KEY_NAME_CONFLICT_MESSAGE,
    API_KEY_NAME_REQUIRED_MESSAGE,
    ApiKeyNameConflictError,
    ApiKeyNameRequiredError,
    Database,
)
from xinference.api.oauth2.advanced.migrate import migrate
from xinference.api.oauth2.advanced.routes import register_advanced_auth_routes


def _create_user(service: AdvancedAuthService, username: str) -> int:
    return service.db.create_user(
        username=username,
        password_hash=None,
        source="local",
        permissions=["keys:create", "keys:manage"],
    )


def _make_service(tmp_path: Path) -> AdvancedAuthService:
    return AdvancedAuthService(
        db_path=str(tmp_path / "auth.db"),
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )


def _make_client(service: AdvancedAuthService) -> TestClient:
    app = FastAPI()
    router = APIRouter()
    app.state.advanced_auth = service
    register_advanced_auth_routes(cast(Any, SimpleNamespace(_app=app, _router=router)))
    app.include_router(router)
    return TestClient(app)


def _auth_headers(service: AdvancedAuthService, user_id: int, username: str):
    permissions = ["keys:create", "keys:manage"]
    token = service.create_access_token(user_id, username, permissions)
    return {"Authorization": f"Bearer {token}"}


def test_database_requires_and_normalizes_api_key_name(tmp_path):
    service = _make_service(tmp_path)
    user_id = _create_user(service, "alice")

    created = service.create_api_key_for_user(
        user_id=user_id, name="  Production-Key  "
    )
    assert created["name"] == "Production-Key"
    assert service.db.get_api_key_by_id(created["id"])["name"] == "Production-Key"

    with pytest.raises(ApiKeyNameRequiredError, match=API_KEY_NAME_REQUIRED_MESSAGE):
        service.create_api_key_for_user(user_id=user_id, name=" \t\n ")


def test_database_enforces_global_case_insensitive_uniqueness(tmp_path):
    service = _make_service(tmp_path)
    alice_id = _create_user(service, "alice")
    bob_id = _create_user(service, "bob")
    service.create_api_key_for_user(user_id=alice_id, name="Production-Key")

    with pytest.raises(ApiKeyNameConflictError, match=API_KEY_NAME_CONFLICT_MESSAGE):
        service.create_api_key_for_user(user_id=bob_id, name=" production-key ")


def test_database_update_allows_own_name_and_rejects_another_name(tmp_path):
    service = _make_service(tmp_path)
    user_id = _create_user(service, "alice")
    first = service.create_api_key_for_user(user_id=user_id, name="First")
    second = service.create_api_key_for_user(user_id=user_id, name="Second")

    assert service.db.update_api_key(first["id"], name=" first ")
    assert service.db.get_api_key_by_id(first["id"])["name"] == "first"

    with pytest.raises(ApiKeyNameConflictError, match=API_KEY_NAME_CONFLICT_MESSAGE):
        service.db.update_api_key(second["id"], name="FIRST")


def test_concurrent_duplicate_creation_only_allows_one_name(tmp_path):
    db_path = str(tmp_path / "auth.db")
    primary_service = AdvancedAuthService(
        db_path=db_path,
        jwt_secret_key="unit-test-secret",
        encryption_key="unit-test-encryption-key",
    )
    user_id = _create_user(primary_service, "alice")

    def create_from_separate_service(index: int) -> str:
        service = AdvancedAuthService(
            db_path=db_path,
            jwt_secret_key="unit-test-secret",
            encryption_key="unit-test-encryption-key",
        )
        try:
            service.create_api_key_for_user(
                user_id=user_id,
                name="Concurrent-Key" if index == 0 else "concurrent-key",
            )
            return "created"
        except ApiKeyNameConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(create_from_separate_service, range(2)))

    assert sorted(results) == ["conflict", "created"]
    matching = [
        key
        for key in primary_service.db.list_api_keys()
        if key["name"].lower() == "concurrent-key"
    ]
    assert len(matching) == 1


def test_api_key_name_routes_return_400_and_409(tmp_path):
    service = _make_service(tmp_path)
    user_id = _create_user(service, "alice")
    headers = _auth_headers(service, user_id, "alice")
    client = _make_client(service)

    missing = client.post("/v1/admin/keys", json={}, headers=headers)
    assert missing.status_code == 400
    assert missing.json()["detail"] == API_KEY_NAME_REQUIRED_MESSAGE

    created = client.post(
        "/v1/admin/keys", json={"name": "  Production-Key  "}, headers=headers
    )
    assert created.status_code == 201
    key_id = created.json()["id"]
    assert created.json()["name"] == "Production-Key"
    assert service.db.get_api_key_by_id(key_id)["name"] == "Production-Key"

    duplicate = client.post(
        "/v1/admin/keys", json={"name": "production-key"}, headers=headers
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == API_KEY_NAME_CONFLICT_MESSAGE

    second = client.post(
        "/v1/admin/keys", json={"name": "Staging-Key"}, headers=headers
    )
    assert second.status_code == 201
    second_key_id = second.json()["id"]

    update_conflict = client.put(
        f"/v1/admin/keys/{second_key_id}",
        json={"name": " PRODUCTION-KEY "},
        headers=headers,
    )
    assert update_conflict.status_code == 409
    assert update_conflict.json()["detail"] == API_KEY_NAME_CONFLICT_MESSAGE

    own_name_update = client.put(
        f"/v1/admin/keys/{key_id}",
        json={"name": " production-key "},
        headers=headers,
    )
    assert own_name_update.status_code == 200
    assert service.db.get_api_key_by_id(key_id)["name"] == "production-key"

    blank_update = client.put(
        f"/v1/admin/keys/{key_id}", json={"name": "   "}, headers=headers
    )
    assert blank_update.status_code == 400
    assert blank_update.json()["detail"] == API_KEY_NAME_REQUIRED_MESSAGE

    enabled_update = client.put(
        f"/v1/admin/keys/{key_id}", json={"enabled": False}, headers=headers
    )
    assert enabled_update.status_code == 200


def _create_legacy_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                password_hash TEXT,
                source TEXT NOT NULL DEFAULT 'local',
                enabled INTEGER DEFAULT 1,
                must_change_password INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_encrypted TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT,
                description TEXT,
                enabled INTEGER DEFAULT 1,
                expires_at TIMESTAMP,
                encryption_version INTEGER DEFAULT 1,
                rate_limit_max_failures INTEGER,
                rate_limit_window_seconds INTEGER,
                rate_limit_ban_seconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO users (id, username) VALUES (1, 'legacy');
            INSERT INTO api_keys
                (id, user_id, key_hash, key_encrypted, key_prefix, name)
            VALUES
                (1, 1, 'hash-1', 'encrypted-1', 'prefix1', NULL),
                (2, 1, 'hash-2', 'encrypted-2', 'prefix2', ' Production '),
                (3, 1, 'hash-3', 'encrypted-3', 'prefix3', 'production'),
                (4, 1, 'hash-4', 'encrypted-4', 'prefix4', 'api-key-1'),
                (5, 1, 'hash-5', 'encrypted-5', 'prefix5', '   ');
            """
        )


def test_legacy_api_key_names_are_migrated_deterministically(tmp_path):
    db_path = tmp_path / "legacy.db"
    _create_legacy_database(db_path)

    database = Database(str(db_path))
    names = {key["id"]: key["name"] for key in database.list_api_keys()}
    assert names == {
        1: "api-key-1",
        2: "Production",
        3: "production-3",
        4: "api-key-1-4",
        5: "api-key-5",
    }

    # Re-running initialization is idempotent.
    Database(str(db_path))
    assert {key["id"]: key["name"] for key in database.list_api_keys()} == names

    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO api_keys
                   (user_id, key_hash, key_encrypted, key_prefix, name)
                   VALUES (1, 'hash-6', 'encrypted-6', 'prefix6', NULL)"""
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO api_keys
                   (user_id, key_hash, key_encrypted, key_prefix, name)
                   VALUES (1, 'hash-7', 'encrypted-7', 'prefix7', 'PRODUCTION')"""
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO api_keys
                   (user_id, key_hash, key_encrypted, key_prefix, name)
                   VALUES (1, 'hash-8', 'encrypted-8', 'prefix8', ' \t\n ')"""
            )


def test_json_auth_migration_generates_unique_names_for_matching_prefixes(tmp_path):
    config_path = tmp_path / "auth.json"
    db_path = tmp_path / "migrated.db"
    config_path.write_text(
        json.dumps(
            {
                "user_config": [
                    {
                        "username": "legacy",
                        "password": "legacy-password",
                        "permissions": [],
                        "api_keys": ["sk-same-alpha", "sk-same-beta"],
                    }
                ]
            }
        )
    )

    migrate(
        auth_config_path=str(config_path),
        db_path=str(db_path),
        encryption_key="unit-test-encryption-key",
    )

    names = [key["name"] for key in Database(str(db_path)).list_api_keys()]
    assert names == ["migrated-sk-same", "migrated-sk-same-2"]
