import asyncio
import hashlib
import json
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, cast

import httpx
import pytest
import yaml  # type: ignore[import-untyped]
from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.security import SecurityScopes
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.api.routers import token_routers
from xinference.api.routers.token_routers import register_routes
from xinference.constants import parse_env_bool
from xinference.core.router_config_store import RouterConfigStore
from xinference.core.router_orchestration import RouterOrchestrationController
from xinference.core.router_registry import RouterRuntimeRegistry
from xinference.core.supervisor import SupervisorActor
from xinference.core.tokenizer_asset_registry import TokenizerAssetRegistry

if TYPE_CHECKING:
    from xinference.api.restful_api import RESTfulAPI

ASSET_ID = "test-external"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_asset_registry(tmp_path: Path, *, allow_custom_path: bool = False):
    asset_root = tmp_path / "tokenizer-assets"
    asset_path = asset_root / ASSET_ID
    asset_path.mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(asset_path / "tokenizer.json"))
    encoding = asset_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    return ' '.join(str(m.get('content', '')) for m in messages)\n",
        encoding="utf-8",
    )
    (asset_path / "asset.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "asset_id": ASSET_ID,
                "display_name": "DeepSeek-V4-Flash-0731",
                "model_family": "deepseek-v4",
                "model_name": "DeepSeek-V4-Flash-0731",
                "revision": "0731",
                "encoding_type": "deepseek_v4",
                "compatible_models": ["DeepSeek-V4-Flash-0731"],
                "required_files": [
                    "tokenizer.json",
                    "encoding/encoding_dsv4.py",
                ],
                "checksums": {
                    "tokenizer.json": f"sha256:{_sha256(asset_path / 'tokenizer.json')}",
                    "encoding/encoding_dsv4.py": f"sha256:{_sha256(encoding / 'encoding_dsv4.py')}",
                },
                "capabilities": {"chat": True, "tools": True, "thinking": True},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "tokenizer-assets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "asset_roots": [str(asset_root)],
                "allow_custom_path": allow_custom_path,
                "assets": [{"asset_id": ASSET_ID, "path": ASSET_ID, "enabled": True}],
            }
        ),
        encoding="utf-8",
    )
    return TokenizerAssetRegistry(str(config_path)), asset_path


def router_payload() -> dict:
    return {
        "router_uid": "router-a",
        "virtual_model_uid": "virtual-model",
        "model_type": "LLM",
        "strategy": "token_budget",
        "tokenizer_asset_id": ASSET_ID,
        "backend_url": "http://xinference.internal:9997",
        "model_aliases": ["virtual-alias"],
        "request_timeout_seconds": 10800,
        "connect_timeout_seconds": 10,
        "backends": {
            "short": {
                "model_uid": "short-model",
                "max_context_tokens": 131072,
                "admission": {
                    "max_active": 8,
                    "max_queue": 32,
                    "queue_timeout_seconds": 5,
                    "retry_after_seconds": 1,
                },
            },
            "long": {
                "model_uid": "long-model",
                "max_context_tokens": 1048576,
                "admission": {
                    "max_active": 1,
                    "max_queue": 2,
                    "queue_timeout_seconds": 30,
                    "retry_after_seconds": 5,
                },
            },
        },
        "routing": {
            "short_threshold_tokens": 131072,
            "context_reserve_tokens": 64,
            "default_output_tokens": 512,
            "thinking_policy": "long",
            "overflow_policy": "reject",
        },
        "tokenization": {
            "executor": "process",
            "multiprocessing_start_method": "spawn",
            "max_workers": 2,
            "max_active": 2,
            "max_queue": 8,
            "queue_timeout_seconds": 5,
            "retry_after_seconds": 1,
        },
    }


def typed_router_payload() -> dict:
    payload = router_payload()
    payload.update(
        {
            "config_version": 2,
            "route_profile": "llm_chat",
            "strategy": "typed_rules",
            "backends": [
                {
                    "id": backend_id,
                    "model_uid": model_uid,
                    "max_context_tokens": context,
                    "admission": {
                        "max_active": max_active,
                        "max_queue": 8,
                        "queue_timeout_seconds": 5,
                        "retry_after_seconds": 1,
                    },
                }
                for backend_id, model_uid, context, max_active in (
                    ("fast", "short-model", 131072, 8),
                    ("tools", "tools-model", 131072, 4),
                    ("reasoning", "reasoning-model", 262144, 2),
                    ("long", "long-model", 1048576, 1),
                )
            ],
            "routing": {
                "evaluation_mode": "first_match",
                "context_reserve_tokens": 64,
                "default_output_tokens": 512,
                "rules": [
                    {
                        "id": "tools-route",
                        "priority": 400,
                        "match": {"tools_present": True},
                        "action": {"type": "route", "backend_id": "tools"},
                    },
                    {
                        "id": "thinking-route",
                        "priority": 300,
                        "match": {"thinking": True},
                        "action": {"type": "route", "backend_id": "reasoning"},
                    },
                    {
                        "id": "short-route",
                        "priority": 200,
                        "match": {"total_tokens_lte": 131072},
                        "action": {"type": "route", "backend_id": "fast"},
                    },
                    {
                        "id": "long-route",
                        "priority": 100,
                        "match": {"total_tokens_gte": 131073},
                        "action": {"type": "route", "backend_id": "long"},
                    },
                ],
                "default_action": {
                    "type": "reject",
                    "reason": "no_compatible_backend",
                },
            },
        }
    )
    return payload


def make_supervisor(tmp_path):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._token_router_store = RouterConfigStore(str(tmp_path / "routers.db"))
    supervisor._token_router_registry = RouterRuntimeRegistry()
    supervisor._token_router_orchestration = RouterOrchestrationController(
        str(tmp_path / "routers.db"), supervisor._token_router_store
    )
    supervisor._tokenizer_asset_registry, _ = make_asset_registry(tmp_path)

    async def list_models(_self):
        return {
            "short-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools", "reasoning", "hybrid"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 131072,
            },
            "long-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools", "reasoning", "hybrid"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 1048576,
            },
            "tools-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 131072,
            },
            "reasoning-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "reasoning"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 262144,
            },
        }

    supervisor.list_models = MethodType(list_models, supervisor)
    return supervisor


class FakeAPI:
    def __init__(
        self,
        supervisor,
        *,
        authenticated: bool,
        auth_service,
        host: str = "xinference-supervisor",
        port: int = 9997,
    ) -> None:
        self._router = APIRouter()
        self._supervisor = supervisor
        self._authenticated = authenticated
        self._auth_service = auth_service
        self._host = host
        self._port = port

    def is_authenticated(self) -> bool:
        return self._authenticated

    async def _get_supervisor_ref(self):
        return self._supervisor


async def unused_auth():
    return {"username": "anonymous"}


def create_app(
    supervisor,
    *,
    authenticated: bool = False,
    auth_service=unused_auth,
    host: str = "xinference-supervisor",
    port: int = 9997,
):
    api = FakeAPI(
        supervisor,
        authenticated=authenticated,
        auth_service=auth_service,
        host=host,
        port=port,
    )
    register_routes(cast("RESTfulAPI", api))
    app = FastAPI()
    app.state.api = api
    app.include_router(api._router)
    return app


@pytest.mark.asyncio
async def test_router_defaults_prefer_configured_backend_url(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "XINFERENCE_TOKEN_ROUTER_DEFAULT_BACKEND_URL",
        "http://internal-supervisor:9997/",
    )
    app = create_app(make_supervisor(tmp_path), host="ignored-supervisor")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/defaults")

    assert response.status_code == 200
    assert response.json() == {
        "backend": {
            "mode": "current_supervisor",
            "display_name": "Current Supervisor",
            "backend_url": "http://internal-supervisor:9997",
            "source": "server_config",
            "available": True,
        }
    }


@pytest.mark.asyncio
async def test_router_defaults_fall_back_to_rest_endpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_DEFAULT_BACKEND_URL", raising=False)
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/defaults")

    assert response.status_code == 200
    assert response.json()["backend"] == {
        "mode": "current_supervisor",
        "display_name": "Current Supervisor",
        "backend_url": "http://xinference-supervisor:9997",
        "source": "rest_endpoint",
        "available": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured_url", "host", "expected_source"),
    [
        ("", "0.0.0.0", "unavailable"),
        (
            "http://xinference-supervisor:9997/v1/chat/completions",
            "xinference-supervisor",
            "server_config",
        ),
    ],
)
async def test_router_defaults_report_unavailable_for_unreliable_endpoint(
    tmp_path, monkeypatch, configured_url, host, expected_source
) -> None:
    if configured_url:
        monkeypatch.setenv(
            "XINFERENCE_TOKEN_ROUTER_DEFAULT_BACKEND_URL", configured_url
        )
    else:
        monkeypatch.delenv("XINFERENCE_TOKEN_ROUTER_DEFAULT_BACKEND_URL", raising=False)
    app = create_app(make_supervisor(tmp_path), host=host)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/defaults")

    assert response.status_code == 200
    backend = response.json()["backend"]
    assert backend["available"] is False
    assert backend["backend_url"] is None
    assert backend["source"] == expected_source
    assert backend["error"]


@pytest.mark.asyncio
async def test_backend_url_is_normalized_and_rejects_non_base_urls(tmp_path) -> None:
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = router_payload()
        payload["backend_url"] = "http://xinference.internal:9997/"
        created = await client.post("/v1/token_routers", json=payload)
        assert created.status_code == 201
        assert created.json()["backend_url"] == "http://xinference.internal:9997"

        for invalid_url in (
            "ftp://xinference.internal:9997",
            "http://user:password@xinference.internal:9997",
            "http://xinference.internal:9997/v1/chat/completions",
            "http://xinference.internal:9997?target=router",
        ):
            invalid = router_payload()
            invalid["router_uid"] = f"invalid-{len(invalid_url)}"
            invalid["backend_url"] = invalid_url
            response = await client.post("/v1/token_routers", json=invalid)
            assert response.status_code == 422


@pytest.mark.asyncio
async def test_management_crud_revision_and_validation(tmp_path) -> None:
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = router_payload()
        created = await client.post("/v1/token_routers", json=payload)
        assert created.status_code == 201
        assert created.json()["revision"] == 1
        assert created.json()["enabled"] is False
        assert created.json()["tokenizer_asset_id"] == ASSET_ID
        assert created.json()["tokenizer_asset_origin"] == "external"
        assert created.json()["tokenizer_asset_revision"] == "0731"
        assert created.json()["tokenizer_asset_fingerprint"].startswith("sha256:")
        assert created.json()["tokenizer_path"].endswith(ASSET_ID)

        repeated = await client.post("/v1/token_routers", json=payload)
        assert repeated.status_code == 201
        assert repeated.json()["revision"] == 1

        listed = await client.get("/v1/token_routers")
        assert listed.status_code == 200
        assert [item["router_uid"] for item in listed.json()] == ["router-a"]

        invalid = deepcopy(payload)
        invalid["router_uid"] = "router-b"
        invalid["backends"]["long"]["model_uid"] = "short-model"
        response = await client.post("/v1/token_routers", json=invalid)
        assert response.status_code == 422

        update = deepcopy(payload)
        update.pop("router_uid")
        update["revision"] = 1
        update["routing"]["default_output_tokens"] = 1024
        updated = await client.put("/v1/token_routers/router-a", json=update)
        assert updated.status_code == 200
        assert updated.json()["revision"] == 2

        conflict = await client.put("/v1/token_routers/router-a", json=update)
        assert conflict.status_code == 409

        validated = await client.post("/v1/token_routers/router-a/validate")
        assert validated.status_code == 200
        assert validated.json()["valid"] is True

        enabled = await client.post("/v1/token_routers/router-a/enable")
        assert enabled.status_code == 200
        assert enabled.json()["enabled"] is True
        assert (await client.delete("/v1/token_routers/router-a")).status_code == 409

        disabled = await client.post("/v1/token_routers/router-a/disable")
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert (await client.delete("/v1/token_routers/router-a")).status_code == 200
        assert (await client.get("/v1/token_routers/router-a")).status_code == 404


@pytest.mark.asyncio
async def test_tokenizer_asset_registry_operations_run_off_event_loop(
    tmp_path,
) -> None:
    event_loop_thread = threading.current_thread()
    supervisor = make_supervisor(tmp_path)
    registry = supervisor._tokenizer_asset_registry
    calls = []
    original_list_assets = registry.list_assets
    original_validate_path = registry.validate_path

    def list_assets():
        calls.append(("list_assets", threading.current_thread()))
        return original_list_assets()

    def validate_path(tokenizer_path: str, *, smoke_test: bool):
        calls.append(("validate_path", threading.current_thread()))
        return original_validate_path(tokenizer_path, smoke_test=smoke_test)

    registry.list_assets = list_assets
    registry.validate_path = validate_path

    await supervisor.validate_tokenizer_asset(ASSET_ID)
    created = await supervisor.create_token_router("router-a", router_payload())
    update_payload = router_payload()
    update_payload.pop("router_uid")
    update_payload["revision"] = created["revision"]
    await supervisor.update_token_router("router-a", update_payload)
    assert await supervisor.validate_token_router("router-a") is not None

    custom_path_payload = router_payload()
    custom_path_payload.pop("tokenizer_asset_id")
    custom_path_payload["tokenizer_path"] = str(tmp_path / "missing-tokenizer")
    custom_path_payload["virtual_model_uid"] = "virtual-model-path"
    supervisor._token_router_store.create("router-path", custom_path_payload)
    assert await supervisor.validate_token_router("router-path") is not None

    assert calls
    assert all(thread is not event_loop_thread for _, thread in calls)


@pytest.mark.asyncio
async def test_tokenizer_asset_registry_operations_are_serialized(tmp_path) -> None:
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    supervisor = make_supervisor(tmp_path)
    registry = supervisor._tokenizer_asset_registry
    original_list_assets = registry.list_assets

    def list_assets():
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        try:
            return original_list_assets()
        finally:
            with state_lock:
                active -= 1

    registry.list_assets = list_assets

    await asyncio.gather(
        supervisor.get_tokenizer_asset(ASSET_ID),
        supervisor.get_tokenizer_asset(ASSET_ID),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_tokenizer_asset_registry_keeps_event_loop_responsive(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()
    supervisor = make_supervisor(tmp_path)
    registry = supervisor._tokenizer_asset_registry
    original_list_assets = registry.list_assets

    def list_assets():
        started.set()
        release.wait(timeout=1)
        return original_list_assets()

    registry.list_assets = list_assets
    safety_timer = threading.Timer(0.5, release.set)
    safety_timer.start()
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    task = asyncio.create_task(supervisor.get_tokenizer_asset(ASSET_ID))
    try:
        await asyncio.sleep(0.02)
        elapsed = loop.time() - started_at
        assert started.is_set()
        assert elapsed < 0.25
    finally:
        release.set()
        safety_timer.cancel()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("process", [None, {"tokenizer_asset": None}])
async def test_validation_tolerates_missing_process_tokenizer_metadata(
    tmp_path, process
) -> None:
    supervisor = make_supervisor(tmp_path)
    created = await supervisor.create_token_router("router-a", router_payload())
    supervisor._token_router_registry.register(
        "router-a",
        "instance-a",
        {"endpoint": "http://router.internal", "acked_revision": created["revision"]},
    )
    supervisor._token_router_registry.heartbeat(
        "instance-a", {"status": "ready", "process": process}
    )

    result = await supervisor.validate_token_router("router-a")

    assert result is not None
    assert result["valid"] is False
    assert any("asset_id differs" in error for error in result["errors"])


@pytest.mark.asyncio
async def test_enable_rejects_invalid_router_and_preserves_disabled_draft(
    tmp_path,
) -> None:
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = router_payload()
        payload["backends"]["long"]["model_uid"] = "offline-model"
        created = await client.post("/v1/token_routers", json=payload)
        assert created.status_code == 201

        enabled = await client.post("/v1/token_routers/router-a/enable")
        assert enabled.status_code == 409
        assert "not running" in enabled.json()["detail"]

        current = await client.get("/v1/token_routers/router-a")
        assert current.status_code == 200
        assert current.json()["enabled"] is False
        assert current.json()["revision"] == 1
        assert current.json()["backends"]["long"]["model_uid"] == "offline-model"


@pytest.mark.asyncio
async def test_enabled_router_rejects_invalid_update_but_disabled_draft_can_save(
    tmp_path,
) -> None:
    supervisor = make_supervisor(tmp_path)
    app = create_app(supervisor)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/v1/token_routers", json=router_payload())
        assert created.status_code == 201
        enabled = await client.post("/v1/token_routers/router-a/enable")
        assert enabled.status_code == 200
        assert enabled.json()["revision"] == 2

        async def no_running_models(_self):
            return {}

        supervisor.list_models = MethodType(no_running_models, supervisor)
        update = router_payload()
        update.pop("router_uid")
        update["revision"] = 2
        update["backends"]["long"]["model_uid"] = "offline-model"

        rejected = await client.put("/v1/token_routers/router-a", json=update)
        assert rejected.status_code == 409
        assert "not running" in rejected.json()["detail"]

        unchanged = await client.get("/v1/token_routers/router-a")
        assert unchanged.json()["enabled"] is True
        assert unchanged.json()["revision"] == 2
        assert unchanged.json()["backends"]["long"]["model_uid"] == "long-model"

        disabled = await client.post("/v1/token_routers/router-a/disable")
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert disabled.json()["revision"] == 3

        update["revision"] = 3
        saved_draft = await client.put("/v1/token_routers/router-a", json=update)
        assert saved_draft.status_code == 200
        assert saved_draft.json()["revision"] == 4
        assert saved_draft.json()["backends"]["long"]["model_uid"] == "offline-model"

        re_enabled = await client.post("/v1/token_routers/router-a/enable")
        assert re_enabled.status_code == 409
        current = await client.get("/v1/token_routers/router-a")
        assert current.json()["enabled"] is False
        assert current.json()["revision"] == 4


@pytest.mark.asyncio
async def test_validation_checks_running_backend_type_and_context(tmp_path) -> None:
    supervisor = make_supervisor(tmp_path)

    async def list_models(_self):
        return {
            "short-model": {
                "model_type": "embedding",
                "model_engine": "vLLM",
                "model_ability": ["chat"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 4096,
            }
        }

    supervisor.list_models = MethodType(list_models, supervisor)
    app = create_app(supervisor)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/v1/token_routers", json=router_payload())
        ).status_code == 201
        response = await client.post("/v1/token_routers/router-a/validate")

    assert response.status_code == 200
    result = response.json()
    assert result["valid"] is False
    assert "short backend model must be an LLM" in result["errors"][0]
    assert any("short max_context_tokens" in error for error in result["errors"])
    assert any(
        "long backend model is not running" in error for error in result["errors"]
    )


@pytest.mark.asyncio
async def test_tokenizer_asset_endpoints_and_request_errors(tmp_path) -> None:
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        listed = await client.get("/v1/tokenizer_assets")
        assert listed.status_code == 200
        assert listed.json()["allow_custom_path"] is False
        items = {item["asset_id"]: item for item in listed.json()["items"]}
        assert items[ASSET_ID]["origin"] == "external"
        assert items["deepseek-v4-flash-0731"]["origin"] == "builtin"
        assert "path" not in items[ASSET_ID]

        detail = await client.get(f"/v1/tokenizer_assets/{ASSET_ID}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "available"

        validated = await client.post(f"/v1/tokenizer_assets/{ASSET_ID}/validate")
        assert validated.status_code == 200
        assert validated.json()["valid"] is True
        assert validated.json()["validated_at"]

        assert (await client.get("/v1/tokenizer_assets/missing")).status_code == 404

        no_source = router_payload()
        no_source.pop("tokenizer_asset_id")
        assert (
            await client.post("/v1/token_routers", json=no_source)
        ).status_code == 422

        custom = router_payload()
        custom.pop("tokenizer_asset_id")
        custom["tokenizer_path"] = "/unregistered/tokenizer"
        assert (await client.post("/v1/token_routers", json=custom)).status_code == 403


@pytest.mark.asyncio
async def test_asset_path_conflict_and_backend_compatibility(tmp_path) -> None:
    supervisor = make_supervisor(tmp_path)

    async def list_models(_self):
        return {
            "short-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools", "reasoning", "hybrid"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "context_length": 131072,
            },
            "long-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools", "reasoning", "hybrid"],
                "model_name": "Another-Model",
                "context_length": 1048576,
            },
        }

    supervisor.list_models = MethodType(list_models, supervisor)
    app = create_app(supervisor)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        conflict = router_payload()
        conflict["tokenizer_path"] = str(tmp_path / "different")
        assert (
            await client.post("/v1/token_routers", json=conflict)
        ).status_code == 409

        assert (
            await client.post("/v1/token_routers", json=router_payload())
        ).status_code == 201
        validation = await client.post("/v1/token_routers/router-a/validate")
        assert validation.status_code == 200
        assert validation.json()["valid"] is False
        assert "not compatible" in " ".join(validation.json()["errors"])


@pytest.mark.asyncio
async def test_runtime_internal_api_and_empty_204(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XINFERENCE_TOKEN_ROUTER_INTERNAL_TOKEN", "internal-secret")
    app = create_app(make_supervisor(tmp_path))
    headers = {"Authorization": "Bearer internal-secret"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.get("/v1/internal/token-router/configs/router-a")
        ).status_code == 401

        assert (
            await client.post("/v1/token_routers", json=router_payload())
        ).status_code == 201
        config = await client.get(
            "/v1/internal/token-router/configs/router-a",
            headers=headers,
            params={"after_revision": 0},
        )
        assert config.status_code == 200
        revision = config.json()["revision"]

        no_change = await client.get(
            "/v1/internal/token-router/configs/router-a",
            headers=headers,
            params={"after_revision": revision},
        )
        assert no_change.status_code == 204
        assert no_change.content == b""

        registered = await client.post(
            "/v1/internal/token-router/instances/register",
            headers=headers,
            json={
                "router_uid": "router-a",
                "instance_id": "instance-a",
                "endpoint": "http://router:10080",
                "acked_revision": 0,
            },
        )
        assert registered.status_code == 200

        heartbeat = await client.post(
            "/v1/internal/token-router/instances/instance-a/heartbeat",
            headers=headers,
            json={"status": "ready", "metrics": {"requests": 1}},
        )
        assert heartbeat.status_code == 200
        assert heartbeat.json()["metrics"] == {"requests": 1}

        ack = await client.post(
            "/v1/internal/token-router/instances/instance-a/config-ack",
            headers=headers,
            json={"router_uid": "router-a", "revision": revision},
        )
        assert ack.status_code == 200
        assert ack.json()["acked_revision"] == revision

        instances = await client.get("/v1/token_routers/router-a/instances")
        assert instances.status_code == 200
        assert instances.json()[0]["instance_id"] == "instance-a"

        future_ack = await client.post(
            "/v1/internal/token-router/instances/instance-a/config-ack",
            headers=headers,
            json={"router_uid": "router-a", "revision": revision + 1},
        )
        assert future_ack.status_code == 409

        stale_ack = await client.post(
            "/v1/internal/token-router/instances/instance-a/config-ack",
            headers=headers,
            json={"router_uid": "router-a", "revision": revision - 1},
        )
        assert stale_ack.status_code == 422

        unregistered = await client.post(
            "/v1/internal/token-router/instances/instance-a/unregister",
            headers=headers,
        )
        assert unregistered.status_code == 200
        assert (
            await client.post(
                "/v1/internal/token-router/instances/instance-a/unregister",
                headers=headers,
            )
        ).status_code == 404


@pytest.mark.asyncio
async def test_management_scope_enforcement(tmp_path) -> None:
    token_scopes = {
        "reader": {"routers:list", "routers:read"},
        "writer": {"routers:write"},
        "admin-token": {"admin"},
    }

    async def auth_service(
        security_scopes: SecurityScopes,
        authorization: str = Header(default=""),
    ) -> dict:
        token = authorization.removeprefix("Bearer ")
        scopes = token_scopes.get(token)
        if scopes is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        if "admin" not in scopes and not set(security_scopes.scopes).issubset(scopes):
            raise HTTPException(status_code=403, detail="Forbidden")
        return {"username": token}

    app = create_app(
        make_supervisor(tmp_path), authenticated=True, auth_service=auth_service
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/v1/token_routers")).status_code == 401
        assert (await client.get("/v1/token_routers/defaults")).status_code == 401
        assert (
            await client.get(
                "/v1/token_routers", headers={"Authorization": "Bearer reader"}
            )
        ).status_code == 200
        assert (
            await client.get(
                "/v1/token_routers/defaults",
                headers={"Authorization": "Bearer reader"},
            )
        ).status_code == 200
        http_sd_path = "/v1/monitor/prometheus/http-sd/token-router-runtimes"
        assert (await client.get(http_sd_path)).status_code == 401
        http_sd = await client.get(
            http_sd_path, headers={"Authorization": "Bearer reader"}
        )
        assert http_sd.status_code == 200
        assert http_sd.json() == []
        assert (
            await client.get(http_sd_path, headers={"Authorization": "Bearer writer"})
        ).status_code == 403
        assert (
            await client.post(
                "/v1/token_routers",
                headers={"Authorization": "Bearer reader"},
                json=router_payload(),
            )
        ).status_code == 403
        assert (
            await client.post(
                "/v1/token_routers",
                headers={"Authorization": "Bearer admin-token"},
                json=router_payload(),
            )
        ).status_code == 201


@pytest.mark.asyncio
async def test_v2_dynamic_router_crud_and_validation(tmp_path) -> None:
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/v1/token_routers", json=typed_router_payload())
        assert created.status_code == 201
        body = created.json()
        assert body["config_version"] == 2
        assert body["route_profile"] == "llm_chat"
        assert body["strategy"] == "typed_rules"
        assert [backend["id"] for backend in body["backends"]] == [
            "fast",
            "tools",
            "reasoning",
            "long",
        ]
        assert [rule["id"] for rule in body["routing"]["rules"]] == [
            "tools-route",
            "thinking-route",
            "short-route",
            "long-route",
        ]

        validation = await client.post("/v1/token_routers/router-a/validate")
        assert validation.status_code == 200
        assert validation.json()["valid"] is True
        assert validation.json()["errors"] == []

        fetched = await client.get("/v1/token_routers/router-a")
        assert fetched.status_code == 200
        assert fetched.json()["routing"]["default_action"] == {
            "type": "reject",
            "reason": "no_compatible_backend",
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(config_version=3),
        lambda payload: payload["backends"][1].update(id="fast"),
        lambda payload: payload["routing"]["rules"][1].update(id="tools-route"),
        lambda payload: payload["routing"]["rules"][1].update(priority=400),
        lambda payload: payload["routing"]["rules"][0].update(match={}),
        lambda payload: payload["routing"]["rules"][0]["action"].update(
            backend_id="missing"
        ),
    ],
)
async def test_v2_schema_rejects_invalid_dynamic_config(tmp_path, mutate) -> None:
    payload = typed_router_payload()
    mutate(payload)
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/v1/token_routers", json=payload)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_backend_candidates_apply_profile_engine_and_asset_filters(
    tmp_path,
) -> None:
    supervisor = make_supervisor(tmp_path)

    async def list_models(_self):
        return {
            "eligible-vllm": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools"],
                "model_name": "DeepSeek-V4-Flash-0731",
                "model_format": "pytorch",
                "context_length": 131072,
            },
            "embedding": {
                "model_type": "embedding",
                "model_engine": "vLLM",
                "model_ability": ["chat"],
                "model_name": "DeepSeek-V4-Flash-0731",
            },
            "completion-only": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["generate"],
                "model_name": "DeepSeek-V4-Flash-0731",
            },
            "nested-router": {
                "model_type": "LLM",
                "model_engine": "token_router",
                "model_ability": ["chat"],
                "model_name": "DeepSeek-V4-Flash-0731",
            },
            "unsupported-engine": {
                "model_type": "LLM",
                "model_engine": "SGLang",
                "model_ability": ["chat"],
                "model_name": "DeepSeek-V4-Flash-0731",
            },
            "asset-mismatch": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat"],
                "model_name": "Another-Model",
            },
        }

    supervisor.list_models = MethodType(list_models, supervisor)
    app = create_app(supervisor)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/backend-candidates")
        asset_response = await client.get(
            "/v1/token_routers/backend-candidates",
            params={"tokenizer_asset_id": ASSET_ID},
        )
        missing = await client.get(
            "/v1/token_routers/backend-candidates",
            params={"tokenizer_asset_id": "missing"},
        )

    assert response.status_code == 200
    candidates = {item["model_uid"]: item for item in response.json()["items"]}
    assert candidates["eligible-vllm"]["eligible"] is True
    assert candidates["eligible-vllm"]["model_name"] == "DeepSeek-V4-Flash-0731"
    assert candidates["eligible-vllm"]["compatibility_status"] == "Verified"
    assert candidates["embedding"]["eligible"] is False
    assert "model_type must be LLM" in candidates["embedding"]["ineligible_reasons"]
    assert candidates["completion-only"]["eligible"] is False
    assert candidates["nested-router"]["compatibility_status"] == "Unsupported"
    assert candidates["unsupported-engine"]["eligible"] is False
    # Without an asset selection, candidate discovery retains the legacy behavior.
    assert candidates["asset-mismatch"]["eligible"] is True
    assert missing.status_code == 200
    assert missing.json()["items"]
    assert "not registered" in missing.json()["errors"][0]

    assert asset_response.status_code == 200
    asset_candidates = {
        item["model_uid"]: item for item in asset_response.json()["items"]
    }
    assert asset_candidates["eligible-vllm"]["eligible"] is True
    assert asset_candidates["asset-mismatch"]["eligible"] is False
    assert any(
        f"not compatible with Tokenizer asset {ASSET_ID}" in reason
        for reason in asset_candidates["asset-mismatch"]["ineligible_reasons"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0", "::1", "::", "localhost"])
async def test_token_router_defaults_rejects_non_routable_bind_address(
    tmp_path, host
) -> None:
    app = create_app(make_supervisor(tmp_path), host=host)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/defaults")

    assert response.status_code == 200
    assert response.json() == {
        "backend": {
            "mode": "current_supervisor",
            "display_name": "Current Supervisor",
            "backend_url": None,
            "source": "unavailable",
            "available": False,
            "error": (
                "REST API bind address is not reachable by a separate Token Router"
            ),
        }
    }


@pytest.mark.asyncio
async def test_token_router_defaults_returns_routable_rest_endpoint(tmp_path) -> None:
    app = create_app(make_supervisor(tmp_path), host="xinference-supervisor")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/token_routers/defaults")

    assert response.status_code == 200
    assert response.json()["backend"] == {
        "mode": "current_supervisor",
        "display_name": "Current Supervisor",
        "backend_url": "http://xinference-supervisor:9997",
        "source": "rest_endpoint",
        "available": True,
    }


@pytest.mark.asyncio
async def test_v2_validation_checks_tools_and_thinking_capabilities(tmp_path) -> None:
    supervisor = make_supervisor(tmp_path)
    original_list_models = supervisor.list_models

    async def list_models(_self):
        models = await original_list_models()
        models["tools-model"]["model_ability"] = ["chat"]
        models["reasoning-model"]["model_ability"] = ["chat"]
        return models

    supervisor.list_models = MethodType(list_models, supervisor)
    app = create_app(supervisor)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (
            await client.post("/v1/token_routers", json=typed_router_payload())
        ).status_code == 201
        validation = await client.post("/v1/token_routers/router-a/validate")

    assert validation.status_code == 200
    result = validation.json()
    assert result["valid"] is False
    assert any("requires tools" in error for error in result["errors"])
    assert any("requires thinking" in error for error in result["errors"])


def test_parse_env_bool_is_strict(monkeypatch):
    monkeypatch.setenv("TEST_BOOL", "YeS")
    assert parse_env_bool("TEST_BOOL", False) is True
    monkeypatch.setenv("TEST_BOOL", "off")
    assert parse_env_bool("TEST_BOOL", True) is False
    monkeypatch.setenv("TEST_BOOL", "maybe")
    with pytest.raises(ValueError, match="TEST_BOOL"):
        parse_env_bool("TEST_BOOL", True)


@pytest.mark.asyncio
async def test_disabled_feature_rejects_public_and_internal_apis(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(token_routers, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    app = create_app(make_supervisor(tmp_path))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        public_response = await client.get("/v1/token_routers")
        internal_response = await client.post(
            "/v1/internal/token-router/instances/register", json={}
        )

    for response in (public_response, internal_response):
        assert response.status_code == 503
        assert response.json()["detail"] == {
            "code": "TOKEN_ROUTER_DISABLED",
            "message": "Token Router feature is disabled",
        }
@pytest.mark.asyncio
async def test_public_router_responses_normalize_sparse_configs_without_persisting(
    tmp_path,
) -> None:
    supervisor = make_supervisor(tmp_path)
    legacy_config = supervisor._token_router_store.create(
        "legacy-sparse",
        {"virtual_model_uid": "legacy-virtual", "backends": {}},
    )
    typed_config = supervisor._token_router_store.create(
        "typed-sparse",
        {
            "virtual_model_uid": "typed-virtual",
            "config_version": 2,
            "backends": [],
            "routing": {},
        },
    )

    legacy = await supervisor.get_token_router("legacy-sparse")
    typed = await supervisor.get_token_router("typed-sparse")
    assert legacy is not None and typed is not None
    assert legacy["model_aliases"] == []
    assert legacy["tokenization"]["executor"] == "process"
    assert legacy["backends"]["short"]["admission"]["max_queue"] == 0
    assert legacy["routing"]["overflow_policy"] == "reject"
    assert legacy["deployment"]["management_mode"] == "external"
    assert typed["config_version"] == 2
    assert typed["backends"] == []
    assert typed["routing"]["rules"] == []
    assert typed["routing"]["default_action"] == {
        "type": "reject",
        "reason": "configuration_error",
    }
    assert typed["deployment"]["management_mode"] == "external"
    assert supervisor._token_router_store.get("legacy-sparse") == legacy_config
    assert supervisor._token_router_store.get("typed-sparse") == typed_config
