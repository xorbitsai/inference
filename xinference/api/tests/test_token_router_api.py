import asyncio
import threading
import time
from copy import deepcopy
from types import MethodType

import httpx
import pytest
from fastapi import APIRouter, FastAPI, Header, HTTPException
from fastapi.security import SecurityScopes

from xinference.api.routers.token_routers import register_routes
from xinference.core.router_config_store import RouterConfigStore
from xinference.core.router_registry import RouterRuntimeRegistry
from xinference.core.supervisor import SupervisorActor


def router_payload() -> dict:
    return {
        "router_uid": "router-a",
        "virtual_model_uid": "virtual-model",
        "model_type": "LLM",
        "strategy": "token_budget",
        "tokenizer_path": "/models/tokenizer",
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


class FakeTokenizerAssetRegistry:
    allow_custom_path = True

    def reload(self) -> None:
        pass

    def match_path(self, tokenizer_path: str):
        return None

    def validate_path(self, tokenizer_path: str, *, smoke_test: bool) -> dict:
        return {"valid": True, "errors": []}


def make_supervisor(tmp_path):
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._token_router_store = RouterConfigStore(str(tmp_path / "routers.db"))
    supervisor._token_router_registry = RouterRuntimeRegistry()
    supervisor._tokenizer_asset_registry = FakeTokenizerAssetRegistry()

    async def list_models(_self):
        return {
            "short-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat"],
                "context_length": 131072,
            },
            "tools-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "tools"],
                "context_length": 131072,
            },
            "reasoning-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat", "reasoning"],
                "context_length": 262144,
            },
            "long-model": {
                "model_type": "LLM",
                "model_engine": "vLLM",
                "model_ability": ["chat"],
                "context_length": 1048576,
            },
        }

    supervisor.list_models = MethodType(list_models, supervisor)
    return supervisor


class FakeAPI:
    def __init__(self, supervisor, *, authenticated: bool, auth_service) -> None:
        self._router = APIRouter()
        self._supervisor = supervisor
        self._authenticated = authenticated
        self._auth_service = auth_service

    def is_authenticated(self) -> bool:
        return self._authenticated

    async def _get_supervisor_ref(self):
        return self._supervisor


async def unused_auth():
    return {"username": "anonymous"}


def create_app(supervisor, *, authenticated: bool = False, auth_service=unused_auth):
    api = FakeAPI(
        supervisor,
        authenticated=authenticated,
        auth_service=auth_service,
    )
    register_routes(api)  # type: ignore[arg-type]
    app = FastAPI()
    app.state.api = api
    app.include_router(api._router)
    return app


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
async def test_tokenizer_asset_registry_operations_run_off_event_loop(tmp_path) -> None:
    event_loop_thread = threading.current_thread()
    calls = []

    class TokenizerAssetRegistry:
        allow_custom_path = False

        @staticmethod
        def _record(name: str) -> None:
            calls.append((name, threading.current_thread()))

        def reload(self) -> None:
            self._record("reload")

        def resolve(self, asset_id: str, tokenizer_path=None) -> dict:
            self._record("resolve")
            assert asset_id == "test-asset"
            assert tokenizer_path is None
            return {
                "tokenizer_asset_id": asset_id,
                "tokenizer_path": "/models/tokenizer",
                "tokenizer_asset_revision": "1",
                "tokenizer_asset_fingerprint": "sha256:test",
            }

        def validate_asset(self, asset_id: str) -> dict:
            self._record("validate_asset")
            assert asset_id == "test-asset"
            return {
                "valid": True,
                "errors": [],
                "revision": "1",
                "fingerprint": "sha256:test",
            }

        def validate_path(self, tokenizer_path: str, *, smoke_test: bool) -> dict:
            self._record("validate_path")
            assert tokenizer_path == "/models/tokenizer"
            assert smoke_test is True
            return {"valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()
    payload = router_payload()
    payload.pop("tokenizer_path")
    payload["tokenizer_asset_id"] = "test-asset"

    await supervisor.validate_tokenizer_asset("test-asset")
    created = await supervisor.create_token_router("router-a", payload)
    update_payload = deepcopy(payload)
    update_payload["revision"] = created["revision"]
    await supervisor.update_token_router("router-a", update_payload)
    assert await supervisor.validate_token_router("router-a") is not None

    custom_path_payload = router_payload()
    custom_path_payload["virtual_model_uid"] = "virtual-model-path"
    supervisor._token_router_store.create("router-path", custom_path_payload)
    assert await supervisor.validate_token_router("router-path") is not None

    assert [name for name, _ in calls].count("resolve") == 2
    assert [name for name, _ in calls].count("validate_asset") == 2
    assert [name for name, _ in calls].count("validate_path") == 1
    assert all(thread is not event_loop_thread for _, thread in calls)


@pytest.mark.asyncio
async def test_tokenizer_asset_registry_operations_are_serialized(tmp_path) -> None:
    state_lock = threading.Lock()
    active = 0
    max_active = 0

    class TokenizerAssetRegistry:
        def reload(self) -> None:
            pass

        def validate_asset(self, asset_id: str) -> dict:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with state_lock:
                active -= 1
            return {"asset_id": asset_id, "valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()

    await asyncio.gather(
        supervisor.validate_tokenizer_asset("asset-a"),
        supervisor.validate_tokenizer_asset("asset-b"),
    )

    assert max_active == 1


@pytest.mark.asyncio
async def test_tokenizer_asset_validation_keeps_event_loop_responsive(
    tmp_path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class TokenizerAssetRegistry:
        def reload(self) -> None:
            pass

        def validate_asset(self, asset_id: str) -> dict:
            assert asset_id == "test-asset"
            started.set()
            release.wait(timeout=1)
            return {"valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()
    safety_timer = threading.Timer(0.5, release.set)
    safety_timer.start()
    loop = asyncio.get_running_loop()
    started_at = loop.time()
    task = asyncio.create_task(supervisor.validate_tokenizer_asset("test-asset"))
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
    class TokenizerAssetRegistry:
        def reload(self) -> None:
            pass

        def validate_asset(self, asset_id: str) -> dict:
            assert asset_id == "test-asset"
            return {"valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()
    payload = router_payload()
    payload["tokenizer_asset_id"] = "test-asset"
    supervisor._token_router_store.create("router-a", payload)
    supervisor._token_router_registry.register(
        "router-a",
        "instance-a",
        {"endpoint": "http://router.internal", "acked_revision": 1},
    )
    supervisor._token_router_registry.heartbeat(
        "instance-a", {"status": "ready", "process": process}
    )

    result = await supervisor.validate_token_router("router-a")

    assert result is not None
    assert result["valid"] is False
    assert any(
        "loaded Tokenizer asset <unknown>, expected test-asset" in error
        for error in result["errors"]
    )


@pytest.mark.asyncio
async def test_validation_checks_running_backend_type_and_context(tmp_path) -> None:
    supervisor = make_supervisor(tmp_path)

    async def list_models(_self):
        return {
            "short-model": {
                "model_type": "embedding",
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
    assert any(
        "short backend model must be an LLM" in error for error in result["errors"]
    )
    assert any("short max_context_tokens" in error for error in result["errors"])
    assert any(
        "long backend model is not running" in error for error in result["errors"]
    )


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
        assert (
            await client.get(
                "/v1/token_routers", headers={"Authorization": "Bearer reader"}
            )
        ).status_code == 200
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
async def test_backend_candidates_apply_profile_and_engine_filters(
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

    assert response.status_code == 200
    candidates = {item["model_uid"]: item for item in response.json()["items"]}
    assert candidates["eligible-vllm"]["eligible"] is True
    assert candidates["eligible-vllm"]["compatibility_status"] == "Verified"
    assert candidates["embedding"]["eligible"] is False
    assert "model_type must be LLM" in candidates["embedding"]["ineligible_reasons"]
    assert candidates["completion-only"]["eligible"] is False
    assert candidates["nested-router"]["compatibility_status"] == "Unsupported"
    assert candidates["unsupported-engine"]["eligible"] is False
    assert candidates["asset-mismatch"]["eligible"] is True


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


@pytest.mark.asyncio
async def test_validation_detects_loaded_fingerprint_mismatch(tmp_path) -> None:
    class TokenizerAssetRegistry:
        def reload(self) -> None:
            pass

        def resolve(self, asset_id: str, tokenizer_path=None) -> dict:
            return {
                "tokenizer_asset_id": asset_id,
                "tokenizer_path": "/models/tokenizer",
                "tokenizer_asset_revision": "0731",
                "tokenizer_asset_fingerprint": "sha256:expected",
            }

        def validate_asset(self, asset_id: str) -> dict:
            assert asset_id == "test-asset"
            return {
                "valid": True,
                "errors": [],
                "revision": "0731",
                "fingerprint": "sha256:expected",
                "capabilities": {"chat": True, "tools": True, "thinking": True},
            }

        def validate_path(self, tokenizer_path: str, *, smoke_test: bool) -> dict:
            return {"valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()
    payload = router_payload()
    payload.pop("tokenizer_path")
    payload["tokenizer_asset_id"] = "test-asset"
    payload["tokenizer_asset_revision"] = "0731"
    payload["tokenizer_asset_fingerprint"] = "sha256:expected"
    supervisor._token_router_store.create("router-a", payload)
    supervisor._token_router_registry.register(
        "router-a",
        "instance-a",
        {"endpoint": "http://router.internal", "acked_revision": 1},
    )
    supervisor._token_router_registry.heartbeat(
        "instance-a",
        {
            "status": "ready",
            "process": {
                "tokenizer_asset": {
                    "asset_id": "test-asset",
                    "revision": "0731",
                    "fingerprint": "sha256:swapped",
                }
            },
        },
    )

    result = await supervisor.validate_token_router("router-a")

    assert result is not None
    assert result["valid"] is False
    assert any(
        "loaded Tokenizer asset fingerprint differs from the Router configuration"
        in error
        for error in result["errors"]
    )


@pytest.mark.asyncio
async def test_v2_validation_rejects_rules_asset_cannot_support(tmp_path) -> None:
    class TokenizerAssetRegistry:
        def reload(self) -> None:
            pass

        def validate_asset(self, asset_id: str) -> dict:
            return {
                "valid": True,
                "errors": [],
                "revision": "0731",
                "fingerprint": "sha256:expected",
                "capabilities": {"chat": True, "tools": False, "thinking": False},
            }

        def validate_path(self, tokenizer_path: str, *, smoke_test: bool) -> dict:
            return {"valid": True, "errors": []}

    supervisor = make_supervisor(tmp_path)
    supervisor._tokenizer_asset_registry = TokenizerAssetRegistry()
    payload = typed_router_payload()
    payload.pop("tokenizer_path")
    payload["tokenizer_asset_id"] = "test-asset"
    payload["tokenizer_asset_revision"] = "0731"
    payload["tokenizer_asset_fingerprint"] = "sha256:expected"
    supervisor._token_router_store.create("router-a", payload)

    result = await supervisor.validate_token_router("router-a")

    assert result is not None
    assert result["valid"] is False
    assert any(
        "requires tools but Tokenizer asset does not support tools" in error
        for error in result["errors"]
    )
    assert any(
        "requires thinking but Tokenizer asset does not support thinking" in error
        for error in result["errors"]
    )
