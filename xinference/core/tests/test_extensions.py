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

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI, Response

from xinference import extensions


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv("XINFERENCE_EXTENSIONS", raising=False)
    monkeypatch.setattr(extensions, "_loaded", None)


def install(monkeypatch, *, devices=None, name="test", version=1):
    module = ModuleType("test_runtime_extension")
    module.create = lambda: SimpleNamespace(
        name=name,
        api_version=version,
        worker_metadata=lambda: {"node": "test"},
        filter_worker_resources=lambda worker, workers: (
            worker["gpu_devices"] if devices is None else devices
        ),
        configure_api=lambda context: context.append(name),
        control_operation=lambda op, payload: (op, payload),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "test_runtime_extension:create")


def test_default_has_no_distribution_dependency():
    assert extensions.get_extensions() == ()
    worker = {"gpu_devices": [0, 1], "extensions": {}}
    assert extensions.filter_worker_resources(worker, [worker]) == [0, 1]


def test_process_local_load_and_named_operation(monkeypatch):
    install(monkeypatch)
    first = extensions.get_extensions()
    assert first is extensions.get_extensions()
    context = []
    extensions.configure_api(context)
    assert context == ["test"]
    assert extensions.worker_extension_metadata() == {"test": {"node": "test"}}
    assert extensions.call_extension("test", "status", {"a": 1}) == ("status", {"a": 1})
    with pytest.raises(ValueError):
        extensions.call_extension("missing", "status", {})


def test_filter_cannot_expand_visible_devices(monkeypatch):
    install(monkeypatch, devices=[9])
    with pytest.raises(ValueError, match="must not expand"):
        extensions.filter_worker_resources(
            {"gpu_devices": [0], "extensions": {"test": {}}}, []
        )


def test_filter_cannot_ignore_missing_worker_extension(monkeypatch):
    install(monkeypatch)
    with pytest.raises(RuntimeError, match="missing required"):
        extensions.filter_worker_resources({"gpu_devices": [0], "extensions": {}}, [])


def test_filter_narrows_devices_without_reordering(monkeypatch):
    install(monkeypatch, devices=[2, 0])
    assert extensions.filter_worker_resources(
        {"gpu_devices": [0, 1, 2], "extensions": {"test": {}}}, []
    ) == [0, 2]


@pytest.mark.parametrize(
    "configured", ["missing:create", "missing", "module:", "a:b:c", " : factory"]
)
def test_configured_extension_failure_is_not_silent(monkeypatch, configured):
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", configured)
    with pytest.raises((ValueError, RuntimeError), match="[Ee]xtension"):
        extensions.get_extensions()


def test_api_version_is_checked(monkeypatch):
    install(monkeypatch, version=99)
    with pytest.raises(ValueError, match="Incompatible"):
        extensions.get_extensions()


def test_duplicate_extension_name_is_rejected(monkeypatch):
    install(monkeypatch)
    monkeypatch.setenv(
        "XINFERENCE_EXTENSIONS",
        "test_runtime_extension:create,test_runtime_extension:create",
    )
    with pytest.raises(ValueError, match="unique"):
        extensions.get_extensions()


def test_fresh_process_loads_configured_extension(tmp_path):
    (tmp_path / "spawn_extension.py").write_text(
        "from types import SimpleNamespace\n"
        "def create(): return SimpleNamespace(name='spawn', api_version=1)\n"
    )
    environment = dict(os.environ, XINFERENCE_EXTENSIONS="spawn_extension:create")
    environment["PYTHONPATH"] = (
        str(tmp_path) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from xinference.extensions import get_extensions; assert get_extensions()[0].name == 'spawn'",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_worker_allocation_applies_policy_to_auto_explicit_and_cpu(monkeypatch):
    from xinference.core.worker import WorkerActor

    install(monkeypatch)
    worker = WorkerActor("supervisor", None, SimpleNamespace(), [0, 1, 2])

    async def policy():
        return [1]

    monkeypatch.setattr(worker, "_get_allowed_gpu_devices", policy)
    _, devices = await worker._allocate_subpool_devices("auto", n_gpu=1)
    assert devices == ["1"]
    with pytest.raises(ValueError, match="excluded"):
        await worker._allocate_subpool_devices("explicit", gpu_idx=[2])

    async def denied():
        raise ValueError("admission denied")

    monkeypatch.setattr(worker, "_get_allowed_gpu_devices", denied)
    with pytest.raises(ValueError, match="admission denied"):
        await worker._allocate_subpool_devices("cpu", n_gpu=None)
    assert not worker._user_specified_gpu_to_model_uids.get(2)


@pytest.mark.asyncio
async def test_worker_revalidates_after_preparation(monkeypatch):
    from xinference.core import worker as worker_module

    install(monkeypatch)
    worker = worker_module.WorkerActor("supervisor", None, SimpleNamespace(), [0, 1])
    snapshot = Mock(return_value=0)
    cleanup = AsyncMock(return_value=[])
    monkeypatch.setattr(worker_module, "_snapshot_gpu_free_ratio", snapshot)
    monkeypatch.setattr(worker_module, "_kill_gpu_orphans_by_ppid", cleanup)

    async def reduced():
        return [0]

    monkeypatch.setattr(worker, "_get_allowed_gpu_devices", reduced)
    with pytest.raises(ValueError, match="Reserved GPUs"):
        await worker._spawn_subpool("prepared", {}, ["1"])
    snapshot.assert_not_called()
    cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_up_skips_policy_failures_instead_of_falling_back(monkeypatch):
    from xinference.core.supervisor import SupervisorActor

    install(monkeypatch)

    async def count():
        return 0

    async def denied():
        raise ValueError("admission denied")

    async def allowed():
        return {"total": [1], "models": {}, "user_specified": {}}

    supervisor = SupervisorActor()
    blocked = SimpleNamespace(
        address="blocked", get_model_count=count, get_gpu_allocation_status=denied
    )
    permitted = SimpleNamespace(
        address="permitted", get_model_count=count, get_gpu_allocation_status=allowed
    )
    supervisor._worker_address_to_worker = {"blocked": blocked, "permitted": permitted}
    assert (await supervisor._select_worker_for_scale_up(None))[0] is permitted
    assert (await supervisor._select_worker_for_scale_up(1))[0] is permitted
    supervisor._worker_address_to_worker.pop("permitted")
    with pytest.raises(RuntimeError, match="No available worker"):
        await supervisor._select_worker_for_scale_up(None)


@pytest.mark.asyncio
async def test_registration_requires_matching_worker_extension(monkeypatch):
    import xoscar as xo

    from xinference.core.supervisor import SupervisorActor

    install(monkeypatch)

    async def resources():
        return {"extensions": {}}

    async def reference(**kwargs):
        return SimpleNamespace(get_extension_resources=resources)

    monkeypatch.setattr(xo, "actor_ref", reference)
    supervisor = SupervisorActor()
    with pytest.raises(RuntimeError, match="must match"):
        await supervisor.add_worker("missing-extension")
    assert not supervisor._worker_address_to_worker


def test_resource_policy_only_includes_live_registered_workers(monkeypatch):
    from xinference.core.supervisor import SupervisorActor

    install(monkeypatch)
    supervisor = SupervisorActor()
    info = {"gpu_devices": [0], "extensions": {"test": {}}}
    supervisor._extension_worker_resources = {"live": info, "removed": info}
    supervisor._worker_address_to_worker = {"live": object()}
    assert supervisor.get_worker_resource_policy("live") == [0]
    with pytest.raises(RuntimeError, match="has not registered"):
        supervisor.get_worker_resource_policy("removed")


def test_extension_entries_strip_parts_and_name_import_failures(monkeypatch):
    install(monkeypatch)
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", " test_runtime_extension : create ")
    assert extensions.get_extensions()[0].name == "test"
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "test_runtime_extension:missing")
    with pytest.raises(RuntimeError, match="test_runtime_extension:missing"):
        extensions.get_extensions()


def test_async_factory_is_rejected_explicitly(monkeypatch):
    install(monkeypatch)

    async def factory():
        return None

    monkeypatch.setattr(sys.modules["test_runtime_extension"], "create", factory)
    with pytest.raises(RuntimeError, match="factory.*must be synchronous"):
        extensions.get_extensions()


@pytest.mark.asyncio
async def test_spawn_revalidates_policy_after_gpu_cleanup(monkeypatch):
    from xinference.core import worker as worker_module

    install(monkeypatch)
    worker = worker_module.WorkerActor("supervisor", None, SimpleNamespace(), [0, 1])
    permitted = [1]

    async def policy():
        return list(permitted)

    async def cleanup(*args, **kwargs):
        permitted.clear()
        return []

    spawn = AsyncMock()
    monkeypatch.setattr(worker, "_get_allowed_gpu_devices", policy)
    monkeypatch.setattr(worker, "_append_sub_pool_protected", spawn)
    monkeypatch.setattr(worker_module, "_snapshot_gpu_free_ratio", lambda devices: 0)
    monkeypatch.setattr(worker_module, "_kill_gpu_orphans_by_ppid", cleanup)
    with pytest.raises(ValueError, match="Reserved GPUs"):
        await worker._spawn_subpool("prepared", {}, ["1"])
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    "hook",
    [
        "configure_api",
        "worker_metadata",
        "filter_worker_resources",
        "control_operation",
    ],
)
def test_async_hooks_are_rejected_explicitly(monkeypatch, hook):
    install(monkeypatch)
    extension = extensions.get_extensions()[0]

    async def async_hook(*args):
        return []

    setattr(extension, hook, async_hook)
    worker = {"gpu_devices": [0], "extensions": {"test": {}}}
    with pytest.raises(TypeError, match=f"test.{hook} must be synchronous"):
        if hook == "configure_api":
            extensions.configure_api([])
        elif hook == "worker_metadata":
            extensions.worker_extension_metadata()
        elif hook == "filter_worker_resources":
            extensions.filter_worker_resources(worker, [worker])
        else:
            extensions.call_extension("test", "status", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_kind", ["supervisor", "worker"])
async def test_invalid_extension_fails_actor_startup(monkeypatch, actor_kind):
    from xinference.core.supervisor import SupervisorActor
    from xinference.core.worker import WorkerActor

    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "missing_runtime_extension:create")
    actor = (
        SupervisorActor()
        if actor_kind == "supervisor"
        else WorkerActor("supervisor", None, SimpleNamespace(), [])
    )
    with pytest.raises(RuntimeError, match="missing_runtime_extension:create"):
        await actor.__post_create__()


@pytest.fixture
def extension_api(monkeypatch):
    from xinference.api import restful_api

    api = restful_api.RESTfulAPI.__new__(restful_api.RESTfulAPI)
    api._app, api._router = FastAPI(), APIRouter()
    api._advanced_auth_service = None
    api._auth_service = object()
    api._monitor_config_store = api._system_settings_store = None
    api._host, api._port = "127.0.0.1", 0
    monkeypatch.setattr(restful_api, "Server", Mock())
    monkeypatch.setattr(restful_api, "mount_frontend", lambda *args: True)
    monkeypatch.setattr(restful_api, "is_metrics_disabled", lambda: True)
    monkeypatch.setattr(restful_api, "XINFERENCE_ENABLE_OTEL", False)
    monkeypatch.setattr("xinference.api.routers.register_all_routes", lambda api: None)
    return api


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_enabled", [False, True])
async def test_api_serve_composes_extension_context_and_route(
    monkeypatch, extension_api, auth_enabled
):
    install(monkeypatch)
    contexts = []

    def configure(context):
        contexts.append(context)

        @context.router.get("/extension")
        async def route() -> Response:
            return Response("extension works")

    extensions.get_extensions()[0].configure_api = configure
    monkeypatch.setattr(
        type(extension_api), "is_authenticated", lambda self: auth_enabled
    )
    extension_api.serve()
    context = contexts[0]
    assert isinstance(context, extensions.APIExtensionContext)
    assert context.auth_enabled is auth_enabled
    assert context.auth_service is extension_api._auth_service
    assert context.app is extension_api._app
    assert context.router is extension_api._router
    assert context.get_supervisor_ref == extension_api._get_supervisor_ref
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=context.app), base_url="http://test"
    ) as client:
        response = await client.get("/extension")
    assert response.status_code == 200
    assert response.text == "extension works"


def test_extension_route_requires_response_annotation(monkeypatch, extension_api):
    install(monkeypatch)
    monkeypatch.setattr(type(extension_api), "is_authenticated", lambda self: False)

    def configure(context):
        @context.router.get("/invalid-extension")
        async def route() -> dict:
            return {}

    extensions.get_extensions()[0].configure_api = configure
    with pytest.raises(Exception, match="is not Response"):
        extension_api.serve()


@pytest.mark.asyncio
async def test_unconfigured_supervisor_rejects_worker_extensions_and_stale_registration(
    monkeypatch,
):
    import xoscar as xo

    from xinference.core.supervisor import SupervisorActor

    resources = {"address": "worker", "gpu_devices": [0], "extensions": {"test": {}}}
    worker = SimpleNamespace(get_extension_resources=AsyncMock(return_value=resources))
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=worker))
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker["worker"] = object()
    supervisor._extension_worker_resources["worker"] = resources
    supervisor._worker_metadata["worker"] = {"software_version": "old"}
    with pytest.raises(RuntimeError, match="must match"):
        await supervisor.add_worker("worker")
    assert "worker" not in supervisor._worker_address_to_worker
    assert "worker" not in supervisor._extension_worker_resources
    assert "worker" not in supervisor._worker_metadata


@pytest.mark.asyncio
async def test_registration_refreshes_extensions_and_remove_cleans_inventory(
    monkeypatch,
):
    import xoscar as xo

    from xinference.core.supervisor import SupervisorActor

    from .test_worker import DummyReplicaWorkerRef, DummyStatusGuardRef

    install(monkeypatch)
    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    old = DummyReplicaWorkerRef("worker")
    new = DummyReplicaWorkerRef("worker")
    old.get_extension_resources = AsyncMock(
        return_value={
            "address": "worker",
            "gpu_devices": [0],
            "extensions": {"test": {"node": "old"}},
        }
    )
    new.get_extension_resources = AsyncMock(
        return_value={
            "address": "worker",
            "gpu_devices": [1],
            "extensions": {"test": {"node": "new"}},
        }
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(side_effect=[old, new]))
    await supervisor.add_worker("worker")
    assert supervisor.get_worker_resource_policy("worker") == [0]
    await supervisor.add_worker("worker")
    assert supervisor._worker_address_to_worker["worker"] is new
    assert (
        supervisor._extension_worker_resources["worker"]["extensions"]["test"]["node"]
        == "new"
    )
    assert supervisor.get_worker_resource_policy("worker") == [1]
    await supervisor.remove_worker("worker")
    assert "worker" not in supervisor._worker_address_to_worker
    assert "worker" not in supervisor._extension_worker_resources


@pytest.mark.asyncio
async def test_unconfigured_supervisor_accepts_legacy_worker(monkeypatch):
    import xoscar as xo

    from xinference.core.supervisor import SupervisorActor

    from .test_worker import DummyReplicaWorkerRef, DummyStatusGuardRef

    supervisor = SupervisorActor()
    supervisor._status_guard_ref = DummyStatusGuardRef()
    worker = DummyReplicaWorkerRef("worker")
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=worker))
    supervisor._extension_worker_resources["worker"] = {"extensions": {"stale": {}}}
    await supervisor.add_worker("worker")
    assert supervisor._worker_address_to_worker["worker"] is worker
    assert "worker" not in supervisor._extension_worker_resources


@pytest.mark.asyncio
async def test_choose_worker_filters_launch_but_not_cache_placement(
    monkeypatch, caplog
):
    from xinference.core.supervisor import SupervisorActor

    install(monkeypatch)
    blocked = SimpleNamespace(
        address="blocked", get_model_count=AsyncMock(return_value=0)
    )
    permitted = SimpleNamespace(
        address="permitted", get_model_count=AsyncMock(return_value=1)
    )
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker = {"blocked": blocked, "permitted": permitted}

    def policy(address):
        if address == "blocked":
            raise ValueError("admission denied")
        return []

    monkeypatch.setattr(supervisor, "get_worker_resource_policy", policy)
    assert await supervisor._choose_worker(require_resource_admission=True) is permitted
    assert "blocked" in caplog.text and "admission denied" in caplog.text
    assert await supervisor._choose_worker(["blocked"]) is blocked
    assert await supervisor._choose_worker() is blocked
    with pytest.raises(RuntimeError, match="No available worker"):
        await supervisor._choose_worker(["blocked"], require_resource_admission=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [False, True])
async def test_primary_launch_allocation_failure_falls_back_only_without_extensions(
    monkeypatch, configured
):
    from .test_launch_strategy import DummySupervisor, DummyWorkerRef

    if configured:
        install(monkeypatch)
    launched = []
    blocked = DummyWorkerRef("blocked:1000", 0, launched)
    permitted = DummyWorkerRef("permitted:1000", 1, launched)
    blocked.get_gpu_allocation_status = AsyncMock(
        side_effect=ValueError("admission denied")
    )
    permitted.get_gpu_allocation_status = AsyncMock(
        return_value={"total": [], "models": {}, "user_specified": {}}
    )
    supervisor = DummySupervisor(
        {blocked.address: blocked, permitted.address: permitted}
    )
    await supervisor.launch_builtin_model(
        model_uid="extension-model",
        model_name="demo",
        model_size_in_billions=None,
        model_format=None,
        quantization=None,
        model_engine=None,
        model_type="LLM",
        n_gpu=None,
        wait_ready=True,
    )
    assert launched == [permitted.address if configured else blocked.address]


@pytest.mark.asyncio
async def test_scale_up_allocation_failure_keeps_default_best_effort(monkeypatch):
    from xinference.core.supervisor import SupervisorActor

    worker = SimpleNamespace(
        address="worker",
        get_model_count=AsyncMock(return_value=0),
        get_gpu_allocation_status=AsyncMock(side_effect=RuntimeError("old worker")),
    )
    supervisor = SupervisorActor()
    supervisor._worker_address_to_worker = {worker.address: worker}
    assert (await supervisor._select_worker_for_scale_up(None))[0] is worker
    assert (await supervisor._select_worker_for_scale_up(1))[0] is worker
