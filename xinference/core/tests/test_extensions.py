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

import pytest

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


@pytest.mark.parametrize("configured", ["missing:create", "missing", "module:"])
def test_configured_extension_failure_is_not_silent(monkeypatch, configured):
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", configured)
    with pytest.raises((ValueError, ModuleNotFoundError)):
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
    assert "explicit" not in worker._gpu_to_model_uids.get(2, set())


@pytest.mark.asyncio
async def test_worker_revalidates_after_preparation(monkeypatch):
    from xinference.core import worker as worker_module

    worker = worker_module.WorkerActor("supervisor", None, SimpleNamespace(), [0, 1])
    monkeypatch.setattr(worker_module, "_snapshot_gpu_free_ratio", lambda devices: 1)

    async def reduced():
        return [0]

    monkeypatch.setattr(worker, "_get_allowed_gpu_devices", reduced)
    with pytest.raises(ValueError, match="Reserved GPUs"):
        await worker._spawn_subpool("prepared", {}, ["1"])


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
