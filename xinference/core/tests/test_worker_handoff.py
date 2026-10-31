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

import asyncio
import contextlib
import copy
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
import xoscar as xo
from xoscar.backends.router import Router
from xoscar.core import create_local_actor_ref

from ..lifecycle import (
    WorkerLifecycle,
    WorkerPhase,
    finish_before_cancelling,
    management_operation,
)
from ..model import ModelActor
from ..supervisor import SupervisorActor
from ..worker import ModelStatus, WorkerActor
from ..worker_handoff import STATE_FIELDS, HandoffPending, WorkerSnapshot
from ..worker_registry import SupervisorHandoffMixin, WorkerOwnership, worker_message


@pytest.mark.asyncio
async def test_management_drain_preserves_concurrency_and_nested_admission():
    lifecycle = WorkerLifecycle(active=True)
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def operation():
        nonlocal count
        async with lifecycle.operation():
            count += 1
            if count == 2:
                entered.set()
            await release.wait()
            async with lifecycle.operation():
                pass

    tasks = [asyncio.create_task(operation()) for _ in range(2)]
    await entered.wait()
    drain = asyncio.create_task(lifecycle.drain(5))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="unavailable"):
        async with lifecycle.operation():
            pass
    assert not drain.done()
    release.set()
    await asyncio.gather(*tasks, drain)
    assert lifecycle.phase == WorkerPhase.DRAINING


@pytest.mark.asyncio
async def test_drain_timeout_restores_admission():
    lifecycle = WorkerLifecycle(active=True)
    entered, release = asyncio.Event(), asyncio.Event()

    async def active():
        async with lifecycle.operation():
            entered.set()
            await release.wait()

    task = asyncio.create_task(active())
    await entered.wait()
    with pytest.raises(asyncio.TimeoutError):
        await lifecycle.drain(0.01)
    assert lifecycle.phase == WorkerPhase.ACTIVE
    release.set()
    await task
    async with lifecycle.operation():
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [True, False])
async def test_repeated_cancellation_finishes_cleanup(fails):
    entered, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def operation():
        entered.set()
        await release.wait()
        completed.append(True)
        if fails:
            raise RuntimeError("preparation failed")

    task = asyncio.create_task(finish_before_cancelling(operation()))
    await entered.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert completed == [True]


def test_retired_worker_cannot_resurrect_or_remove_replacement():
    registry = WorkerOwnership()
    registry.retired["old"] = {"generation-1"}
    for registration in (True, False):
        with pytest.raises(RuntimeError, match="transferred"):
            registry.check("old", "generation-1", registration)
    registry.check("old", "generation-2", registration=True)
    registry.incarnations["old"] = "generation-2"
    with pytest.raises(RuntimeError, match="transferred"):
        registry.check("old", "generation-1")


class CounterModel(ModelActor):
    def __init__(self, worker):
        xo.StatelessActor.__init__(self)
        self._model = SimpleNamespace()
        self._worker_address = worker
        self._worker_ref = None
        self._worker_generation = 0
        self._metrics_labels = {"worker_address": worker}
        self.value = 0

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    def add(self, value):
        self.value += value
        return self.value, os.getpid(), self._worker_address

    async def wait_file(self, path):
        while not Path(path).exists():
            await asyncio.sleep(0.02)
        return self.value, os.getpid()

    @xo.generator
    async def stream(self):
        for i in range(4):
            yield self.value + i
            await asyncio.sleep(0.02)


class HandoffSupervisor(SupervisorActor):
    """Use production routing/fencing with no model catalog or telemetry services."""

    def __init__(self, ownership_path=None):
        xo.StatelessActor.__init__(self)
        self._worker_address_to_worker = {}
        self._worker_status = {}
        self._extension_worker_resources = {}
        self._replica_model_uid_to_worker = {}
        self._replica_model_uid_to_worker_shards = {}
        self._model_uid_to_replica_info = {}
        self._list_models_cache = {}
        self._reverse_ping_failures = {}
        self._worker_metadata_refresh_tasks = {}
        self._worker_metadata = {}
        self._worker_metadata_generation = {}
        self._list_models_result_cache = {}
        self._list_models_result_cache_time = 0
        self._list_models_cache_version = 0
        self._worker_model_gpu_memory = {}
        self._worker_model_gpu_memory_update_time = {}
        self._worker_ownership = WorkerOwnership(ownership_path)

    async def __post_create__(self):
        pass

    async def __pre_destroy__(self):
        pass

    @worker_message(registration=True)
    async def add_worker(self, address, replica_states=None):
        worker = await xo.actor_ref(address=address, uid=WorkerActor.default_uid())
        self._worker_address_to_worker[address] = worker
        self._extension_worker_resources[address] = {"extensions": {}}
        self._rebuild_worker_replica_state(worker, replica_states or [])

    @worker_message()
    async def receive_heartbeat(self, address):
        return True

    @worker_message()
    async def remove_worker(self, address):
        self._worker_address_to_worker.pop(address, None)

    def inspect_workers(self):
        return list(self._worker_address_to_worker)


class HandoffWorker(WorkerActor):
    async def _initialize_worker(self):
        self._actor_loop = asyncio.get_running_loop()
        from ...model.embedding.custom import (
            CustomEmbeddingModelFamilyV2,
            register_embedding,
            unregister_embedding,
        )

        self._custom_register_type_to_cls = {
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                lambda spec: {},
            )
        }

    async def _activate_worker(self, *, recover):
        self._lifecycle.phase = WorkerPhase.ACTIVE
        supervisor = await self._handoff_supervisor()
        await supervisor.add_worker(
            self.address,
            replica_states=self._get_running_replica_states(),
            incarnation=self._incarnation,
        )
        self._services_started = True

    def _persist_launch_args(self):
        pass

    def get_extension_resources(self):
        return {"extensions": {}}

    @management_operation
    async def install_model(self, uid="counter-rep0"):
        from xoscar.backends.allocate_strategy import RandomSubPool

        if not self._main_pool.sub_processes:
            await self._main_pool.append_sub_pool()
        ref = await xo.create_actor(
            CounterModel,
            self.address,
            uid=uid,
            address=self.address,
            allocate_strategy=RandomSubPool(),
        )
        self._model_uid_to_model[uid] = ref
        self._model_uid_to_model_status[uid] = ModelStatus(model_state="ready")
        self._model_uid_to_model_spec[uid] = {
            "model_name": "counter",
            "model_ability": ["generate"],
        }
        self._model_uid_to_launch_args[uid] = {"model_uid": uid}
        self._model_uid_to_addr[uid] = ref.address
        return ref

    @management_operation
    async def recover_sub_pool(self, address):
        affected = [
            uid
            for uid, ref in self._model_uid_to_model.items()
            if ref.address == address
        ]
        await self._main_pool.remove_sub_pool(address)
        for uid in affected:
            await self.install_model(uid)

    def incarnation(self):
        return self._incarnation

    def inspect_phase(self):
        return self._lifecycle.phase.value


@pytest_asyncio.fixture
async def runtime(tmp_path):
    pools = [await xo.create_actor_pool("127.0.0.1", n_process=0) for _ in range(3)]
    supervisor = await xo.create_actor(
        HandoffSupervisor,
        uid=SupervisorActor.default_uid(),
        address=pools[0].external_address,
        ownership_path=tmp_path / "handoffs.json",
    )
    workers = []
    for pool, standby in zip(pools[1:], (False, True)):
        workers.append(
            await xo.create_actor(
                HandoffWorker,
                uid=WorkerActor.default_uid(),
                address=pool.external_address,
                supervisor_address=pools[0].external_address,
                supervisor_endpoint=None,
                main_pool=pool,
                gpu_devices=[],
                standby=standby,
            )
        )
    yield pools, supervisor, workers
    with contextlib.suppress(xo.ActorNotExist):
        await xo.destroy_actor(supervisor)
    for pool in reversed(pools):
        with contextlib.suppress(Exception):
            await pool.stop()
        # Captured tracebacks can retain stopped pools after their TCP ports
        # are reused. They must no longer expose actors through local lookup.
        pool._actors.clear()
    Router.set_instance(None)


@pytest.mark.asyncio
async def test_runtime_teardown_releases_local_actors(tmp_path):
    fixture = runtime.__wrapped__(tmp_path)
    pools, supervisor, workers = await anext(fixture)
    supervisor = await xo.actor_ref(address=supervisor.address, uid=supervisor.uid)
    with pytest.raises(StopAsyncIteration):
        await anext(fixture)
    # Keep the stopped pools alive to reproduce a later test reusing an address.
    assert pools
    for actor in [supervisor, *workers]:
        assert create_local_actor_ref(actor.address, actor.uid) is None
    with pytest.raises(xo.ActorNotExist):
        await supervisor.inspect_workers()


@pytest.mark.asyncio
async def test_handoff_preserves_model_and_fences_old_owner(runtime):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    value, pid, _ = await model.add(5)
    old_incarnation = await old.incarnation()
    assert await new.replace_worker(old.address)
    assert await model.add(1) == (value + 1, pid, new.address)
    assert await supervisor.inspect_workers() == [new.address]
    assert not await supervisor.can_recover_model(old.address, "counter-rep0")
    assert await supervisor.can_recover_model(new.address, "counter-rep0")
    for method in ("receive_heartbeat", "remove_worker", "add_worker"):
        with pytest.raises(RuntimeError, match="transferred"):
            await getattr(supervisor, method)(old.address, incarnation=old_incarnation)
    await asyncio.wait_for(pools[1].join(), 5)
    assert await model.add(1) == (value + 2, pid, new.address)
    assert await new.replace_worker(old.address)  # Acknowledgement is idempotent.
    other = await new.install_model("another-rep0")
    assert (await other.add(1))[0] == 1


@pytest.mark.asyncio
async def test_custom_catalog_snapshot_includes_writes_finished_during_reservation(
    runtime, monkeypatch
):
    from ...model.custom import RegistryManager
    from ...model.embedding.custom import (
        CustomEmbeddingModelFamilyV2,
        register_embedding,
    )
    from ...model.embedding.embed_family import BUILTIN_EMBEDDING_MODELS

    _, supervisor, (old, new) = runtime
    definition = next(iter(BUILTIN_EMBEDDING_MODELS.values()))[0].dict()
    definition["model_name"] = "handoff-reservation-catalog"
    definition["is_builtin"] = False
    original = HandoffSupervisor.prepare_worker_handoff

    async def reserve(self, snapshot, timeout):
        result = await original(self, snapshot, timeout)
        register_embedding(
            CustomEmbeddingModelFamilyV2.parse_obj(definition), persist=False
        )
        return result

    monkeypatch.setattr(HandoffSupervisor, "prepare_worker_handoff", reserve)
    try:
        snapshot = await old.prepare_worker_handoff(
            new.address, await new.worker_incarnation(), "catalog-ticket", 5
        )
        assert definition["model_name"] in snapshot.custom_models["embedding"]
    finally:
        await old.abort_worker_handoff("catalog-ticket")
        registry = RegistryManager.get_registry("embedding")
        if registry.find_model(definition["model_name"]):
            registry.unregister(definition["model_name"], remove_file=False)


@pytest.mark.asyncio
async def test_incompatible_snapshot_rolls_back(runtime, monkeypatch):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    from ..worker_handoff import WorkerSnapshot

    def fail(self, worker):
        raise ValueError("incompatible state")

    with monkeypatch.context() as patch:
        patch.setattr(WorkerSnapshot, "validate", fail)
        with pytest.raises(ValueError, match="incompatible"):
            await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"
    assert await new.inspect_phase() == "prepared"
    assert await supervisor.inspect_workers() == [old.address]
    assert (await model.add(2))[0] == 2
    assert await new.replace_worker(old.address)


_SOURCE = r"""
import asyncio, json, sys
from pathlib import Path
import xoscar as xo
from xinference.core.worker import WorkerActor
from xinference.core.tests.test_worker_handoff import HandoffWorker
async def main():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    ref = await xo.create_actor(HandoffWorker, uid=WorkerActor.default_uid(), address=pool.external_address, supervisor_address=sys.argv[1], supervisor_endpoint=None, main_pool=pool, gpu_devices=[])
    from xinference.model.embedding.embed_family import BUILTIN_EMBEDDING_MODELS
    definition = next(iter(BUILTIN_EMBEDDING_MODELS.values()))[0].dict()
    definition["model_name"] = "handoff-in-memory-custom"
    definition["is_builtin"] = False
    await ref.register_model("embedding", json.dumps(definition), False)
    model = await ref.install_model()
    value, pid, _ = await model.add(7)
    Path(sys.argv[2]).write_text(json.dumps({"worker":ref.address,"model":model.address,"pid":pid}))
    await pool.join()
asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_old_process_exits_while_direct_call_and_stream_survive(
    runtime, tmp_path
):
    pools, supervisor, (_, new) = runtime
    ready = tmp_path / "ready.json"
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _SOURCE, pools[0].external_address, str(ready)
    )
    try:

        async def wait_ready():
            while not ready.exists():
                if process.returncode is not None:
                    raise RuntimeError("Source worker exited before readiness")
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait_ready(), 120 if sys.platform == "win32" else 45)
        data = json.loads(ready.read_text())
        model = await xo.actor_ref(address=data["model"], uid="counter-rep0")
        release = tmp_path / "release"
        pending = asyncio.create_task(model.wait_file(str(release)))
        stream = await model.stream()
        assert await stream.__anext__() == 7
        assert await new.replace_worker(data["worker"])
        assert await asyncio.wait_for(process.wait(), 10) == 0
        release.touch()
        assert await pending == (7, data["pid"])
        assert [value async for value in stream] == [8, 9, 10]
        assert await model.add(2) == (9, data["pid"], new.address)
        registrations = await new.list_model_registrations("embedding")
        assert any(
            item["model_name"] == "handoff-in-memory-custom" for item in registrations
        )
        await stream.destroy()
        stream._gc_destroy = False
    finally:
        from ...model.custom import RegistryManager

        registry = RegistryManager.get_registry("embedding")
        if registry.find_model("handoff-in-memory-custom"):
            registry.unregister("handoff-in-memory-custom", remove_file=False)
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_empty_worker_replacement(runtime):
    pools, supervisor, (old, new) = runtime
    assert await new.replace_worker(old.address)
    assert await supervisor.inspect_workers() == [new.address]
    await asyncio.wait_for(pools[1].join(), 5)
    model = await new.install_model()
    assert (await model.add(1))[0] == 1


@pytest.mark.asyncio
async def test_postcommit_reply_loss_retries_without_rollback(runtime, monkeypatch):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    value, pid, _ = await model.add(3)
    commit = HandoffWorker._commit_worker_handoff
    dropped = False

    async def drop(self, ticket, timeout):
        nonlocal dropped
        await commit(self, ticket, timeout)
        if not dropped:
            dropped = True
            raise ConnectionResetError("lost commit acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(HandoffWorker, "_commit_worker_handoff", drop)
        with pytest.raises(HandoffPending):
            await new.replace_worker(old.address)
    assert await old.inspect_phase() == "transferred"
    assert await new.replace_worker(old.address)
    assert await model.add(1) == (value + 1, pid, new.address)


@pytest.mark.asyncio
async def test_unsupported_model_layout_fails_before_ownership_changes(runtime):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    # Keep the test at the same state boundary used by real launch snapshots.
    actor = pools[1]._actors[old.uid]
    actor._model_uid_to_launch_args["counter-rep0"]["n_worker"] = 2
    with pytest.raises(ValueError, match="one worker"):
        await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"
    assert (await model.add(1))[2] == old.address


@pytest.mark.asyncio
@pytest.mark.parametrize("starting", [False, True])
async def test_unfinished_startup_cannot_be_snapshotted(runtime, starting):
    pools, _, (old, new) = runtime
    model = await old.install_model()
    actor = pools[1]._actors[old.uid]
    if starting:
        actor._services_started = False
    else:
        actor._model_uid_to_model_status["counter-rep0"].model_state = "loading"
    with pytest.raises(RuntimeError, match="still starting|not ready"):
        await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"
    assert await new.inspect_phase() == "prepared"
    assert (await model.add(1))[2] == old.address


@pytest.mark.asyncio
async def test_repeated_replacement_and_new_models_keep_binding_generations(runtime):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    await model.add(4)
    assert await new.replace_worker(old.address)
    other = await new.install_model("other-rep0")
    async with await xo.create_actor_pool("127.0.0.1", n_process=0) as third_pool:
        third = await xo.create_actor(
            HandoffWorker,
            uid=WorkerActor.default_uid(),
            address=third_pool.external_address,
            supervisor_address=pools[0].external_address,
            supervisor_endpoint=None,
            main_pool=third_pool,
            gpu_devices=[],
            standby=True,
        )
        assert await third.replace_worker(new.address)
        assert (await model.add(1))[0::2] == (5, third.address)
        assert (await other.add(1))[2] == third.address


@pytest.mark.asyncio
async def test_supervisor_protocol_mismatch_does_not_freeze_source(
    runtime, monkeypatch
):
    _, _, (old, new) = runtime
    await old.install_model()
    with monkeypatch.context() as patch:
        patch.setattr(HandoffSupervisor, "worker_handoff_protocol", lambda self: 0)
        with pytest.raises(ValueError, match="Supervisor handoff protocol"):
            await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"
    assert await new.inspect_phase() == "prepared"


@pytest.mark.asyncio
async def test_stale_model_binding_does_not_freeze_source(runtime):
    _, _, (old, new) = runtime
    model = await old.install_model()
    await model.rebind_worker(old.address, "different-owner", 1)
    with pytest.raises(ValueError, match="Model process"):
        await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"


@pytest.mark.parametrize("phase", [WorkerPhase.DRAINING, WorkerPhase.TRANSFERRED])
@pytest.mark.asyncio
async def test_sync_management_uses_the_same_admission_gate(phase):
    class Owner:
        _lifecycle = WorkerLifecycle(active=True)

        @management_operation
        def mutation(self):
            return True

    owner = Owner()
    owner._lifecycle.phase = phase
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.mutation()


def test_worker_metadata_requires_negotiated_supervisor_capability():
    owner = SimpleNamespace(
        _incarnation="current", _supervisor_ownership_supported=False
    )
    assert WorkerActor._ownership_metadata(owner, "add_worker") == {}
    owner._supervisor_ownership_supported = True
    assert WorkerActor._ownership_metadata(owner, "add_worker") == {
        "incarnation": "current"
    }
    assert WorkerActor._ownership_metadata(owner, "get_model") == {}


def test_worker_background_shutdown_is_idempotent():
    calls = []
    owner = SimpleNamespace(
        _isolation=SimpleNamespace(stop=lambda: calls.append("stopped"))
    )
    WorkerActor._stop_worker_services(owner)
    WorkerActor._stop_worker_services(owner)
    assert calls == ["stopped"]


_REAL_RUNTIME = r"""
import asyncio
import json
import xoscar as xo
from xoscar.backends.allocate_strategy import RandomSubPool
from xinference.core.supervisor import SupervisorActor
from xinference.core.worker import WorkerActor
from xinference.core.tests.test_worker_handoff import CounterModel

async def main():
    pools=[]
    workers=[]
    try:
        for _ in range(3):
            pools.append(await xo.create_actor_pool('127.0.0.1', n_process=0, auto_recover='process'))
        sup=await xo.create_actor(SupervisorActor, uid=SupervisorActor.default_uid(), address=pools[0].external_address)
        print('supervisor ready',flush=True)
        for pool, standby in zip(pools[1:],(False,True)):
            workers.append(await xo.create_actor(WorkerActor, uid=WorkerActor.default_uid(), address=pool.external_address, supervisor_address=pools[0].external_address, supervisor_endpoint=None, main_pool=pool, gpu_devices=[], standby=standby))
        print('real workers ready',flush=True)
        old,new=workers
        await pools[1].append_sub_pool()
        model=await xo.create_actor(CounterModel,old.address,uid='counter-rep0',address=old.address,allocate_strategy=RandomSubPool())
        owner=pools[1]._actors[old.uid]
        owner._model_uid_to_model['counter-rep0']=model
        owner._model_uid_to_model_spec['counter-rep0']={'model_name':'counter','model_ability':['generate']}
        owner._model_uid_to_launch_args['counter-rep0']={'model_uid':'counter-rep0'}
        owner._model_uid_to_addr['counter-rep0']=model.address
        await sup.add_worker(old.address,replica_states=owner._get_running_replica_states(),incarnation=owner._incarnation)
        before=await model.add(8)
        assert await new.replace_worker(old.address)
        after=await model.add(1)
        assert (await sup.get_model('counter')).address==model.address
        assert after==(9,before[1],new.address)
        await pools[1].join()
        print(json.dumps({'before':before,'after':after,'source_stopped':pools[1].stopped}),flush=True)
        await xo.destroy_actor(new)
        await xo.destroy_actor(sup)
    finally:
        for pool in reversed(pools):
            await pool.stop()

asyncio.run(main())

"""


@pytest.mark.asyncio
async def test_real_worker_registration_heartbeat_and_shutdown(tmp_path):
    env = dict(
        os.environ,
        XINFERENCE_HOME=str(tmp_path),
        XINFERENCE_DISABLE_METRICS="1",
        XINFERENCE_TOKEN_ROUTER_ENABLED="0",
        XINFERENCE_DISABLE_HEALTH_CHECK="0",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _REAL_RUNTIME,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(
            process.communicate(), 180 if sys.platform == "win32" else 60
        )
        assert process.returncode == 0, output.decode()
        line = next(
            line
            for line in output.decode().splitlines()
            if line.startswith('{"before":')
        )
        result = json.loads(line)
        assert result["source_stopped"]
        assert result["before"][1] == result["after"][1]
        assert result["after"][0] == result["before"][0] + 1
        assert result["after"][2] != result["before"][2]
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_adopted_process_failure_is_recovered_by_new_worker(runtime):
    pools, _, (old, new) = runtime
    model = await old.install_model()
    _, pid, _ = await model.add(2)
    await new.replace_worker(old.address)
    pools[2].sub_processes[model.address].kill()

    async def recovered():
        while True:
            ref = await new.get_model("counter-rep0")
            if ref.address != model.address:
                return await ref.add(1)
            await asyncio.sleep(0.05)

    # A replacement imports the runtime in a fresh Windows process too.
    value, new_pid, owner = await asyncio.wait_for(
        recovered(), 60 if sys.platform == "win32" else 15
    )
    assert value == 1 and new_pid != pid and owner == new.address


@pytest.mark.asyncio
async def test_abandoned_preparation_restores_old_worker(runtime, monkeypatch):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    prepared = asyncio.Event()
    expire = HandoffWorker._expire_worker_handoff

    async def expire_after_preparation(worker, ticket, timeout):
        if ticket == "abandoned":
            await prepared.wait()
            timeout = 0.15
        await expire(worker, ticket, timeout)

    # Test lease expiry after successful preparation without imposing a 150ms
    # budget on the reservation RPC and its durable filesystem writes.
    monkeypatch.setattr(
        HandoffWorker, "_expire_worker_handoff", expire_after_preparation
    )
    await old.prepare_worker_handoff(
        new.address, await new.incarnation(), "abandoned", 5
    )
    assert await old.inspect_phase() == "draining"
    prepared.set()

    async def resumed():
        while await old.inspect_phase() != "active":
            await asyncio.sleep(0.02)

    await asyncio.wait_for(resumed(), 5)
    assert not pools[0]._actors[supervisor.uid]._ownership_registry().pending
    assert (await model.add(1))[2] == old.address
    assert await new.replace_worker(old.address)


@pytest.mark.parametrize(
    "problem", ["version", "identity", "supervisor", "devices", "fields"]
)
def test_snapshot_preflight_rejects_incompatible_targets(monkeypatch, problem):
    from .. import worker_handoff

    monkeypatch.setattr(worker_handoff, "is_metrics_disabled", lambda: False)
    owner = SimpleNamespace(
        address="new",
        _incarnation="new-instance",
        _supervisor_address="supervisor",
        _total_gpu_devices=[0],
        _metrics_exporter_config=("127.0.0.1", 19002),
    )
    snapshot = WorkerSnapshot(
        1,
        "ticket",
        "old",
        "new",
        "supervisor",
        "old-instance",
        "new-instance",
        1,
        [0],
        {field: {} for field in STATE_FIELDS},
        metrics_port=19001,
    )
    snapshot.validate(owner)
    if problem == "version":
        snapshot.version = 2
    elif problem == "identity":
        snapshot.target_incarnation = "another-instance"
    elif problem == "supervisor":
        snapshot.supervisor = "another-supervisor"
    elif problem == "devices":
        snapshot.devices = [1]
    elif problem == "fields":
        snapshot.state.pop(STATE_FIELDS[0])
    with pytest.raises(ValueError):
        snapshot.validate(owner)


@pytest.mark.asyncio
async def test_lost_commit_reply_and_permanently_dead_source_finish_transfer(
    runtime, monkeypatch
):
    pools, _, (old, new) = runtime
    model = await old.install_model()
    commit = HandoffWorker._commit_worker_handoff

    async def lose_reply(self, ticket, timeout):
        await commit(self, ticket, timeout)
        raise ConnectionResetError("lost acknowledgement")

    with monkeypatch.context() as patch:
        patch.setattr(HandoffWorker, "_commit_worker_handoff", lose_reply)
        with pytest.raises(HandoffPending):
            await new.replace_worker(old.address)
    actor_ref = xo.actor_ref

    async def disconnected(*args, **kwargs):
        if kwargs.get("address") == old.address:
            raise ConnectionRefusedError("source unreachable")
        return await actor_ref(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(xo, "actor_ref", disconnected)
        assert await new.replace_worker(old.address)
    assert pools[2].sub_processes
    assert (await model.add(1)) == (
        1,
        pools[2].sub_processes[model.address].pid,
        new.address,
    )


@pytest.mark.asyncio
async def test_model_failure_between_pool_commit_and_binding_repair_recovers(
    runtime, monkeypatch
):
    pools, _, (old, new) = runtime
    model = await old.install_model()
    _, original_pid, _ = await model.add(5)
    commit = HandoffWorker._commit_worker_handoff
    killed = False

    async def lose_process(self, ticket, timeout):
        nonlocal killed
        result = await commit(self, ticket, timeout)
        if not killed:
            killed = True
            pools[2].sub_processes[model.address].kill()
        return result

    async def complete():
        while True:
            try:
                return await new.replace_worker(old.address, timeout=2)
            except HandoffPending:
                await asyncio.sleep(0.05)

    with monkeypatch.context() as patch:
        patch.setattr(HandoffWorker, "_commit_worker_handoff", lose_process)
        assert await asyncio.wait_for(complete(), 60 if sys.platform == "win32" else 20)
    recovered = await new.get_model("counter-rep0")
    _, recovered_pid, owner = await recovered.add(1)
    assert recovered_pid != original_pid and owner == new.address
    await asyncio.wait_for(pools[1].join(), 5)


@pytest.mark.parametrize("conflict", [False, True])
def test_custom_catalog_state_is_restored_without_persistence(monkeypatch, conflict):
    import uuid

    from ...model.custom import RegistryManager
    from ...model.embedding.custom import (
        CustomEmbeddingModelFamilyV2,
        register_embedding,
        unregister_embedding,
    )
    from ...model.embedding.embed_family import BUILTIN_EMBEDDING_MODELS
    from ..worker_handoff import CustomModelState

    name = "handoff-catalog-" + uuid.uuid4().hex
    definition = next(iter(BUILTIN_EMBEDDING_MODELS.values()))[0].dict()
    definition.update(model_name=name, is_builtin=False)
    spec = CustomEmbeddingModelFamilyV2.parse_obj(definition)
    worker = SimpleNamespace(
        _custom_register_type_to_cls={
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                None,
            )
        }
    )
    registry = RegistryManager.get_registry("embedding")
    register_embedding(spec, persist=False)
    try:
        snapshot = CustomModelState.capture(worker)
        if conflict:
            payload = json.loads(snapshot["embedding"][name])
            payload["dimensions"] = payload["dimensions"] + 1
            snapshot["embedding"][name] = json.dumps(payload)
            with pytest.raises(ValueError, match="definition differs"):
                CustomModelState.restore(worker, snapshot)
            assert registry.find_model(name).dimensions == spec.dimensions
        else:
            registry.unregister(name, remove_file=False)
            # Persistence would touch files shared with the source Worker.
            from ...model.cache_manager import CacheManager

            def reject_write(*args, **kwargs):
                raise AssertionError("Handoff must not persist custom registrations")

            monkeypatch.setattr(CacheManager, "register_custom_model", reject_write)
            CustomModelState.restore(worker, snapshot)
            assert registry.find_model(name).dict() == spec.dict()
    finally:
        if registry.find_model(name):
            registry.unregister(name, remove_file=False)


@pytest.mark.asyncio
async def test_validation_failure_needs_no_unrelated_supervisor_rollback(
    runtime, monkeypatch
):
    _, _, (old, new) = runtime
    model = await old.install_model()
    await model.rebind_worker(old.address, "different-owner", 1)

    async def disconnected(*args):
        raise ConnectionResetError("abort endpoint unreachable")

    with monkeypatch.context() as patch:
        patch.setattr(HandoffSupervisor, "abort_worker_handoff", disconnected)
        with pytest.raises(ValueError, match="Model process"):
            await new.replace_worker(old.address)
    assert await old.inspect_phase() == "active"


@pytest.mark.asyncio
async def test_inflight_worker_lookup_cannot_restore_retired_binding(monkeypatch):
    owner = SimpleNamespace(
        _worker_address="old",
        _worker_ref=None,
        _worker_generation=0,
        _metrics_labels={"worker_address": "old"},
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def lookup(*, address, uid):
        if address == "old":
            entered.set()
            await release.wait()
        return SimpleNamespace(address=address)

    monkeypatch.setattr(xo, "actor_ref", lookup)
    pending = asyncio.create_task(ModelActor._get_worker_ref(owner))
    await entered.wait()
    ModelActor.rebind_worker(owner, "old", "new", 1)
    release.set()
    assert (await pending).address == "new"
    assert owner._worker_ref.address == "new"


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True])
async def test_inflight_model_route_retries_only_after_owner_changes(replace):
    from ..worker_registry import call_worker_route

    calls = []

    class Current:
        async def get_model(self, **kwargs):
            calls.append("new")
            return "live-model"

    current = Current()
    owner = SimpleNamespace(_replica_model_uid_to_worker={})

    class Previous:
        async def get_model(self, **kwargs):
            calls.append("old")
            if replace:
                owner._replica_model_uid_to_worker["counter-rep0"] = current
            raise ConnectionResetError("old control connection closed")

    previous = Previous()
    owner._replica_model_uid_to_worker["counter-rep0"] = previous
    if replace:
        assert (
            await call_worker_route(owner, "counter-rep0", previous, "get_model")
            == "live-model"
        )
        assert calls == ["old", "new"]
    else:
        with pytest.raises(ConnectionResetError):
            await call_worker_route(owner, "counter-rep0", previous, "get_model")
        assert calls == ["old"]


@pytest.mark.asyncio
async def test_scheduled_management_is_drained_before_first_run():
    lifecycle = WorkerLifecycle(active=True)
    release = asyncio.Event()

    async def update():
        await release.wait()

    task = lifecycle.start_background(update())
    drain = asyncio.create_task(lifecycle.drain(5))
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    await task
    await drain
    skipped = update()
    assert lifecycle.start_background(skipped) is None
    assert skipped.cr_frame is None


@pytest.mark.asyncio
async def test_cancelled_unstarted_background_task_releases_admission():
    lifecycle = WorkerLifecycle(active=True)
    task = lifecycle.start_background(asyncio.sleep(100))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await lifecycle.drain(1)


@pytest.mark.asyncio
async def test_admitted_operation_can_finish_background_work_during_drain():
    lifecycle = WorkerLifecycle(active=True)
    entered, release = asyncio.Event(), asyncio.Event()
    finished = []

    async def update():
        await asyncio.sleep(0)
        finished.append(True)

    async def operation():
        async with lifecycle.operation():
            entered.set()
            await release.wait()
            assert lifecycle.start_background(update()) is not None

    task = asyncio.create_task(operation())
    await entered.wait()
    drain = asyncio.create_task(lifecycle.drain(1))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(task, drain)
    assert finished == [True]


@pytest.mark.asyncio
async def test_model_load_completion_is_drained_before_snapshot():
    entered, release = asyncio.Event(), asyncio.Event()
    states = []

    class Model:
        async def wait_for_load(self):
            entered.set()
            await release.wait()

    async def update(uid, state):
        states.append((uid, state))

    owner = SimpleNamespace(
        _lifecycle=WorkerLifecycle(active=True),
        _model_uid_to_model={"model": Model()},
        _update_model_state=update,
    )
    task = asyncio.create_task(WorkerActor.wait_for_load(owner, "model"))
    await entered.wait()
    drain = asyncio.create_task(owner._lifecycle.drain(1))
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    await asyncio.gather(task, drain)
    assert states == [("model", "ready")]


def test_cache_locations_follow_replacement_without_stale_owner():
    from ..cache_tracker import CacheTrackerActor

    tracker = CacheTrackerActor()
    tracker.record_model_version(
        {
            "model": [
                {
                    "model_version": "v1",
                    "cache_status": True,
                    "model_file_location": "/cache/model",
                }
            ]
        },
        "old",
    )
    tracker.rebind_worker("old", "new")
    tracker.rebind_worker("old", "new")
    assert tracker.list_cached_models("old") == []
    assert tracker.list_cached_models("new")[0]["model_file_location"] == "/cache/model"
    tracker.confirm_and_remove_model("v1", "new")
    assert not tracker.get_model_versions("model")[0]["cache_status"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["none", "reserved", "replaced", "missing"])
async def test_health_cleanup_rechecks_ownership_after_waiting(change):
    registry = WorkerOwnership()
    original = object()
    cleaned = []

    async def handle(address):
        cleaned.append(address)

    owner = SimpleNamespace(
        _ownership_registry=lambda: registry,
        _worker_address_to_worker={"worker": original},
        _worker_status={"worker": object()},
        _extension_worker_resources={"worker": {}},
        _handle_dead_worker=handle,
        _discard_worker_metadata=lambda address: None,
        _clear_worker_model_gpu_memory=lambda address: None,
    )
    async with registry.lock("worker"):
        removal = asyncio.create_task(
            SupervisorHandoffMixin._remove_dead_worker(owner, "worker", original)
        )
        await asyncio.sleep(0)
        if change == "reserved":
            registry.pending["worker"] = {"committed": True}
        elif change == "replaced":
            owner._worker_address_to_worker["worker"] = object()
        elif change == "missing":
            owner._worker_address_to_worker.pop("worker")
    removed = change in ("none", "missing")
    assert await removal is removed
    assert cleaned == (["worker"] if removed else [])
    assert ("worker" in owner._worker_address_to_worker) is (not removed)


@pytest.mark.asyncio
@pytest.mark.parametrize("after_commit", [False, True])
async def test_supervisor_restart_during_handoff_restores_ownership(
    runtime, monkeypatch, after_commit
):
    pools, supervisor, (old, new) = runtime
    model = await old.install_model()
    before = await model.add(5)
    path = pools[0]._actors[supervisor.uid]._ownership_registry().path
    original = HandoffWorker._commit_worker_handoff
    restarted = False

    async def restart():
        await xo.destroy_actor(supervisor)
        await xo.create_actor(
            HandoffSupervisor,
            uid=SupervisorActor.default_uid(),
            address=pools[0].external_address,
            ownership_path=path,
        )

    async def commit(self, ticket, timeout):
        nonlocal restarted
        if not restarted:
            restarted = True
            if not after_commit:
                await restart()
            result = await original(self, ticket, timeout)
            if after_commit:
                await restart()
            return result
        return await original(self, ticket, timeout)

    with monkeypatch.context() as patch:
        patch.setattr(HandoffWorker, "_commit_worker_handoff", commit)
        assert await new.replace_worker(old.address)
    assert await supervisor.inspect_workers() == [new.address]
    after = await model.add(1)
    assert after == (6, before[1], new.address)
    assert not pools[0]._actors[supervisor.uid]._ownership_registry().pending


@pytest.mark.asyncio
async def test_abort_rpc_failure_restores_admission_and_allows_new_handoff(
    runtime, monkeypatch
):
    from .. import worker_registry

    pools, supervisor, (old, new) = runtime
    await old.install_model()
    prepared = asyncio.Event()
    aborted = asyncio.Event()
    expire = HandoffWorker._expire_worker_handoff

    async def expire_after_preparation(worker, ticket, timeout):
        if ticket == "expired":
            await prepared.wait()
            timeout = 0
        await expire(worker, ticket, timeout)

    async def disconnected(*args):
        aborted.set()
        raise ConnectionResetError("Supervisor abort unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(HandoffSupervisor, "abort_worker_handoff", disconnected)
        patch.setattr(HandoffWorker, "_expire_worker_handoff", expire_after_preparation)
        await old.prepare_worker_handoff(
            new.address, await new.incarnation(), "expired", 5
        )
        prepared.set()
        await asyncio.wait_for(aborted.wait(), 5)

        async def resumed():
            while await old.inspect_phase() != "active":
                await asyncio.sleep(0.02)

        await asyncio.wait_for(resumed(), 5)
        await old.install_model("second-rep0")
    registry = pools[0]._actors[supervisor.uid]._ownership_registry()
    assert registry.pending
    # Expire the independent Supervisor lease without imposing a tiny timeout
    # on preparation RPCs or waiting for the normal lease duration.
    deadline = registry.pending[old.address]["deadline"]
    monkeypatch.setattr(
        worker_registry,
        "time",
        SimpleNamespace(time=worker_registry.time.time, monotonic=lambda: deadline),
    )
    assert not registry.protected(old.address)
    assert await new.replace_worker(old.address)


@pytest.mark.asyncio
async def test_late_commit_cannot_revive_a_worker_removed_by_health_cleanup(runtime):
    pools, supervisor, (old, new) = runtime
    snapshot = await old.prepare_worker_handoff(
        new.address, await new.incarnation(), "late", 30
    )
    pools[0]._actors[supervisor.uid]._worker_address_to_worker.pop(old.address)
    with pytest.raises(RuntimeError, match="no longer registered"):
        await supervisor.commit_worker_handoff(snapshot)


@pytest.mark.asyncio
async def test_restarted_supervisor_waits_for_retained_models_before_autostart(runtime):
    pools, supervisor, (old, new) = runtime
    await old.install_model()
    await old.prepare_worker_handoff(
        new.address, await new.incarnation(), "reserved", 30
    )
    registry = pools[0]._actors[supervisor.uid]._ownership_registry()
    restored = WorkerOwnership(registry.path)
    owner = SimpleNamespace(
        _worker_ownership=restored, is_local_deployment=lambda: False
    )
    assert SupervisorActor._autostart_waiting_for_worker(
        owner, {"model_uid": "counter"}
    )


def test_catalog_replacement_validates_before_deletion_and_does_not_persist(
    monkeypatch,
):
    import uuid

    from pydantic.v1 import ValidationError

    from ...model.custom import RegistryManager
    from ...model.embedding.custom import (
        CustomEmbeddingModelFamilyV2,
        register_embedding,
        unregister_embedding,
    )
    from ...model.embedding.embed_family import BUILTIN_EMBEDDING_MODELS
    from ..worker_handoff import CustomModelState

    registry = RegistryManager.get_registry("embedding")
    original = list(registry.get_custom_models())
    definition = next(iter(BUILTIN_EMBEDDING_MODELS.values()))[0].dict()
    name = "replacement-" + uuid.uuid4().hex
    definition.update(model_name=name, is_builtin=False)
    spec = CustomEmbeddingModelFamilyV2.parse_obj(definition)
    owner = SimpleNamespace(
        _custom_register_type_to_cls={
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                None,
            )
        }
    )
    try:
        register_embedding(spec, persist=False)
        before = CustomModelState.capture(owner)
        with pytest.raises(ValidationError):
            CustomModelState.restore(owner, {"embedding": {name: "{}"}}, replace=True)
        assert CustomModelState.capture(owner) == before
        # A valid replacement removes stale in-memory entries, without files.
        CustomModelState.restore(owner, {"embedding": {}}, replace=True)
        assert not registry.get_custom_models()
        CustomModelState.restore(owner, before, replace=True)
        assert registry.find_model(name).model_name == name
    finally:
        for item in list(registry.get_custom_models()):
            registry.unregister(item.model_name, remove_file=False)
        for item in original:
            register_embedding(item, persist=False)


@pytest.mark.parametrize("failure", ["uri", "engine"])
def test_failed_catalog_replacement_restores_registrations_and_engines(
    tmp_path, monkeypatch, failure
):
    from ...model import embedding
    from ...model.custom import RegistryManager
    from ...model.embedding.custom import (
        CustomEmbeddingModelFamilyV2,
        register_embedding,
        unregister_embedding,
    )
    from ...model.embedding.embed_family import (
        BUILTIN_EMBEDDING_MODELS,
        EMBEDDING_ENGINES,
    )
    from ..worker_handoff import CustomModelState

    registry = RegistryManager.get_registry("embedding")
    original = list(registry.get_custom_models())
    definition = next(iter(BUILTIN_EMBEDDING_MODELS.values()))[0].dict()
    prefix = "replacement-" + uuid.uuid4().hex
    definition.update(model_name=prefix, is_builtin=False)
    owner = SimpleNamespace(
        _custom_register_type_to_cls={
            "embedding": (
                CustomEmbeddingModelFamilyV2,
                register_embedding,
                unregister_embedding,
                None,
            )
        }
    )
    try:
        register_embedding(CustomEmbeddingModelFamilyV2.parse_obj(definition), False)
        before = CustomModelState.capture(owner)
        engines = copy.deepcopy(EMBEDDING_ENGINES)
        replacement = {}
        for suffix in ("first", "rejected"):
            model = copy.deepcopy(definition)
            model["model_name"] = prefix + "-" + suffix
            if failure == "uri" and suffix == "rejected":
                model["model_specs"][0]["model_uri"] = (tmp_path / "missing").as_uri()
            replacement[model["model_name"]] = json.dumps(model)
        if failure == "engine":
            generate = embedding.generate_engine_config_by_model_name

            def reject_after_generation(spec):
                generate(spec)
                if spec.model_name.endswith("-rejected"):
                    raise ValueError("Engine registration rejected")

            monkeypatch.setattr(
                embedding,
                "generate_engine_config_by_model_name",
                reject_after_generation,
            )
        with pytest.raises(ValueError):
            CustomModelState.restore(owner, {"embedding": replacement}, replace=True)
        assert CustomModelState.capture(owner) == before
        assert EMBEDDING_ENGINES == engines
    finally:
        for item in list(registry.get_custom_models()):
            registry.unregister(item.model_name, remove_file=False)
        for item in original:
            register_embedding(item, persist=False)
