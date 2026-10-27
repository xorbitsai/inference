# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ...core.router_registry import RouterRuntimeRegistry
from ...core.supervisor_upgrade import SupervisorGateway
from ...core.system_settings_store import SystemSettingsStore
from ...core.worker import WorkerActor
from ..supervisor_host import SupervisorHost


class Model(xo.StatelessActor):
    def identity(self):
        return os.getpid()


class Worker(xo.StatelessActor):
    def __init__(self, model, replica_model_uid="model-rep0"):
        super().__init__()
        self.model = model
        self.replica_model_uid = replica_model_uid
        self.incarnation = uuid.uuid4().hex
        self.inventory_available = True

    def worker_incarnation(self):
        return self.incarnation

    def get_extension_resources(self):
        return {"extensions": {}}

    def get_node_metadata(self):
        return {}

    def update_system_settings(self, settings):
        pass

    def list_models(self):
        if not self.inventory_available:
            raise RuntimeError("Worker inventory unavailable")
        return {self.replica_model_uid: {}}

    def get_model(self, model_uid):
        assert model_uid == self.replica_model_uid
        return self.model

    def set_inventory_available(self, available):
        self.inventory_available = available


class RestoreWorker(Worker):
    def __init__(self, *args):
        super().__init__(*args)
        self.inspected = asyncio.Event()
        self.release_inventory = asyncio.Event()
        self.hold_inventory = False
        self.model_alive = True

    async def list_models(self):
        inventory = super().list_models()
        self.inspected.set()
        if self.hold_inventory:
            await self.release_inventory.wait()
            if not self.inventory_available:
                raise RuntimeError("Worker inventory unavailable")
        return inventory

    async def wait_inspected(self):
        await self.inspected.wait()

    def block_inventory(self):
        self.inspected.clear()
        self.hold_inventory = True

    def finish_inventory(self, available):
        self.inventory_available = available
        self.release_inventory.set()

    def stop_model(self):
        self.model_alive = False

    def get_model(self, model_uid):
        if not self.model_alive:
            raise RuntimeError("Model stopped")
        return super().get_model(model_uid)


class Control(xo.StatelessActor):
    def __init__(self, name, gateway):
        super().__init__()
        self.name = name
        self.gateway = gateway
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.callbacks = 0

    async def management(self):
        self.entered.set()
        await self.release.wait()
        await self.gateway.report_worker_status()
        return self.name

    def get_model(self):
        return self.name

    describe_running_model = get_model
    resolve_token_router_runtime = get_model

    def report_worker_status(self, *args, **kwargs):
        self.callbacks += 1

    receive_heartbeat = report_worker_status

    async def wait_entered(self):
        await self.entered.wait()

    def unblock(self):
        self.release.set()

    def count(self):
        return self.callbacks


class RouterControl(Control):
    def __init__(self, *args):
        super().__init__(*args)
        self.acks = 0
        self.registry = RouterRuntimeRegistry()
        self.registry.register(
            "router", "runtime", {"endpoint": "http://runtime", "acked_revision": 2}
        )

    def get_token_router_config_after(self, router_uid, after_revision):
        return {"revision": 2} if after_revision < 2 else None

    def heartbeat_token_router_instance(self, instance_id, data):
        return self.registry.heartbeat(instance_id, data)

    def ack_token_router_config(self, instance_id, router_uid, revision):
        self.acks += 1
        return self.registry.ack(instance_id, revision)

    def count_acks(self):
        return self.acks

    def resolve_token_router_runtime(self, model_uid):
        return self.registry.get("runtime")["online"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["get_model", "describe_running_model", "resolve_token_router_runtime"]
)
async def test_gateway_drains_callbacks_and_queues_new_management(route):
    pool = await xo.create_actor_pool(address="127.0.0.1", n_process=0)
    try:
        gateway = await xo.create_actor(
            SupervisorGateway, address=pool.external_address
        )
        source = await xo.create_actor(
            Control, "source", gateway, address=pool.external_address
        )
        target = await xo.create_actor(
            Control, "target", gateway, address=pool.external_address
        )
        await gateway.select_supervisor(source)
        await gateway.release_supervisor()
        admitted = asyncio.create_task(gateway.management())
        await source.wait_entered()
        preparing = asyncio.create_task(gateway.prepare_supervisor_switch(final=False))
        await asyncio.sleep(0.01)
        queued = asyncio.create_task(gateway.management())
        assert await getattr(gateway, route)() == "source"
        await source.unblock()
        assert await asyncio.wait_for(admitted, 2) == "source"
        await asyncio.wait_for(preparing, 2)
        assert not queued.done()
        # Background maintenance can still finish its callbacks after the
        # management RPC has drained, before the final snapshot is frozen.
        await asyncio.wait_for(gateway.report_worker_status(), 2)
        await gateway.prepare_supervisor_switch()
        assert await source.count() == 2
        await gateway.select_supervisor(target)
        assert await getattr(gateway, route)() == "target"
        await target.unblock()
        await gateway.release_supervisor()
        assert await asyncio.wait_for(queued, 2) == "target"
        assert await source.count() == 2
        assert await target.count() == 1
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_worker_liveness_never_triggers_reconnect_during_freeze(monkeypatch):
    monkeypatch.setattr("xinference.core.worker.XINFERENCE_TCP_REQUEST_TIMEOUT", 0.1)
    pool = await xo.create_actor_pool(address="127.0.0.1:0", n_process=0)
    try:
        gateway = await xo.create_actor(
            SupervisorGateway, address=pool.external_address
        )
        source = await xo.create_actor(
            Control, "source", gateway, address=pool.external_address
        )
        await gateway.select_supervisor(source)
        await gateway.release_supervisor()
        worker = SimpleNamespace(
            _get_supervisor_ref_with_generation=AsyncMock(return_value=(gateway, 1)),
            _ownership_metadata=lambda method: {},
            _clear_supervisor_refs=MagicMock(),
        )
        for final in (False, True):
            await gateway.prepare_supervisor_switch(final=final)
            for method in ("receive_heartbeat", "report_worker_status"):
                await WorkerActor._call_supervisor_on_actor_loop(
                    worker,
                    method,
                    "worker",
                    add_worker=method == "report_worker_status",
                )
        worker._clear_supervisor_refs.assert_not_called()
        assert await source.count() == 4
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_router_heartbeats_remain_live_through_long_drain_and_freeze(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("xinference.core.router_registry.time.time", lambda: now[0])
    pool = await xo.create_actor_pool(address="127.0.0.1:0", n_process=0)
    try:
        gateway = await xo.create_actor(
            SupervisorGateway, address=pool.external_address
        )
        source = await xo.create_actor(
            RouterControl, "source", gateway, address=pool.external_address
        )
        await gateway.select_supervisor(source)
        await gateway.release_supervisor()
        admitted = asyncio.create_task(gateway.management())
        await source.wait_entered()
        closing = asyncio.create_task(gateway.prepare_supervisor_switch(final=False))
        await asyncio.sleep(0.01)
        revision = 1

        async def runtime_cycle():
            nonlocal revision
            now[0] += 10
            config = await asyncio.wait_for(
                gateway.get_token_router_config_after("router", revision), 1
            )
            if config:
                await asyncio.wait_for(
                    gateway.ack_token_router_config(
                        "runtime", "router", config["revision"]
                    ),
                    1,
                )
                revision = config["revision"]
            await asyncio.wait_for(
                gateway.heartbeat_token_router_instance("runtime", {"status": "ready"}),
                1,
            )
            assert await gateway.resolve_token_router_runtime("virtual") is True

        # A 100-second sweep exceeds the default 90-second runtime deadline.
        for _ in range(10):
            await runtime_cycle()
        await source.unblock()
        await admitted
        await closing
        await gateway.prepare_supervisor_switch()
        for _ in range(10):
            await runtime_cycle()
        late_ack = asyncio.create_task(
            gateway.ack_token_router_config("runtime", "router", 2)
        )
        await asyncio.sleep(0.01)
        assert not late_ack.done()
        target = await xo.create_actor(
            RouterControl, "target", gateway, address=pool.external_address
        )
        await gateway.select_supervisor(target)
        await gateway.release_supervisor()
        await asyncio.wait_for(late_ack, 1)
        assert await source.count_acks() == 1
        assert await target.count_acks() == 1
    finally:
        await pool.stop()


async def completed(host, operation):
    await asyncio.wait_for(
        asyncio.shield(host.operation_task), 180 if sys.platform == "win32" else 60
    )
    return host.operations[operation["operation_id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback", [False, True])
async def test_terminal_replica_death_survives_frozen_callback_timeout(
    tmp_path, monkeypatch, rollback, unused_tcp_port_factory
):
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "")
    monkeypatch.setenv("XINFERENCE_DISABLE_HEALTH_CHECK", "1")
    address = f"127.0.0.1:{unused_tcp_port_factory()}"
    host = SupervisorHost(
        {"supervisor_address": address},
        tmp_path,
        SystemSettingsStore(str(tmp_path / "settings.json")),
    )
    pools = []
    try:
        workers, models = [], []
        for index in range(2):
            pool = await xo.create_actor_pool(address="127.0.0.1", n_process=0)
            pools.append(pool)
            model = await xo.create_actor(Model, address=pool.external_address)
            models.append(model)
            workers.append(
                await xo.create_actor(
                    RestoreWorker,
                    model,
                    f"model-rep{index}",
                    address=pool.external_address,
                    uid="worker",
                )
            )
        await host.start(dict(os.environ))
        # Use the RPC envelope/cancellation path rather than LocalActorRef.
        supervisor = xo.ActorRef(address, "supervisor")
        for index, worker in enumerate(workers):
            await supervisor.add_worker(
                worker.address,
                replica_states=[
                    {
                        "replica_model_uid": f"model-rep{index}",
                        "n_worker": 1,
                        "shard": 0,
                    }
                ],
                incarnation=await worker.worker_incarnation(),
            )
        source = host.active
        await workers[1].block_inventory()
        operation = host.create()
        await asyncio.wait_for(
            asyncio.gather(*(worker.wait_inspected() for worker in workers)),
            180 if sys.platform == "win32" else 15,
        )
        # A has already been inspected; B holds restoration in the frozen phase.
        await workers[0].stop_model()
        await xo.destroy_actor(models[0])
        notifier = SimpleNamespace(
            get_supervisor_ref=AsyncMock(return_value=supervisor)
        )
        await asyncio.wait_for(
            WorkerActor._evict_replica_from_supervisor(notifier, "model-rep0"), 7
        )
        assert not host.operation_task.done()
        # A post-snapshot notification must not mutate only the old owner.
        info = await source.ref.get_instance_info(model_name=None, model_uid="model")
        assert info[0]["replica"] == 2
        await workers[1].finish_inventory(not rollback)
        result = await completed(host, operation)
        assert result["status"] == ("failed" if rollback else "succeeded"), result
        assert (host.active is source) is rollback

        async def wait_evicted():
            while True:
                info = await supervisor.get_instance_info(
                    model_name=None, model_uid="model"
                )
                if info[0]["replica"] == 1:
                    return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_evicted(), 2)
        for _ in range(4):
            assert await supervisor.get_model("model") == models[1]
    finally:
        await host.stop()
        for pool in pools:
            await pool.stop()


@pytest.mark.asyncio
async def test_supervisor_process_replacement_retains_routes(
    tmp_path, monkeypatch, unused_tcp_port_factory
):
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "")
    monkeypatch.setenv("XINFERENCE_DISABLE_HEALTH_CHECK", "1")
    address = f"127.0.0.1:{unused_tcp_port_factory()}"
    host = SupervisorHost(
        {"supervisor_address": address},
        tmp_path,
        SystemSettingsStore(str(tmp_path / "settings.json")),
    )
    worker_pool = await xo.create_actor_pool(address="127.0.0.1:0", n_process=0)
    try:
        model = await xo.create_actor(Model, address=worker_pool.external_address)
        worker = await xo.create_actor(
            Worker, model, address=worker_pool.external_address, uid="worker"
        )
        await host.start(dict(os.environ))
        supervisor = await xo.actor_ref(address=address, uid="supervisor")
        await supervisor.add_worker(
            worker.address,
            replica_states=[
                {"replica_model_uid": "model-rep0", "n_worker": 1, "shard": 0}
            ],
            incarnation=await worker.worker_incarnation(),
        )
        source = host.active
        original = await supervisor.get_model("model")
        assert await original.identity() == os.getpid()
        seen = []
        stopped = asyncio.Event()

        async def inference():
            while not stopped.is_set():
                current = await supervisor.get_model("model")
                seen.append(await current.identity())
                await asyncio.sleep(0.005)

        requests = asyncio.create_task(inference())
        try:
            operation = host.create(request_id="first")
            assert (
                host.create(request_id="first")["operation_id"]
                == operation["operation_id"]
            )
            with pytest.raises(RuntimeError, match="different Supervisor relaunch"):
                host.create(str(tmp_path / "different-python"), request_id="first")
            with pytest.raises(RuntimeError, match="already in progress"):
                host.create(request_id="concurrent")
            result = await completed(host, operation)
            assert result["status"] == "succeeded", json.dumps(result, indent=2)
            assert result["source_pid"] != result["target_pid"]
            assert source.process.poll() is not None
            assert await (await supervisor.get_model("model")).identity() == os.getpid()
            assert await supervisor.get_supervisor_info() == {"supervisor_ip": address}
            assert (
                await supervisor.get_instance_info(model_name=None, model_uid="model")
            )[0]["replica"] == 1
            result = await completed(
                host,
                host.create(
                    os.getenv("XINFERENCE_TEST_SUPERVISOR_PYTHON"), request_id="second"
                ),
            )
            assert result["status"] == "succeeded", json.dumps(result, indent=2)
            active = host.active
            result = await completed(
                host,
                host.create(str(tmp_path / "missing-python"), request_id="bad-python"),
            )
            assert result["status"] == "failed"
            assert host.active is active
            await worker.set_inventory_available(False)
            result = await completed(host, host.create(request_id="bad-inventory"))
            assert result["status"] == "failed", result
            assert host.active is active
            assert await (await supervisor.get_model("model")).identity() == os.getpid()
        finally:
            stopped.set()
            await requests
        assert len(seen) > 10
        assert set(seen) == {os.getpid()}
    finally:
        await host.stop()
        await worker_pool.stop()
