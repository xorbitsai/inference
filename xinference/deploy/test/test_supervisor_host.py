# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import os
import uuid

import pytest
import xoscar as xo
from xoscar.utils import get_next_port

from ...core.supervisor_upgrade import SupervisorGateway
from ...core.system_settings_store import SystemSettingsStore
from ..supervisor_host import SupervisorHost

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process hosting")


class Model(xo.StatelessActor):
    def identity(self):
        return os.getpid()


class Worker(xo.StatelessActor):
    def __init__(self, model):
        super().__init__()
        self.model = model
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
        return {"model-rep0": {}}

    def get_model(self, model_uid):
        assert model_uid == "model-rep0"
        return self.model

    def set_inventory_available(self, available):
        self.inventory_available = available


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

    def report_worker_status(self):
        self.callbacks += 1

    async def wait_entered(self):
        await self.entered.wait()

    def unblock(self):
        self.release.set()

    def count(self):
        return self.callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route", ["get_model", "describe_running_model", "resolve_token_router_runtime"]
)
async def test_gateway_drains_callbacks_and_queues_new_management(route):
    pool = await xo.create_actor_pool(
        address=f"127.0.0.1:{get_next_port()}", n_process=0
    )
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


async def completed(host, operation):
    await asyncio.wait_for(asyncio.shield(host.operation_task), 60)
    return host.operations[operation["operation_id"]]


@pytest.mark.asyncio
async def test_supervisor_process_replacement_retains_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XINFERENCE_EXTENSIONS", "")
    monkeypatch.setenv("XINFERENCE_DISABLE_HEALTH_CHECK", "1")
    address = f"127.0.0.1:{get_next_port()}"
    host = SupervisorHost(
        {"supervisor_address": address},
        tmp_path,
        SystemSettingsStore(str(tmp_path / "settings.json")),
    )
    worker_pool = await xo.create_actor_pool(
        address=f"127.0.0.1:{get_next_port()}", n_process=0
    )
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
                host.create("/different/python", request_id="first")
            with pytest.raises(RuntimeError, match="already in progress"):
                host.create(request_id="concurrent")
            result = await completed(host, operation)
            assert result["status"] == "succeeded", result
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
            assert result["status"] == "succeeded", result
            active = host.active
            result = await completed(
                host, host.create("/missing/python", request_id="bad-python")
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
