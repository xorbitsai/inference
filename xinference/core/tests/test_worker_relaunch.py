# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ..worker_registry import WorkerOwnership
from ..worker_relaunch import WorkerRelaunchManager, validate_relaunch_options


def manager_at(tmp_path, monkeypatch):
    worker = SimpleNamespace(
        worker_incarnation=AsyncMock(return_value="source-generation"),
        worker_handoff_protocol=AsyncMock(return_value=1),
    )
    registry = WorkerOwnership()
    supervisor = SimpleNamespace(
        _ownership_registry=lambda: registry, _worker_address_to_worker={"old": worker}
    )
    manager = WorkerRelaunchManager(supervisor, tmp_path / "jobs.json")
    monkeypatch.setattr(manager, "resume", lambda: None)
    return manager


@pytest.mark.asyncio
async def test_relaunch_deduplicates_and_preserves_terminal_status(
    tmp_path, monkeypatch
):
    manager = manager_at(tmp_path, monkeypatch)
    first = await manager.create("old", request_id="click")
    assert await manager.create("old", request_id="click") == first
    assert await manager.create("old") == first
    with pytest.raises(RuntimeError, match="in progress"):
        await manager.create("old", "/another/python")
    manager.plan(first["operation_id"], "old", "source-generation", "new")
    manager.update(first["operation_id"], status="succeeded")
    manager.supervisor._worker_address_to_worker.clear()
    assert (await manager.create("old"))["operation_id"] == first["operation_id"]
    reloaded = WorkerRelaunchManager(manager.supervisor, manager.path)
    assert reloaded.records[first["operation_id"]]["status"] == "succeeded"
    assert (
        reloaded.update(first["operation_id"], status="failed")["status"] == "succeeded"
    )


@pytest.mark.asyncio
async def test_request_id_cannot_be_reused_for_another_target(tmp_path, monkeypatch):
    manager = manager_at(tmp_path, monkeypatch)
    await manager.create("old", request_id="click")
    with pytest.raises(RuntimeError, match="request_id"):
        await manager.create("other", request_id="click")


@pytest.mark.asyncio
async def test_older_worker_is_rejected_before_creating_job(tmp_path, monkeypatch):
    manager = manager_at(tmp_path, monkeypatch)
    manager.supervisor._worker_address_to_worker[
        "old"
    ].worker_handoff_protocol.side_effect = AttributeError("not supported")
    del manager.supervisor._worker_address_to_worker["old"].worker_incarnation
    with pytest.raises(ValueError, match="does not support"):
        await manager.create("old")
    assert not manager.records


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["worker_handoff_protocol", "worker_incarnation"])
async def test_worker_probe_timeout_is_a_clear_client_error(
    tmp_path, monkeypatch, method
):
    manager = manager_at(tmp_path, monkeypatch)
    getattr(manager.supervisor._worker_address_to_worker["old"], method).side_effect = (
        asyncio.TimeoutError
    )
    with pytest.raises(ValueError, match="Timed out"):
        await manager.create("old")
    assert not manager.records


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [True, False])
async def test_definitely_unlaunched_operation_fails(tmp_path, monkeypatch, missing):
    import xoscar as xo

    manager = manager_at(tmp_path, monkeypatch)
    record = await manager.create("old")
    manager._worker = (
        AsyncMock(side_effect=xo.ActorNotExist("old"))
        if missing
        else AsyncMock(
            return_value=SimpleNamespace(
                worker_incarnation=AsyncMock(return_value="restarted")
            )
        )
    )
    await asyncio.wait_for(manager._monitor(record["operation_id"]), 2)
    assert manager.get(record["operation_id"])["status"] == "failed"
    with pytest.raises(RuntimeError, match="finished"):
        manager.plan(record["operation_id"], "old", "source-generation", "too-late")


@pytest.mark.asyncio
async def test_reads_never_resume_operations(tmp_path, monkeypatch):
    from unittest.mock import Mock

    manager = manager_at(tmp_path, monkeypatch)
    record = await manager.create("old")
    manager.resume = Mock()
    assert manager.get(record["operation_id"])
    assert manager.list()
    manager.resume.assert_not_called()


@pytest.mark.asyncio
async def test_manager_creation_resumes_persisted_operations(tmp_path, monkeypatch):
    from ... import constants
    from ..worker_relaunch import SupervisorRelaunchMixin

    manager = manager_at(tmp_path, monkeypatch)
    record = await manager.create("old")
    manager.path.rename(tmp_path / "worker-relaunches.json")
    monkeypatch.setattr(constants, "XINFERENCE_HOME", str(tmp_path))
    resumed = asyncio.Event()

    async def monitor(self, operation_id):
        assert operation_id == record["operation_id"]
        resumed.set()

    monkeypatch.setattr(WorkerRelaunchManager, "_monitor", monitor)
    restored = SupervisorRelaunchMixin._relaunch_manager(SimpleNamespace())
    await asyncio.wait_for(resumed.wait(), 2)
    await restored.close()


@pytest.mark.parametrize(
    "payload",
    ["{", '{"version":2,"operations":{}}', '{"version":1,"operations":{"bad":{}}}'],
)
def test_invalid_journal_is_reported_without_replacing_it(tmp_path, payload):
    from ..worker_state import WorkerJournalError

    path = tmp_path / "journal.json"
    path.write_text(payload)
    with pytest.raises(WorkerJournalError, match="journal"):
        WorkerRelaunchManager(None, path)
    assert path.read_text() == payload


def test_replacement_persists_to_original_service_recovery_path(tmp_path, monkeypatch):
    from .. import worker as module

    monkeypatch.setattr(module, "XINFERENCE_HOME", str(tmp_path))
    fresh = SimpleNamespace(_recovery_address=None, address="host:1000")
    assert module.WorkerActor._get_recovery_file_path(fresh).endswith(
        "host_1000/models.json"
    )
    worker = module.WorkerActor.__new__(module.WorkerActor)
    worker._recovery_address = "host:1000"
    worker._model_uid_to_launch_args = {
        "after-upgrade-rep0": {"model_uid": "after-upgrade-rep0"}
    }
    worker._persist_launch_args()
    assert worker._load_persisted_launch_args() == worker._model_uid_to_launch_args
    assert (
        json.loads((tmp_path / "worker_recovery/host_1000/models.json").read_text())
        == worker._model_uid_to_launch_args
    )


@pytest.mark.asyncio
async def test_initial_status_report_failure_does_not_abort_replacement(monkeypatch):
    import xoscar as xo

    from ...deploy import worker as deployment

    pool = SimpleNamespace(stop=AsyncMock(), join=AsyncMock())
    worker = SimpleNamespace(
        worker_incarnation=AsyncMock(return_value="target"),
        replace_worker=AsyncMock(),
        worker_replacement_safe_to_stop=AsyncMock(return_value=True),
    )
    supervisor = SimpleNamespace(
        report_worker_relaunch=AsyncMock(
            side_effect=[ConnectionResetError("first report lost"), None]
        )
    )
    from ...deploy import utils

    monkeypatch.setattr(utils, "create_worker_actor_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(
        deployment, "start_worker_components", AsyncMock(return_value=worker)
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=supervisor))
    await deployment._start_worker(
        "new", "supervisor", replace_worker="old", relaunch_operation_id="one"
    )
    worker.replace_worker.assert_awaited_once_with("old")
    pool.stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_journal_failure_does_not_publish_target(tmp_path, monkeypatch):
    manager = manager_at(tmp_path, monkeypatch)
    first = await manager.create("old")
    monkeypatch.setattr(
        os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        manager.plan(first["operation_id"], "old", "source-generation", "new")
    assert manager.records[first["operation_id"]]["target_worker"] is None
    assert (
        json.loads(manager.path.read_text())["operations"][first["operation_id"]][
            "target_worker"
        ]
        is None
    )


@pytest.mark.asyncio
async def test_lost_connection_remains_pending_after_restart(tmp_path, monkeypatch):
    manager = manager_at(tmp_path, monkeypatch)
    first = await manager.create("old")
    manager._worker = AsyncMock(side_effect=ConnectionResetError("unknown outcome"))
    task = asyncio.create_task(manager._monitor(first["operation_id"]))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    reloaded = WorkerRelaunchManager(manager.supervisor, manager.path)
    assert reloaded.records[first["operation_id"]]["status"] == "pending"


@pytest.mark.asyncio
async def test_restarted_supervisor_reconciles_a_completed_target(
    tmp_path, monkeypatch
):
    manager = manager_at(tmp_path, monkeypatch)
    first = await manager.create("old")
    operation_id = first["operation_id"]
    manager.plan(operation_id, "old", "source-generation", "new")
    reloaded = WorkerRelaunchManager(manager.supervisor, manager.path)
    target = SimpleNamespace(
        worker_incarnation=AsyncMock(return_value="new-generation"),
        worker_relaunch_state=AsyncMock(
            return_value={
                "status": "succeeded",
                "incarnation": "new-generation",
                "pid": 123,
            }
        ),
    )
    reloaded._worker = AsyncMock(return_value=target)
    await asyncio.wait_for(reloaded._monitor(operation_id), 2)
    assert reloaded.records[operation_id]["status"] == "succeeded"
    assert reloaded.records[operation_id]["target_pid"] == 123
    assert all(call.args == ("new",) for call in reloaded._worker.call_args_list)


@pytest.mark.asyncio
async def test_reused_worker_address_gets_a_new_operation(tmp_path, monkeypatch):
    manager = manager_at(tmp_path, monkeypatch)
    first = await manager.create("old")
    manager.supervisor._worker_address_to_worker[
        "old"
    ].worker_incarnation.return_value = "next-generation"
    second = await manager.create("old")
    assert second["operation_id"] != first["operation_id"]
    assert second["source_incarnation"] == "next-generation"


@pytest.mark.parametrize(
    "options",
    [
        dict(python_executable="python"),
        dict(python_executable=1),
        dict(request_id=""),
        dict(request_id=1),
        dict(request_id="x" * 129),
    ],
)
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        validate_relaunch_options(**options)


@pytest.mark.asyncio
async def test_pid_one_worker_cannot_end_its_container_during_relaunch(monkeypatch):
    from ..worker_handoff import WorkerHandoffMixin

    if sys.platform == "win32":
        pytest.skip("Worker handoff requires POSIX")
    monkeypatch.setattr(os, "getpid", lambda: 1)
    with pytest.raises(ValueError, match="container supervisor"):
        await WorkerHandoffMixin.check_worker_handoff_support(SimpleNamespace())


_PROCESS_TEST = r"""
import asyncio, contextlib, json, os, subprocess, sys, time
from pathlib import Path
import psutil
import xoscar as xo
from xoscar.backends.allocate_strategy import RandomSubPool
from xinference.core.worker import WorkerActor, ModelStatus
from xinference.core.supervisor import SupervisorActor
from xinference.core.tests.test_worker_handoff import CounterModel
from xinference.api.routers import admin
from xinference.client.restful.restful_client import Client
from xinference.client.restful.async_restful_client import AsyncClient
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import FastAPI
import httpx, uvicorn

class SourceWorker(WorkerActor):
    async def _commit_worker_handoff(self,ticket,timeout):
        result=await super()._commit_worker_handoff(ticket,timeout)
        if os.environ['TEST_EXIT_DURING_COMMIT']=='1': os._exit(0)
        return result

async def source(supervisor):
    pool=await xo.create_actor_pool('127.0.0.1',n_process=0,auto_recover='process')
    worker=await xo.create_actor(SourceWorker,uid=WorkerActor.default_uid(),address=pool.external_address,
        supervisor_address=supervisor,supervisor_endpoint=None,main_pool=pool,gpu_devices=[],
        metrics_exporter_host='127.0.0.1',metrics_exporter_port=int(os.environ['TEST_METRICS_PORT']))
    await pool.append_sub_pool()
    model=await xo.create_actor(CounterModel,worker.address,uid='counter-rep0',address=worker.address,allocate_strategy=RandomSubPool())
    owner=pool._actors[worker.uid]
    owner._model_uid_to_model['counter-rep0']=model
    owner._model_uid_to_model_spec['counter-rep0']={'model_name':'counter','model_ability':['generate']}
    owner._model_uid_to_model_status['counter-rep0']=ModelStatus(model_state='ready')
    owner._model_uid_to_launch_args['counter-rep0']={'model_uid':'counter-rep0'}
    owner._model_uid_to_addr['counter-rep0']=model.address
    sup=await xo.actor_ref(address=supervisor,uid=SupervisorActor.default_uid())
    await sup.add_worker(worker.address,replica_states=owner._get_running_replica_states(),incarnation=owner._incarnation)
    before=await model.add(8)
    print(json.dumps(dict(worker=worker.address,model=model.address,before=before)),flush=True)
    await pool.join()

async def driver():
    pool=await xo.create_actor_pool('127.0.0.1',n_process=0)
    sup=await xo.create_actor(SupervisorActor,address=pool.external_address,uid=SupervisorActor.default_uid())
    async def supervisor_ref(): return sup
    app=FastAPI()
    app.state.api=SimpleNamespace(_get_supervisor_ref=supervisor_ref)
    app.add_api_route('/v1/workers/{worker_address}/relaunch',admin.relaunch_worker,methods=['POST'])
    app.add_api_route('/v1/worker-relaunches/{operation_id}',admin.get_worker_relaunch,methods=['GET'])
    app.add_api_route('/v1/worker-relaunches',admin.list_worker_relaunches,methods=['GET'])
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=0,log_level='warning',lifespan='off'))
    server_task=asyncio.create_task(server.serve())
    while not server.started: await asyncio.sleep(.01)
    endpoint='http://127.0.0.1:'+str(server.servers[0].sockets[0].getsockname()[1])
    client=await asyncio.to_thread(Client,endpoint)
    with patch.object(AsyncClient,'_check_cluster_authenticated',lambda self:None):
        async_client=AsyncClient(endpoint)
    source_log=open('source.log','w')
    process=await asyncio.create_subprocess_exec(sys.executable,__file__,'source',pool.external_address,
        stdout=asyncio.subprocess.PIPE,stderr=source_log,start_new_session=True)
    owned={}
    target_pid=None
    try:
        raw=await asyncio.wait_for(process.stdout.readline(),45)
        initial=json.loads(raw)
        for p in [psutil.Process(process.pid),*psutil.Process(process.pid).children(recursive=True)]:
            owned[p.pid]=p.create_time()
        bad=await asyncio.to_thread(client.relaunch_worker,initial['worker'],'/does/not/exist/python','bad')
        async def done(operation):
            nonlocal target_pid
            for _ in range(300):
                state=await async_client.get_worker_relaunch(operation['operation_id'])
                if state.get('target_pid'):
                    target_pid=state['target_pid']
                    with contextlib.suppress(psutil.NoSuchProcess): owned[target_pid]=psutil.Process(target_pid).create_time()
                if state['status'] in ('failed','succeeded'): return state
                await asyncio.sleep(.1)
            raise AssertionError(state)
        bad=await done(bad)
        assert bad['status']=='failed',bad
        job=await asyncio.to_thread(client.relaunch_worker,initial['worker'],None,'one')
        duplicate=await async_client.relaunch_worker(initial['worker'],request_id='one')
        assert duplicate['operation_id']==job['operation_id']
        result=await done(job)
        target_pid=result.get('target_pid')
        if target_pid: owned[target_pid]=psutil.Process(target_pid).create_time()
        assert result['status']=='succeeded',result
        # Retained model processes can keep the source's stdout pipe open.
        # returncode follows process exit without waiting for inherited pipe EOF.
        for _ in range(200):
            if process.returncode is not None: break
            await asyncio.sleep(.1)
        assert process.returncode==0,process.returncode
        model=await xo.actor_ref(address=initial['model'],uid='counter-rep0')
        after=await model.add(1)
        assert tuple(after)==(9,initial['before'][1],result['target_worker']),after
        assert (await asyncio.to_thread(client.get_worker_relaunch,result['operation_id']))['status']=='succeeded'
        assert any(j['operation_id']==job['operation_id'] for j in await async_client.list_worker_relaunches())
        assert any(j['operation_id']==job['operation_id'] for j in await asyncio.to_thread(client.list_worker_relaunches))
        retry=await async_client.relaunch_worker(initial['worker'],request_id='one')
        assert retry['operation_id']==result['operation_id']
        if os.environ['XINFERENCE_DISABLE_METRICS']=='0':
            async with httpx.AsyncClient() as http:
                metrics=await http.get('http://127.0.0.1:'+os.environ['TEST_METRICS_PORT']+'/metrics')
                assert metrics.status_code==200,metrics.text
        print(json.dumps(dict(before=initial['before'],after=after,status=result)),flush=True)
    finally:
        if process.returncode is None:
            for p in psutil.Process(process.pid).children(recursive=True): owned[p.pid]=p.create_time()
            owned[process.pid]=psutil.Process(process.pid).create_time()
        if target_pid and psutil.pid_exists(target_pid):
            for p in psutil.Process(target_pid).children(recursive=True): owned[p.pid]=p.create_time()
        for pid,created in reversed(list(owned.items())):
            with contextlib.suppress(psutil.NoSuchProcess):
                p=psutil.Process(pid)
                if p.uids().real==os.getuid() and p.create_time()==created: p.kill()
        with contextlib.suppress(Exception): await asyncio.wait_for(process.wait(),5)
        with contextlib.suppress(Exception): await asyncio.wait_for(process.stdout.read(),5)
        await xo.destroy_actor(sup)
        await pool.stop()
        source_log.close()
        await async_client.close()
        client.close()
        server.should_exit=True
        await server_task

if __name__=='__main__':
    asyncio.run(source(sys.argv[2]) if len(sys.argv)>1 else driver())
"""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics_enabled,source_exit",
    [(False, False), (True, False), (False, True)],
    ids=["normal", "fixed-metrics", "source-exits-before-ack"],
)
async def test_relaunch_starts_replacement_and_status_outlives_source(
    tmp_path, metrics_enabled, source_exit
):
    from xoscar.utils import get_next_port

    if sys.platform == "win32":
        pytest.skip("Worker handoff requires POSIX")
    script = tmp_path / "process_test.py"
    script.write_text(_PROCESS_TEST)
    env = dict(
        os.environ,
        XINFERENCE_HOME=str(tmp_path / "home"),
        XINFERENCE_DISABLE_METRICS="0" if metrics_enabled else "1",
        TEST_METRICS_PORT=str(get_next_port()),
        TEST_EXIT_DURING_COMMIT="1" if source_exit else "0",
        XINFERENCE_DISABLE_HEALTH_CHECK="0",
        XINFERENCE_TOKEN_ROUTER_ENABLED="0",
        CUDA_VISIBLE_DEVICES="-1",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        cwd=tmp_path,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 100)
        assert process.returncode == 0, (
            output.decode() + (tmp_path / "source.log").read_text()
        )
        record = next(
            json.loads(line)
            for line in output.decode().splitlines()
            if line.startswith('{"before":')
        )
        assert record["status"]["status"] == "succeeded"
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
