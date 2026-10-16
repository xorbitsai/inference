# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Exercise the real deployment and model-subpool TCP paths without model downloads."""

import asyncio
import multiprocessing
import os
import subprocess
import sys
import time
from contextlib import ExitStack

import psutil
import pytest
import requests
import xoscar as xo

from ...conftest import TEST_LOGGING_CONF, _get_test_port
from ...core.worker import WorkerActor
from .. import local, supervisor
from ..utils import create_worker_actor_pool, health_check


class _ProbeActor(xo.Actor):
    def identity(self):
        return os.getpid(), bool(os.environ.get("XOSCAR_AUTH_KEY"))


class _ProbeWorker(WorkerActor):
    """Use the production spawn/recovery paths with a download-free model actor."""

    async def launch_probe(self, model_uid="auth-probe"):
        address, _ = await self._create_subpool(
            model_uid, n_gpu=None, start_python=sys.executable
        )
        ref = await xo.create_actor(_ProbeActor, uid=model_uid, address=address)
        self._model_uid_to_addr[model_uid] = address
        self._model_uid_to_model[model_uid] = ref
        self._model_uid_to_launch_args[model_uid] = {"model_uid": model_uid}
        return await ref.identity()

    async def probe_identity(self):
        return await self._model_uid_to_model["auth-probe"].identity()

    async def recover_model(self, launch_args):
        await self.launch_probe(**launch_args)

    async def terminate_model(self, model_uid, is_model_die=False):
        self._model_uid_to_model.pop(model_uid, None)


def _run_probe_worker(address, supervisor_address, ready):
    async def run():
        pool = await create_worker_actor_pool(address, TEST_LOGGING_CONF)
        try:
            await xo.create_actor(
                _ProbeWorker,
                address=address,
                uid=WorkerActor.default_uid(),
                supervisor_address=supervisor_address,
                supervisor_endpoint=None,
                main_pool=pool,
                gpu_devices=[],
                metrics_exporter_host=None,
                metrics_exporter_port=None,
            )
            ready.send(True)
            ready.close()
            await pool.join()
        finally:
            await pool.stop()

    asyncio.run(run())


def _stop_process_tree(process):
    # Only reap descendants of this test's own process, including model subpools.
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=5)


@pytest.fixture
def authenticated_cluster(request, monkeypatch, tmp_path):
    from ...api.restful_api import run_in_subprocess as run_api

    mode, keyed = request.param
    if keyed:
        monkeypatch.setenv("XOSCAR_AUTH_KEY", "actor-auth-integration-test-secret")
    else:
        monkeypatch.delenv("XOSCAR_AUTH_KEY", raising=False)
    monkeypatch.setenv("XINFERENCE_HOME", str(tmp_path))
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    monkeypatch.setenv("XINFERENCE_DISABLE_METRICS", "1")
    monkeypatch.setenv("XINFERENCE_DISABLE_HEALTH_CHECK", "1")
    monkeypatch.setenv("XINFERENCE_LIST_MODELS_DEBOUNCE_SECONDS", "0")
    multiprocessing.set_start_method("spawn", force=True)
    address = f"127.0.0.1:{_get_test_port()}"
    with ExitStack() as cleanup:
        if mode == "local":
            process = local.run_in_subprocess(address, logging_conf=TEST_LOGGING_CONF)
            worker_address = address
        else:
            process = supervisor.run_in_subprocess(address, TEST_LOGGING_CONF)
        cleanup.callback(_stop_process_tree, process)
        assert health_check(address, max_attempts=30, sleep_interval=1)
        if mode == "distributed":
            worker_address = f"127.0.0.1:{_get_test_port()}"
            ready, child_ready = multiprocessing.Pipe(duplex=False)
            cleanup.callback(ready.close)
            worker = multiprocessing.Process(
                target=_run_probe_worker, args=(worker_address, address, child_ready)
            )
            worker.start()
            child_ready.close()
            cleanup.callback(_stop_process_tree, worker)
            assert ready.poll(60) and ready.recv() is True

        port = _get_test_port()
        api = run_api(address, "127.0.0.1", port, TEST_LOGGING_CONF)
        cleanup.callback(_stop_process_tree, api)
        endpoint = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 60
        while True:
            try:
                if requests.get(f"{endpoint}/status", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            assert api.is_alive() and time.monotonic() < deadline
            time.sleep(0.1)
        yield endpoint, address, worker_address, mode, keyed


@pytest.mark.parametrize(
    "authenticated_cluster",
    [(mode, keyed) for mode in ("local", "distributed") for keyed in (False, True)],
    indirect=True,
)
async def test_actor_transport_authentication(authenticated_cluster):
    endpoint, address, worker_address, mode, keyed = authenticated_cluster
    # REST -> supervisor -> worker traverses real TCP actor channels.
    response = requests.get(f"{endpoint}/v1/models", timeout=10)
    response.raise_for_status()
    assert response.json()["data"] == []
    worker = await xo.actor_ref(address=worker_address, uid=WorkerActor.default_uid())
    if mode == "distributed":
        pid, configured = await worker.launch_probe()
        assert pid != os.getpid()
        assert configured is keyed

    if keyed:
        # A new client process avoids cached connections and independently
        # exercises both missing-key and wrong-key rejection.
        for bad_key in (None, "different-deployment-secret"):
            env = os.environ.copy()
            env.pop("XOSCAR_AUTH_KEY", None)
            if bad_key is not None:
                env["XOSCAR_AUTH_KEY"] = bad_key
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import asyncio,sys,xoscar as xo\n"
                    "async def check():\n"
                    "    ref = await xo.actor_ref(address=sys.argv[1], uid='supervisor')\n"
                    "    await ref.get_status()\n"
                    "try:\n"
                    "    asyncio.run(asyncio.wait_for(check(), 5))\n"
                    "except (EOFError, ConnectionError, xo.errors.ServerClosed):\n"
                    "    sys.exit(0)\n"
                    "sys.exit(1)\n",
                    address,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0, result.stderr
        # Rejected clients must not break valid API or actor traffic.
        assert requests.get(f"{endpoint}/v1/models", timeout=10).status_code == 200

    if mode == "distributed":
        # The real WorkerActor.recover_sub_pool callback recreates the probe
        # through the same model subpool path, including start_python.
        psutil.Process(pid).kill()
        deadline = time.monotonic() + 60
        while True:
            try:
                recovered_pid, configured = await asyncio.wait_for(
                    worker.probe_identity(), 2
                )
                if recovered_pid != pid:
                    break
            except (
                KeyError,
                EOFError,
                ConnectionError,
                xo.errors.ServerClosed,
                asyncio.TimeoutError,
            ):
                pass
            assert time.monotonic() < deadline, "Model subpool was not recovered"
            await asyncio.sleep(0.2)
        assert configured is keyed
