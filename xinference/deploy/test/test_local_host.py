# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest
import requests
from xoscar.utils import get_next_port

EXTENSION = """
import os
from pathlib import Path
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio

class Extension:
    name = "local-host-test"
    api_version = 1
    def worker_metadata(self): return {}
    def configure_api(self, context):
        async def info() -> JSONResponse:
            supervisor = await context.get_supervisor_ref()
            return JSONResponse({
                "api_pid": os.getpid(),
                "supervisor": await supervisor.supervisor_upgrade_info(),
                "local": await supervisor.is_local_deployment(),
                "workers": await supervisor.get_workers_info(),
            })
        async def stream() -> EventSourceResponse:
            async def chunks():
                yield "start"
                while not Path(os.environ["LOCAL_TEST_RELEASE"]).exists():
                    await asyncio.sleep(.01)
                yield "finish"
            return EventSourceResponse(chunks())
        context.router.add_api_route("/local-test/info", info)
        context.router.add_api_route("/local-test/stream", stream)

def create(): return Extension()
"""


def until(function, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = function()
            if value:
                return value
        except (requests.RequestException, KeyError):
            pass
        time.sleep(0.05)
    raise TimeoutError("Local service did not reach the expected state")


def alive(pid):
    return (
        psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    )


@pytest.mark.skipif(os.name != "posix", reason="Local service hosting requires POSIX")
@pytest.mark.parametrize("first", ["worker", "supervisor"])
def test_local_components_migrate_on_demand_and_keep_local_semantics(tmp_path, first):
    ports = [get_next_port() for _ in range(3)]
    configuration = {
        "host": "127.0.0.1",
        "port": ports[0],
        "supervisor_address": f"127.0.0.1:{ports[1]}",
        "worker_address": f"127.0.0.1:{ports[2]}",
        "local_deployment": True,
        "api_role": "local",
    }
    (tmp_path / "local_extension.py").write_text(EXTENSION)
    script = tmp_path / "start.py"
    script.write_text(
        "from xinference.deploy.api_host import run\n"
        "from xinference.deploy.local_host import LocalHost\n"
        "if __name__ == '__main__':\n"
        f"    run({configuration!r}, host_class=LocalHost)\n"
    )
    source_root = str(Path(__file__).resolve().parents[3])
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(tmp_path), source_root, os.environ.get("PYTHONPATH", "")]
        ),
        "XINFERENCE_HOME": str(tmp_path / "home"),
        "XINFERENCE_AUTH_ADVANCED": "0",
        "XINFERENCE_DISABLE_METRICS": "1",
        "XINFERENCE_EXTENSIONS": "local_extension:create",
        "LOCAL_TEST_RELEASE": str(tmp_path / "release"),
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    endpoint = f"http://127.0.0.1:{ports[0]}"
    session = requests.Session()
    session.trust_env = False
    seen = set()
    operations = {}
    failures = []
    stop_polling = threading.Event()
    stream_started = threading.Event()

    def get_info():
        response = session.get(endpoint + "/local-test/info", timeout=5)
        response.raise_for_status()
        return response.json()

    def poll():
        client = requests.Session()
        client.trust_env = False
        try:
            while not stop_polling.is_set():
                response = client.get(endpoint + "/status", timeout=10)
                if response.status_code != 200:
                    failures.append(response.status_code)
                stop_polling.wait(0.03)
        except requests.RequestException as error:
            failures.append(str(error))
        finally:
            client.close()

    def stream():
        client = requests.Session()
        client.trust_env = False
        try:
            with client.get(
                endpoint + "/local-test/stream", stream=True, timeout=90
            ) as response:
                response.raise_for_status()
                chunks = []
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: "):
                        chunks.append(line[6:].decode())
                        stream_started.set()
                return chunks
        finally:
            client.close()

    def submit(component, request_id, python=None):
        address = get_info()["workers"][0]["work-ip"]
        path = (
            f"/v1/workers/{address}/relaunch"
            if component == "worker"
            else f"/v1/{component}/relaunch"
        )
        response = session.post(
            endpoint + path,
            json={"request_id": request_id, "python_executable": python},
            timeout=10,
        )
        response.raise_for_status()
        assert response.status_code == 202
        return response.json()

    def status(component, operation):
        response = session.get(
            endpoint + f"/v1/{component}-relaunches/" + operation["operation_id"],
            timeout=5,
        )
        response.raise_for_status()
        return response.json()

    def finish(component, operation):
        result = until(
            lambda: (
                value
                if (value := status(component, operation))["status"]
                in {"succeeded", "failed"}
                else None
            )
        )
        assert result["status"] == "succeeded", result
        if result.get("target_pid"):
            seen.add(result["target_pid"])
        return result

    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            info = until(get_info)
            initial_pid = info["supervisor"]["pid"]
            seen.add(initial_pid)
            assert info["api_pid"] == process.pid
            assert info["local"] is True
            children = psutil.Process(process.pid).children(recursive=True)
            control_pids = {
                child.pid
                for child in children
                if "multiprocessing.resource_tracker" not in " ".join(child.cmdline())
            }
            assert control_pids == {initial_pid}
            executor.submit(poll)
            stream_future = executor.submit(stream)
            assert stream_started.wait(10)

            for component in ["api", "supervisor", "worker"]:
                bad = submit(component, "bad-" + component, "/does/not/exist/python")
                failed = until(
                    lambda: (
                        value
                        if (value := status(component, bad))["status"] == "failed"
                        else None
                    )
                )
                assert failed.get("error")
                assert get_info()["supervisor"]["pid"] == initial_pid
                assert not stream_future.done()

            # A model registration exercises the shared module registry before
            # migration and remote synchronization after roles separate.
            response = session.get(
                endpoint + "/v1/model_registrations/LLM/qwen2.5-instruct", timeout=10
            )
            response.raise_for_status()
            custom = response.json()
            custom["model_family"] = custom["model_name"]
            custom["model_name"] = "local-upgrade-regression"

            def registration_cycle():
                path = "/v1/model_registrations/LLM"
                response = session.post(
                    endpoint + path,
                    json={"model": json.dumps(custom), "persist": False},
                    timeout=10,
                )
                assert response.status_code == 200, response.text
                response = session.get(
                    endpoint + path + "/" + custom["model_name"], timeout=10
                )
                assert response.status_code == 200, response.text
                response = session.delete(
                    endpoint + path + "/" + custom["model_name"], timeout=10
                )
                assert response.status_code == 200, response.text
                assert (
                    session.get(
                        endpoint + path + "/" + custom["model_name"], timeout=10
                    ).status_code
                    == 400
                )

            registration_cycle()
            second = "supervisor" if first == "worker" else "worker"
            for component in [first, second]:
                operation = finish(component, submit(component, "first-" + component))
                operations[component] = operation
                assert get_info()["local"] is True
                assert alive(initial_pid)
                assert not stream_future.done()
                registration_cycle()

            operation = submit("api", "first-api")
            until(lambda: status("api", operation)["status"] == "draining")
            assert get_info()["api_pid"] != process.pid
            assert not stream_future.done()
            (tmp_path / "release").touch()
            assert stream_future.result(timeout=15) == ["start", "finish"]
            finish("api", operation)

            for component in [second, first, "api"]:
                finish(component, submit(component, "second-" + component))
                assert get_info()["local"] is True
            registration_cycle()
            assert alive(initial_pid)
            assert process.poll() is None
            until(
                lambda: all(
                    child.status() != psutil.STATUS_ZOMBIE
                    for child in psutil.Process(process.pid).children(recursive=True)
                )
            )
        finally:
            stop_polling.set()
            (tmp_path / "release").touch()
            session.close()
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            executor.shutdown(wait=True)
        assert process.returncode == 0, (tmp_path / "service.log").read_text()[-5000:]
        assert not failures
        until(lambda: not any(alive(pid) for pid in seen), timeout=10)
