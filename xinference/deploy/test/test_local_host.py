# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import venv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest
import requests
from xoscar.utils import get_next_port

from ..local_host import LocalHost

EXTENSION = """
import os
import logging
import xoscar as xo
from pathlib import Path
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import asyncio

class Extension:
    name = "local-host-test"
    api_version = 1
    def worker_metadata(self):
        logger = logging.getLogger("xinference.core.worker")
        return {
            "pid": os.getpid(),
            "has_api_host": "XINFERENCE_API_HOST" in os.environ,
            "level": logger.getEffectiveLevel(),
            "formats": [h.formatter._fmt for h in logger.handlers if h.formatter],
        }
    def configure_api(self, context):
        async def info() -> JSONResponse:
            supervisor = await context.get_supervisor_ref()
            workers = await supervisor.get_workers_info()
            metadata = []
            for item in workers:
                worker = await xo.actor_ref(address=item["work-ip"], uid="worker")
                resources = await worker.get_extension_resources()
                metadata.append(resources["extensions"]["local-host-test"])
            return JSONResponse({
                "api_pid": os.getpid(),
                "supervisor": await supervisor.supervisor_upgrade_info(),
                "local": await supervisor.is_local_deployment(),
                "workers": workers,
                "worker_metadata": metadata,
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


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Local service hosting requires POSIX")
async def test_local_runtime_preserves_original_environment(tmp_path, monkeypatch):
    from ...core.utils import build_subpool_envs_for_virtual_env

    monkeypatch.delenv("PYTHONPATH", raising=False)
    engine_env = tmp_path / "engine-env"
    venv.EnvBuilder(with_pip=False).create(engine_env)
    engine_python = str(engine_env / "bin" / "python")
    engine_site = Path(
        subprocess.check_output(
            [engine_python, "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    main_site = tmp_path / "main-site-packages"
    main_site.mkdir()
    (main_site / "engine_version.py").write_text("VERSION = 'main'\n")
    (engine_site / "engine_version.py").write_text("VERSION = 'engine-pinned'\n")
    monkeypatch.setenv("XINFERENCE_API_HOST", "original-host")
    monkeypatch.setattr(
        "xinference.deploy.local_host.__file__",
        str(main_site / "xinference/deploy/local_host.py"),
    )
    monkeypatch.setattr(
        "xinference.deploy.local_host.site.getsitepackages", lambda: [str(main_site)]
    )
    captured = {}
    popen = subprocess.Popen

    def spawn(*args, **kwargs):
        if "env" not in kwargs:
            return popen(*args, **kwargs)
        captured.update(kwargs["env"])
        raise RuntimeError("captured runtime environment")

    host = LocalHost(
        MagicMock(),
        {"supervisor_address": "127.0.0.1:1", "worker_address": "127.0.0.1:2"},
        MagicMock(),
    )
    host.directory = tmp_path
    host.connection = {"port": 1234, "token": "test"}
    with monkeypatch.context() as context:
        context.setattr("xinference.deploy.local_host.subprocess.Popen", spawn)
        with pytest.raises(RuntimeError, match="captured runtime environment"):
            await host.start_components({**os.environ, "PYTHONPATH": str(main_site)})
    monkeypatch.setattr(
        "xinference.core.virtual_env_manager.resolve_virtualenv_python_path",
        lambda manager: engine_python,
    )
    environment = build_subpool_envs_for_virtual_env(captured, True, MagicMock())
    assert "XINFERENCE_API_HOST" not in captured
    result = subprocess.check_output(
        [engine_python, "-c", "import engine_version; print(engine_version.VERSION)"],
        env=environment,
        text=True,
    )
    assert result.strip() == "engine-pinned"


@pytest.mark.skipif(os.name != "posix", reason="POSIX runtime startup")
def test_source_checkout_starts_temp_script_without_inherited_pythonpath(
    tmp_path, monkeypatch
):
    from ..local_host import _runtime_environment

    monkeypatch.delenv("PYTHONPATH", raising=False)
    script = tmp_path / "probe.py"
    script.write_text(
        "import xinference.deploy.local_host as module\nprint(module.__file__)\n"
    )
    result = subprocess.check_output(
        [sys.executable, str(script)],
        env=_runtime_environment(),
        cwd=tmp_path,
        text=True,
        timeout=15,
    )
    assert (
        Path(result.strip()).resolve()
        == Path(__file__).resolve().parents[1] / "local_host.py"
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX local startup")
@pytest.mark.parametrize("failure", ["exit", "hang"])
def test_local_startup_failure_stops_unregistered_cluster_and_children(
    tmp_path, failure
):
    ports = [get_next_port() for _ in range(3)]
    config = {
        "host": "127.0.0.1",
        "port": ports[0],
        "supervisor_address": f"127.0.0.1:{ports[1]}",
        "worker_address": f"127.0.0.1:{ports[2]}",
        "local_deployment": True,
    }
    if failure == "exit":
        cluster = "print('injected startup failure',flush=True); raise SystemExit(17)"
    else:
        cluster = (
            "import subprocess,sys,time\nfrom pathlib import Path\n"
            "p=subprocess.Popen([sys.executable,'-S','-c','import time; time.sleep(60)'])\n"
            f"Path({str(tmp_path / 'grandchild')!r}).write_text(str(p.pid))\n"
            "time.sleep(60)\n"
        )
    script = tmp_path / "start.py"
    script.write_text(
        "import subprocess,sys\nfrom pathlib import Path\n"
        "from xinference.deploy.api_host import run\n"
        "from xinference.deploy import local_host\n"
        "local_host.STARTUP_TIMEOUT=1\noriginal=subprocess.Popen\n"
        "def spawn(args,**kwargs):\n"
        "    if len(args)<3 or Path(args[2]).name!='start_local.py':\n"
        "        return original(args,**kwargs)\n"
        f"    p=original([sys.executable,'-S','-c',{cluster!r}],**kwargs)\n"
        f"    Path({str(tmp_path / 'cluster')!r}).write_text(str(p.pid))\n"
        "    return p\n"
        "local_host.subprocess.Popen=spawn\n"
        "try:\n"
        f"    run({config!r}, host_class=local_host.LocalHost)\n"
        "except RuntimeError as error:\n"
        "    assert 'Local cluster failed to start' in str(error), str(error)\n"
        "    print('startup failure cleaned up',flush=True)\n"
        "else: raise AssertionError('startup unexpectedly succeeded')\n"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(Path(__file__).resolve().parents[3]), os.environ.get("PYTHONPATH", "")]
        ),
        "XINFERENCE_HOME": str(tmp_path / "home"),
        "XINFERENCE_AUTH_ADVANCED": "0",
    }
    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=25)
            output = (tmp_path / "service.log").read_text()
            assert process.returncode == 0, output[-4000:]
            assert "startup failure cleaned up" in output
            for name in ("cluster", "grandchild"):
                if (tmp_path / name).exists():
                    until(
                        lambda: not alive(int((tmp_path / name).read_text())), timeout=5
                    )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            for name in ("cluster", "grandchild"):
                if (tmp_path / name).exists():
                    try:
                        psutil.Process(int((tmp_path / name).read_text())).kill()
                    except psutil.NoSuchProcess:
                        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["exited", "closed", "connection", "hung", "destructor"]
)
async def test_shared_supervisor_stop_tolerates_dead_or_hung_runtime(
    tmp_path, monkeypatch, failure
):
    import xoscar as xo

    from ..supervisor_host import SupervisorHost

    host = SupervisorHost({"supervisor_address": "unused:1"}, tmp_path, MagicMock())
    child = MagicMock(shared=True)
    child.process.poll.return_value = 0 if failure == "exited" else None
    destroy = AsyncMock(
        side_effect=(
            xo.ServerClosed()
            if failure == "closed"
            else (
                ConnectionError()
                if failure == "connection"
                else (
                    RuntimeError("destructor failed")
                    if failure == "destructor"
                    else None
                )
            )
        )
    )
    monkeypatch.setattr(xo, "destroy_actor", destroy)
    if failure == "hung":

        async def hang(*args):
            await asyncio.Event().wait()

        destroy.side_effect = hang

        async def timed_out(awaitable, timeout):
            assert timeout == 10
            return await asyncio.wait_for(awaitable, 0.01)

        monkeypatch.setattr(xo, "wait_for", timed_out)
    await asyncio.wait_for(host.stop_child(child), 0.5)
    assert destroy.call_count == (0 if failure == "exited" else 1)


@pytest.mark.asyncio
async def test_inline_api_failure_keeps_traceback(caplog):
    from ..local_host import InlineAPIProcess, _log_api_failure

    async def fail():
        raise RuntimeError("inline API failure outside startup")

    process = InlineAPIProcess()
    process.task = asyncio.create_task(fail())
    process.task.add_done_callback(_log_api_failure)
    await asyncio.gather(process.task, return_exceptions=True)
    assert process.poll() == 1
    assert "inline API failure outside startup" in caplog.text
    assert caplog.records[-1].exc_info[2] is not None


@pytest.mark.parametrize(
    "stop_mode",
    [
        *[
            pytest.param(
                mode,
                marks=pytest.mark.skipif(
                    os.name != "posix", reason="POSIX process group signals"
                ),
            )
            for mode in ("group-term", "group-int", "double-int")
        ],
        pytest.param(
            "ctrl-break",
            marks=pytest.mark.skipif(os.name != "nt", reason="Windows console signal"),
        ),
    ],
)
def test_local_shutdown_handles_process_group_and_second_interrupt(tmp_path, stop_mode):
    ports = [get_next_port() for _ in range(3)]
    configuration = {
        "host": "127.0.0.1",
        "port": ports[0],
        "supervisor_address": f"127.0.0.1:{ports[1]}",
        "worker_address": f"127.0.0.1:{ports[2]}",
        "local_deployment": True,
        "api_role": "local",
    }
    script = tmp_path / "start.py"
    script.write_text(
        "from pathlib import Path\n"
        "from xinference.deploy.api_host import run\n"
        "from xinference.deploy.local_host import LocalHost\n"
        "from xinference.deploy.api_worker import HostedServer\n"
        "original_drain = HostedServer.begin_drain\n"
        "def drain(process):\n"
        "    original_drain(process)\n"
        f"    Path({str(tmp_path / 'draining')!r}).touch()\n"
        "HostedServer.begin_drain = drain\n"
        "if __name__ == '__main__':\n"
        f"    run({configuration!r}, host_class=LocalHost)\n"
    )
    (tmp_path / "local_extension.py").write_text(EXTENSION)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(tmp_path),
                str(Path(__file__).resolve().parents[3]),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
        "XINFERENCE_HOME": str(tmp_path / "home"),
        "XINFERENCE_AUTH_ADVANCED": "0",
        "XINFERENCE_DISABLE_METRICS": "1",
        "XINFERENCE_EXTENSIONS": "local_extension:create",
        "LOCAL_TEST_RELEASE": str(tmp_path / "release"),
        "CUDA_VISIBLE_DEVICES": "-1",
    }
    session = requests.Session()
    session.trust_env = False
    stream = None
    owned = {}
    with (tmp_path / "service.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            endpoint = f"http://127.0.0.1:{ports[0]}"
            until(lambda: session.get(endpoint + "/local-test/info", timeout=2).ok)
            owned = {
                p.pid: p.create_time()
                for p in psutil.Process(process.pid).children(recursive=True)
            }
            if stop_mode == "double-int":
                stream = session.get(
                    endpoint + "/local-test/stream", stream=True, timeout=10
                )
                stream.raise_for_status()
                lines = stream.iter_lines(chunk_size=1)
                assert (
                    next(line for line in lines if line.startswith(b"data:"))
                    == b"data: start"
                )
                process.send_signal(signal.SIGINT)
                until(lambda: (tmp_path / "draining").exists(), timeout=5)
                assert (
                    process.poll() is None
                ), "First interrupt should drain the active stream"
                process.send_signal(signal.SIGINT)
            elif stop_mode == "ctrl-break":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(
                    process.pid,
                    signal.SIGTERM if stop_mode == "group-term" else signal.SIGINT,
                )
            process.wait(timeout=10)
            assert process.returncode == 0, (tmp_path / "service.log").read_text()[
                -5000:
            ]
            until(lambda: not any(alive(pid) for pid in owned), timeout=5)
        finally:
            if stream is not None:
                stream.close()
            session.close()
            if process.poll() is None:
                if os.name == "nt":
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            for pid, created in owned.items():
                try:
                    child = psutil.Process(pid)
                    if child.create_time() == created:
                        child.kill()
                except psutil.NoSuchProcess:
                    pass


@pytest.mark.parametrize("first", ["worker", "supervisor"])
def test_local_components_migrate_on_demand_and_keep_local_semantics(tmp_path, first):
    from xinference.client import Client

    ports = [get_next_port() for _ in range(3)]
    configuration = {
        "host": "127.0.0.1",
        "port": ports[0],
        "supervisor_address": f"127.0.0.1:{ports[1]}",
        "worker_address": f"127.0.0.1:{ports[2]}",
        "local_deployment": True,
        "api_role": "local",
        "logging_conf": {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "test": {"format": "LOCAL-UPGRADE %(levelname)s %(message)s"}
            },
            "handlers": {
                "test": {
                    "class": "logging.StreamHandler",
                    "formatter": "test",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "xinference.core.worker": {
                    "handlers": ["test"],
                    "level": "DEBUG",
                    "propagate": False,
                }
            },
        },
    }
    (tmp_path / "local_extension.py").write_text(EXTENSION)
    script = tmp_path / "start.py"
    script.write_text(
        "import asyncio\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from xinference.deploy.api_host import run\n"
        "from xinference.deploy.local_host import LocalHost\n"
        "class LimitedHost(LocalHost):\n"
        "    async def run(self, stopping):\n"
        "        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))\n"
        "        await super().run(stopping)\n"
        "if __name__ == '__main__':\n"
        f"    run({configuration!r}, host_class=LimitedHost)\n"
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
    client = None
    seen = set()
    operations = {}
    failures = []
    stop_polling = threading.Event()
    stream_started = threading.Event()

    def get_info():
        response = session.get(endpoint + "/local-test/info", timeout=5)
        response.raise_for_status()
        return response.json()

    def check_worker_logging(info):
        metadata = info["worker_metadata"][0]
        assert metadata["has_api_host"] is False
        assert metadata["level"] == logging.DEBUG
        assert metadata["formats"] == ["LOCAL-UPGRADE %(levelname)s %(message)s"]

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
        assert result["status"] == "succeeded", json.dumps(result, indent=2)
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
            client = Client(endpoint)
            client.session.trust_env = False
            assert client.list_models() == {}
            initial_pid = info["supervisor"]["pid"]
            seen.add(initial_pid)
            assert info["api_pid"] == process.pid
            assert info["local"] is True
            check_worker_logging(info)
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
                bad = submit(
                    component, "bad-" + component, str(tmp_path / "missing-python")
                )
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
                client.register_model("LLM", json.dumps(custom), False)
                assert (
                    client.get_model_registration("LLM", custom["model_name"])[
                        "model_name"
                    ]
                    == custom["model_name"]
                )
                client.unregister_model("LLM", custom["model_name"])
                with pytest.raises(RuntimeError, match="not found"):
                    client.get_model_registration("LLM", custom["model_name"])
                assert client.list_models() == {}

            registration_cycle()
            second = "supervisor" if first == "worker" else "worker"
            for component in [first, second]:
                operation = finish(component, submit(component, "first-" + component))
                operations[component] = operation
                assert get_info()["local"] is True
                check_worker_logging(get_info())
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
            check_worker_logging(get_info())
            assert alive(initial_pid)
            assert process.poll() is None
            until(
                lambda: all(
                    child.status() != psutil.STATUS_ZOMBIE
                    for child in psutil.Process(process.pid).children(recursive=True)
                )
            )
        except BaseException:
            print((tmp_path / "service.log").read_text()[-12000:], file=sys.stderr)
            raise
        finally:
            stop_polling.set()
            (tmp_path / "release").touch()
            session.close()
            if client is not None:
                client.session.close()
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            executor.shutdown(wait=True)
        # Windows terminate() stops the host immediately; its Job Object still
        # closes every retained child. POSIX SIGTERM exits through cleanup.
        assert process.returncode == (1 if os.name == "nt" else 0), (
            tmp_path / "service.log"
        ).read_text()[-5000:]
        assert not failures
        until(lambda: not any(alive(pid) for pid in seen), timeout=10)
