# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Install real A/B wheels and prove which code handles each component call."""

import asyncio
import base64
import concurrent.futures
import contextlib
import functools
import hashlib
import http.server
import importlib.util
import json
import os
import site
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import psutil
import pytest
import requests
from xoscar.utils import get_next_port

from .. import upgrade

A, B, C, D, E = "9999.0.1", "9999.0.2", "9999.0.3", "9999.0.4", "9999.0.5"

# This module lives inside each test wheel. BUILD is code in that artifact,
# never supplied by the caller, environment or mocked version metadata.
ACCEPTANCE = """
import asyncio, json, os, sys
from pathlib import Path
from types import SimpleNamespace
import xoscar as xo
from xinference import __version__
from xinference.constants import XINFERENCE_HOME
from xinference.core.model import ModelActor
from xinference.core.lifecycle import management_operation
from starlette.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

BUILD = {build!r}
SOURCE_CHECKOUT = {checkout!r}

def identity(component):
    value = dict(build=BUILD, version=__version__, pid=os.getpid(), python=sys.executable, module=__file__, engine=__import__('upgrade_test_engine').VERSION, torch=str(__import__('torch').__version__))
    root = Path(XINFERENCE_HOME) / 'evidence'
    root.mkdir(exist_ok=True)
    (root / (component + '-' + BUILD + '.json')).write_text(json.dumps(value))
    return value

def actor_identity(self):
    return identity(type(self).__name__)

class Counter(ModelActor):
    def __init__(self, worker):
        xo.StatelessActor.__init__(self)
        self._model = SimpleNamespace()
        self._worker_address = worker
        self._worker_ref = None
        self._worker_generation = 0
        self._metrics_labels = {{'worker_address': worker}}
        self.value = 0
    async def __post_create__(self): pass
    async def __pre_destroy__(self): pass
    def add(self):
        self.value += 1
        return dict(identity('model'), value=self.value)
    async def finish(self):
        while not (Path(XINFERENCE_HOME) / 'release-stream').exists():
            await asyncio.sleep(.02)
        return identity('model')

@management_operation
async def install_model(self):
    from xinference.core.worker import ModelStatus
    uid = 'counter-rep0'
    address = await self._main_pool.append_sub_pool()
    model = await xo.create_actor(Counter, self.address, uid=uid, address=address)
    self._model_uid_to_model[uid] = model
    self._model_uid_to_model_status[uid] = ModelStatus(model_state='ready')
    self._model_uid_to_model_spec[uid] = {{'model_name': 'counter', 'model_type': 'LLM', 'model_ability': ['generate']}}
    self._model_uid_to_launch_args[uid] = {{'model_uid': uid}}
    self._model_uid_to_addr[uid] = address
    self._model_uid_to_pid[uid] = (await model.add())['pid']
    # Exercise an upgrade launched from a checkout: -m must not load that
    # checkout instead of the installed target package.
    os.chdir(SOURCE_CHECKOUT)
    return True

def migrate(context):
    root = Path(context['home']) / 'evidence'
    root.mkdir(exist_ok=True)
    (root / ('migration-' + context['component'] + '-' + BUILD + '.json')).write_text(json.dumps(context))
    if BUILD == 'C': raise RuntimeError('deliberate migration failure in C')

class Extension:
    name = 'component-upgrade-acceptance'
    api_version = 1
    def worker_metadata(self): return {{}}
    def configure_api(self, context):
        async def worker():
            supervisor = await context.get_supervisor_ref()
            info = await supervisor.get_workers_info()
            return await xo.actor_ref(address=info[0]['work-ip'], uid='worker')
        async def info() -> JSONResponse:
            supervisor = await context.get_supervisor_ref()
            return JSONResponse(dict(api=identity('api'), supervisor=await supervisor.upgrade_acceptance(), worker=await (await worker()).upgrade_acceptance()))
        async def install() -> JSONResponse:
            return JSONResponse(await (await worker()).install_acceptance_model())
        async def model() -> JSONResponse:
            ref = await (await worker()).get_model('counter-rep0')
            return JSONResponse(await ref.add())
        async def stream() -> EventSourceResponse:
            ref = await (await worker()).get_model('counter-rep0')
            async def chunks():
                yield json.dumps(await ref.add())
                yield json.dumps(await ref.finish())
            return EventSourceResponse(chunks())
        for path, handler in [('info', info), ('install', install), ('model', model), ('stream', stream)]:
            context.router.add_api_route('/acceptance/' + path, handler)

def create(): return Extension()
"""


def write_wheel(directory, name, version, files, requires=(), entry_points=""):
    metadata = f"{name}-{version}.dist-info"
    files = dict(files)
    files[f"{metadata}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {item}\n" for item in requires)
    ).encode()
    files[f"{metadata}/WHEEL"] = (
        b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    if entry_points:
        files[f"{metadata}/entry_points.txt"] = entry_points.encode()
    record = ""
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as wheel:
        for name, data in files.items():
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .decode()
                .rstrip("=")
            )
            record += f"{name},sha256={digest},{len(data)}\n"
            wheel.writestr(name, data)
        wheel.writestr(f"{metadata}/RECORD", record + f"{metadata}/RECORD,,\n")
    return path.name


@pytest.fixture
def package_index(tmp_path):
    root = tmp_path / "index"
    root.mkdir()
    # Share only the test runner's dependency directories. Xinference itself is
    # installed from each wheel into its own environment and verified there.
    dependencies = list(
        dict.fromkeys(
            [
                *site.getsitepackages(),
                *(
                    value
                    for value in sys.path
                    if value.endswith(("site-packages", "dist-packages"))
                ),
            ]
        )
    )
    runtime = write_wheel(
        root,
        "upgrade_test_runtime",
        "1.0",
        {
            "upgrade_test_runtime.pth": (
                "import sys; sys.path.extend(" + repr(dependencies) + ")\n"
            ).encode(),
        },
    )
    (root / "upgrade-test-runtime").mkdir()
    (root / "upgrade-test-runtime/index.html").write_text(
        f'<a href="../{runtime}">{runtime}</a>'
    )
    engine_links = []
    for version in ("1.0", "2.0"):
        filename = write_wheel(
            root,
            "upgrade_test_engine",
            version,
            {
                "upgrade_test_engine.py": f"VERSION = {version!r}\n".encode(),
            },
        )
        engine_links.append(f'<a href="../{filename}">{filename}</a>')
    (root / "upgrade-test-engine").mkdir()
    (root / "upgrade-test-engine/index.html").write_text("\n".join(engine_links))
    source = Path(__file__).resolve().parents[2]
    files = {}
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not path.is_file() or {"tests", "test", "__pycache__", "ui"} & set(
            relative.parts
        ):
            continue
        if path.suffix in {".pyc", ".so", ".pyd"}:
            continue
        files["xinference/" + relative.as_posix()] = path.read_bytes()
    links = []

    def publish(version, build):
        contents = dict(files)
        contents["xinference/_version.py"] = f"__version__ = {version!r}\n".encode()
        contents["xinference/upgrade_acceptance.py"] = ACCEPTANCE.format(
            build=build, checkout=str(source.parent)
        ).encode()
        for module, cls in [
            ("worker", "WorkerActor"),
            ("supervisor", "SupervisorActor"),
        ]:
            path = f"xinference/core/{module}.py"
            contents[path] += (
                "\nfrom xinference.upgrade_acceptance import actor_identity, install_model\n"
                f"{cls}.upgrade_acceptance = actor_identity\n"
                + (
                    f"{cls}.install_acceptance_model = install_model\n"
                    if module == "worker"
                    else ""
                )
            ).encode()
        if build == "D":
            contents["xinference/deploy/api_worker.py"] = (
                b"raise RuntimeError('deliberate API startup failure in D')\n"
            )
        filename = write_wheel(
            root,
            "xinference",
            version,
            contents,
            requires=[
                "upgrade-test-runtime==1.0",
                (
                    "upgrade-test-engine==1.0"
                    if build == "A"
                    else (
                        "upgrade-test-engine>=2.0"
                        if build == "E"
                        else "upgrade-test-engine>=1.0"
                    )
                ),
            ],
            entry_points="[xinference.migrations]\nprepare = xinference.upgrade_acceptance:migrate\n",
        )
        links.append(f'<a href="../{filename}">{filename}</a>')
        package = root / "xinference"
        package.mkdir(exist_ok=True)
        (package / "index.html").write_text("\n".join(links))

    publish(A, "A")
    publish(B, "B")

    class Index(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            # Several test releases can be published within the same second.
            # Last-Modified has only second precision and would return a stale
            # 304 for a newly published version.
            if "If-Modified-Since" in self.headers:
                del self.headers["If-Modified-Since"]
            super().do_GET()

    handler = functools.partial(Index, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", publish
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def until(call, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = call()
            if value:
                return value
        except requests.RequestException:
            pass
        time.sleep(0.05)
    raise TimeoutError("Component upgrade did not reach the expected state")


@pytest.mark.parametrize("version", [True, "../oops", "4.2rc1", "4.2.dev1", ""])
def test_upgrade_requires_an_exact_stable_version(version):
    with pytest.raises(ValueError, match="stable"):
        upgrade.validate_upgrade(version)


def test_upgrade_source_override_and_mutually_exclusive_environment():
    upgrade.validate_upgrade("4.2.0+private.1")
    assert upgrade.upgrade_index({}) == "https://pypi.org/simple"
    assert upgrade.upgrade_index({"PIP_INDEX_URL": "mirror"}) == "mirror"
    assert (
        upgrade.upgrade_index(
            {"PIP_INDEX_URL": "mirror", upgrade.UPGRADE_INDEX_ENV: "test"}
        )
        == "test"
    )
    with pytest.raises(ValueError, match="either"):
        upgrade.validate_upgrade("4.2", sys.executable)


def test_upgrade_output_redacts_rewritten_index_query_and_userinfo():
    text = upgrade.redact_output(
        "Failed https://user:secret@private.example/simple/xinference/?token=QTOKEN123#secret"
    )
    assert text == "Failed https://private.example/simple/xinference/"
    assert "QTOKEN123" not in text and "secret" not in text


@pytest.mark.asyncio
@pytest.mark.platform_slow
async def test_mock_pypi_discovers_installs_and_runs_the_new_version(
    tmp_path, monkeypatch, package_index
):
    from ... import constants

    module_path = Path(__file__).resolve().parents[3] / "scripts/mock_upgrade_index.py"
    spec = importlib.util.spec_from_file_location("mock_upgrade_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "xinference"
    source.mkdir()
    (source / "__init__.py").write_text(
        "from ._version import __version__\ndef ping(): return 'service-ok'\n"
    )
    (source / "_version.py").write_text("__version__ = '1.0.0'\n")
    metadata = 'Metadata-Version: 2.1\nName: xinference\nVersion: 1.0.0\nRequires-Python: >=3.10\nRequires-Dist: upgrade-test-runtime==1.0\nProvides-Extra: test\nRequires-Dist: sample==1.0; extra == "test"\n'
    wheel = module.build_wheel(source, tmp_path, B, metadata, "")
    with zipfile.ZipFile(wheel) as archive:
        assert archive.namelist().count("xinference/_version.py") == 1
        assert (
            "Requires-Dist: sample==1.0"
            in archive.read(f"xinference-{B}.dist-info/METADATA").decode()
        )
    upstream, _ = package_index
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        module.handler_for(wheel, upstream + "?token=private"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    home = tmp_path / "home"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(constants, "XINFERENCE_HOME", "./home")
    env = {
        **os.environ,
        upgrade.UPGRADE_INDEX_ENV: f"http://127.0.0.1:{server.server_port}/simple?token=QTOKEN123",
        "NO_PROXY": "127.0.0.1",
        "UV_CACHE_DIR": str(tmp_path / "cache"),
        "UV_OFFLINE": "true",
        "UV_CONSTRAINT": "nonexistent.txt",
        "UV_OVERRIDE": "nonexistent.txt",
        "UV_LINK_MODE": "symlink",
        "UV_EXCLUDE_NEWER": "2000-01-01",
    }
    original_start = upgrade.start_process

    async def start(command, **kwargs):
        assert all("QTOKEN123" not in arg for arg in command)
        assert kwargs["env"]["UV_INDEX_URL"] == env[upgrade.UPGRADE_INDEX_ENV]
        return await original_start(command, **kwargs)

    monkeypatch.setattr(upgrade, "start_process", start)
    try:
        update = await upgrade.check_update(sys.executable, "1.0.0", env)
        assert update == {
            "current_version": "1.0.0",
            "latest_version": B,
            "update_available": True,
        }
        python = await upgrade.prepare_upgrade(
            B, "api", sys.executable, "1.0.0", env, lambda **fields: None
        )
        assert Path(python).is_absolute() and Path(python).is_relative_to(home)
        assert not (home / "upgrades/home").exists()
        output = subprocess.check_output(
            [
                python,
                "-I",
                "-c",
                "import xinference; print(xinference.__version__, xinference.ping())",
            ],
            text=True,
        )
        assert output.strip() == B + " service-ok"
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(
                f"http://127.0.0.1:{server.server_port}/simple/dependency/",
                allow_redirects=False,
                timeout=5,
            )
            assert (
                response.headers["Location"] == upstream + "/dependency/?token=private"
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platform_slow
def test_installed_component_upgrades_execute_build_b_and_keep_model_requests(
    tmp_path, monkeypatch, package_index
):
    from ... import constants

    index, publish = package_index
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(constants, "XINFERENCE_HOME", str(home))
    environment = {
        **os.environ,
        upgrade.UPGRADE_INDEX_ENV: index,
        "XINFERENCE_HOME": str(home),
        "PYTHONPATH": "deliberately-invalid-source-path",
        "XINFERENCE_EXTENSIONS": "xinference.upgrade_acceptance:create",
        "XINFERENCE_AUTH_ADVANCED": "0",
        "XINFERENCE_DISABLE_METRICS": "1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "NO_PROXY": "127.0.0.1,localhost",
    }
    python = asyncio.run(
        upgrade.prepare_upgrade(
            A, "api", sys.executable, "0.0.0", environment, lambda **fields: None
        )
    )
    # No PyPI calls and no fake version response: A really runs from its wheel.
    configuration = {
        "host": "127.0.0.1",
        "port": get_next_port(),
        "supervisor_address": f"127.0.0.1:{get_next_port()}",
        "worker_address": f"127.0.0.1:{get_next_port()}",
        "local_deployment": True,
        "api_role": "local",
    }
    script = tmp_path / "start.py"
    script.write_text(
        "from xinference.deploy.api_host import run\nfrom xinference.deploy.local_host import LocalHost\n"
        + f"if __name__ == '__main__':\n    run({configuration!r}, host_class=LocalHost)\n"
    )
    endpoint = f"http://127.0.0.1:{configuration['port']}"
    session = requests.Session()
    session.trust_env = False
    failures, values = [], []
    stopped, stream_started = threading.Event(), threading.Event()

    def get(path):
        assert process.poll() is None, (tmp_path / "service.log").read_text(
            errors="replace"
        )[-8000:]
        response = session.get(endpoint + path, timeout=15)
        response.raise_for_status()
        return response.json()

    def traffic():
        client = requests.Session()
        client.trust_env = False
        try:
            while not stopped.is_set():
                response = client.get(endpoint + "/acceptance/model", timeout=30)
                response.raise_for_status()
                values.append(response.json())
                stopped.wait(0.03)
        except Exception as error:
            failures.append(str(error))
        finally:
            client.close()

    def stream():
        client = requests.Session()
        client.trust_env = False
        chunks = []
        try:
            with client.get(
                endpoint + "/acceptance/stream", stream=True, timeout=180
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(chunk_size=1):
                    if line.startswith(b"data: "):
                        chunks.append(json.loads(line[6:]))
                        stream_started.set()
            return chunks
        finally:
            client.close()

    def relaunch(component, version, worker=None, failed=False):
        path = f"workers/{worker}" if worker else component
        response = session.post(
            endpoint + f"/v1/{path}/relaunch",
            json={"version": version, "request_id": f"{component}-{version}"},
            timeout=15,
        )
        assert response.status_code == 202, response.text
        job = response.json()
        history = "worker" if worker else component

        def complete():
            result = get(f"/v1/{history}-relaunches/{job['operation_id']}")
            if not failed:
                assert result["status"] != "failed", result
            if (
                component == "api"
                and version == B
                and result["status"] == "draining"
                and not (home / "release-stream").exists()
            ):
                assert get("/acceptance/info")["api"]["build"] == "B"
                assert not streaming.done()
                (home / "release-stream").touch()
            return result if result["status"] in {"succeeded", "failed"} else None

        return until(complete)

    with (
        open(tmp_path / "service.log", "wb") as log,
        concurrent.futures.ThreadPoolExecutor(2) as executor,
    ):
        process = subprocess.Popen(
            [python, "-I", str(script)],
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        owned = {}
        try:
            info = until(lambda: get("/acceptance/info"))
            assert {value["build"] for value in info.values()} == {"A"}
            initial = info
            original_torch = info["worker"]["torch"]
            upgraded = set()
            assert get("/acceptance/install") is True
            model = get("/acceptance/model")
            worker = get("/v1/workers")[0]["work-ip"]
            executor.submit(traffic)
            streaming = executor.submit(stream)
            assert stream_started.wait(30)
            for component in ("worker", "supervisor", "api"):
                path = f"workers/{worker}" if component == "worker" else component
                available = get(f"/v1/{path}/updates")
                assert available == {
                    "current_version": A,
                    "latest_version": B,
                    "update_available": True,
                }
                operation = relaunch(
                    component, B, worker if component == "worker" else None
                )
                assert operation["status"] == "succeeded", operation
                info = get("/acceptance/info")
                upgraded.add(component)
                assert {key: value["build"] for key, value in info.items()} == {
                    key: "B" if key in upgraded else "A" for key in initial
                }
                evidence = info[component]
                assert evidence["build"] == "B" and evidence["version"] == B
                assert evidence["pid"] != initial[component]["pid"]
                assert Path(evidence["module"]).is_relative_to(
                    Path(evidence["python"]).parent.parent
                )
                assert (
                    json.loads(
                        (
                            home / "evidence" / f"migration-{component}-B.json"
                        ).read_text()
                    )["target_version"]
                    == B
                )
                assert get("/acceptance/model")["pid"] == model["pid"]
                assert info["worker"]["engine"] == "1.0"
                assert info["worker"]["torch"] == original_torch
                if component != "api":
                    assert not streaming.done()
                worker = get("/v1/workers")[0]["work-ip"]
            chunks = streaming.result(timeout=30)
            assert len(chunks) == 2 and {item["pid"] for item in chunks} == {
                model["pid"]
            }
            assert {value["build"] for value in info.values()} == {"B"}
            assert get("/v1/cluster/version")["version"] == B
            assert get("/status")
            assert get("/v1/api/updates")["update_available"] is False
            # Failed installation, migration and startup all leave B serving.
            before = set((home / "upgrades").iterdir())
            failure = relaunch("api", "9999.0.99", failed=True)
            assert "9999.0.99" in failure["error"], failure
            assert set((home / "upgrades").iterdir()) == before
            publish(C, "C")
            for component in ("api", "supervisor", "worker"):
                failure = relaunch(
                    component,
                    C,
                    worker=worker if component == "worker" else None,
                    failed=True,
                )
                assert "deliberate migration failure in C" in failure["error"], failure
                assert (home / "evidence" / f"migration-{component}-C.json").exists()
                assert set((home / "upgrades").iterdir()) == before
            publish(D, "D")
            failure = relaunch("api", D, failed=True)
            assert "deliberate API startup failure in D" in failure["error"], failure
            # A prepared environment is retained after a handoff failure for
            # diagnosis and manual reuse; failed preparations above are removed.
            assert len(set((home / "upgrades").iterdir()) - before) == 1
            assert get("/acceptance/info")["api"]["build"] == "B"
            publish(E, "E")
            failure = relaunch("worker", E, worker=worker, failed=True)
            assert "upgrade-test-engine" in failure["error"], failure
            current = get("/acceptance/info")["worker"]
            assert (current["build"], current["engine"], current["torch"]) == (
                "B",
                "1.0",
                original_torch,
            )
        except BaseException:
            print((tmp_path / "service.log").read_text(errors="replace")[-16000:])
            raise
        finally:
            stopped.set()
            (home / "release-stream").touch()
            owned = (
                {
                    child.pid: child.create_time()
                    for child in psutil.Process(process.pid).children(recursive=True)
                }
                if process.poll() is None
                else {}
            )
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            for pid, created in owned.items():
                with contextlib.suppress(psutil.Error):
                    child = psutil.Process(pid)
                    if child.create_time() == created:
                        child.kill()
            session.close()
    assert not failures, failures
    assert len(values) > 10
    assert {item["pid"] for item in values} == {model["pid"]}
    assert {item["build"] for item in values} == {"A"}
    assert all(
        right["value"] > left["value"] for left, right in zip(values, values[1:])
    )


@pytest.mark.parametrize("component", ["api", "supervisor"])
@pytest.mark.parametrize("version", ["4.1.0", "4.0.0"])
def test_host_rejects_non_upgrade_before_creating_operation(
    component, version, tmp_path
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from ..api_host import APIHost
    from ..supervisor_host import SupervisorHost

    host = (
        APIHost(MagicMock(), {}, MagicMock())
        if component == "api"
        else SupervisorHost({"supervisor_address": "host:1"}, tmp_path, MagicMock())
    )
    host.active = SimpleNamespace(version="4.1.0")
    with pytest.raises(ValueError, match="must be newer"):
        host.create(version=version)
    assert not host.operations


@pytest.mark.asyncio
@pytest.mark.platform_slow
async def test_missing_package_version_fails_installation(
    tmp_path, monkeypatch, package_index
):
    from ... import constants

    monkeypatch.setattr(constants, "XINFERENCE_HOME", str(tmp_path / "home"))
    index, _ = package_index
    events = []
    with pytest.raises(RuntimeError, match="9999.0.99"):
        await upgrade.prepare_upgrade(
            "9999.0.99",
            "api",
            sys.executable,
            "0.0.0",
            {
                **os.environ,
                upgrade.UPGRADE_INDEX_ENV: index,
                "NO_PROXY": "127.0.0.1",
                "UV_CACHE_DIR": str(tmp_path / "cache"),
            },
            lambda **fields: events.append(fields),
        )
    assert all(event.get("status") != "migrating" for event in events)
    assert not list((tmp_path / "home/upgrades").iterdir())


@pytest.mark.asyncio
async def test_local_reaper_preserves_package_command_exit_status(
    tmp_path, monkeypatch
):
    from ..worker_host import ProcessOwner

    owner = ProcessOwner()
    original = upgrade.start_process

    async def start(*args, **kwargs):
        process = await original(*args, **kwargs)
        owner.descendants[process.pid] = psutil.Process(process.pid).create_time()
        return process

    monkeypatch.setattr(upgrade, "start_process", start)
    task = asyncio.create_task(
        upgrade._command(
            [
                sys.executable,
                "-I",
                "-c",
                "import time; time.sleep(.15); print('deliberate install failure', flush=True); raise SystemExit(7)",
            ],
            dict(os.environ),
            tmp_path,
            lambda **fields: None,
        )
    )
    while not task.done():
        owner.reap(upgrade.active_command_pids)
        await asyncio.sleep(0.01)
    with pytest.raises(RuntimeError, match="deliberate install failure"):
        await task
    assert not upgrade.active_command_pids
