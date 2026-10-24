# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in GPU acceptance with continuous HTTP requests and an in-flight stream.

Set XINFERENCE_TEST_GENERATION_MODEL_PATH to a local Qwen3-0.6B directory.
Optionally select prepared target environments with
XINFERENCE_TEST_GENERATION_VLLM_PYTHON / XINFERENCE_TEST_GENERATION_SGLANG_PYTHON.
Both environments must contain this Xinference feature and a supported engine.
Every scenario upgrades API, Supervisor, Worker and model twice, including
local mode's first migration out of the shared startup processes.
Set XINFERENCE_TEST_GENERATION_TP=2 and XINFERENCE_TEST_GENERATION_GPU=0,1 for
fixed two-GPU tensor parallelism, or XINFERENCE_TEST_GENERATION_CUDA_GRAPH=1
for CUDA Graph capture. Leave enough free memory for two serving engines.
Results and service logs are saved under the pytest temporary directory.
Only this test's process descendants are stopped, with PID identity checks.
"""

import asyncio
import contextlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time

import httpx
import psutil
import pytest
import xoscar as xo

pytestmark = pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_GENERATION_MODEL_PATH"),
    reason="Set XINFERENCE_TEST_GENERATION_MODEL_PATH to a local Qwen3-0.6B model",
)

BOOTSTRAP = """
import sys
role = sys.argv.pop(1)
if role in {"supervisor", "local"}:
    from xinference.deploy.cmdline import supervisor
    from xinference.deploy.cmdline import local
    (local if role == "local" else supervisor)()
else:
    from xinference.deploy.cmdline import worker
    worker()
"""

GPU_TEST_POLICY = """
try:
    from xinference.core import worker

    async def no_reap(*args, **kwargs):
        return []

    worker._kill_gpu_orphans_by_ppid = no_reap
    worker._kill_orphan_gpu_pids = no_reap
    worker.WorkerActor._cleanup_gpu_orphans_on_startup = no_reap
except BaseException:
    # sitecustomize normally suppresses import failures. Never continue into
    # a real Worker when the shared-GPU test's cleanup protection failed.
    import os
    import traceback
    traceback.print_exc()
    os._exit(90)
"""


def protected_gpu_environment(root, environment, target_python):
    policy = root / "gpu-test-policy"
    policy.mkdir()
    (policy / "sitecustomize.py").write_text(GPU_TEST_POLICY)
    environment = dict(environment)
    environment["PYTHONPATH"] = str(policy) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    if target_python:
        # Explicit interpreter upgrades intentionally discard PYTHONPATH. A
        # private launcher restores only this test policy, without modifying
        # the selected environment or restoring the source package search path.
        launcher = root / "protected-python"
        launcher.write_text(
            "#!/bin/sh\n"
            f'export PYTHONPATH={shlex.quote(str(policy))}"${{PYTHONPATH:+:$PYTHONPATH}}"\n'
            f'exec {shlex.quote(target_python)} "$@"\n'
        )
        launcher.chmod(0o700)
        target_python = str(launcher)
    return environment, target_python


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("deployment", ["local", "distributed"])
async def test_gpu_upgrade_keeps_streams_and_requests_serving(
    engine, deployment, tmp_path
):
    pytest.importorskip(engine)
    root = tmp_path
    model_path = os.environ["XINFERENCE_TEST_GENERATION_MODEL_PATH"]
    (root / "bootstrap.py").write_text(BOOTSTRAP)
    target_python = os.getenv(f"XINFERENCE_TEST_GENERATION_{engine.upper()}_PYTHON")
    graph = os.getenv("XINFERENCE_TEST_GENERATION_CUDA_GRAPH") == "1"
    tp = int(os.getenv("XINFERENCE_TEST_GENERATION_TP", "1"))
    folder = root / (deployment + "-" + engine + "-" + time.strftime("%Y%m%d-%H%M%S"))
    folder.mkdir()
    (root / (engine + "-latest")).write_text(str(folder))
    env = dict(
        os.environ,
        XINFERENCE_HOME=str(folder / "home"),
        XINFERENCE_ENABLE_VIRTUAL_ENV="0",
        XINFERENCE_DISABLE_METRICS="1",
        XINFERENCE_TOKEN_ROUTER_ENABLED="0",
        XINFERENCE_AUTH_ADVANCED="false",
        CUDA_VISIBLE_DEVICES=os.getenv("XINFERENCE_TEST_GENERATION_GPU", "0"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        SGLANG_PLUGINS="__seamless_test__",
        MAX_JOBS="4",
        TMPDIR="/tmp",
        PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
    )
    env, target_python = protected_gpu_environment(root, env, target_python)
    api_port, sup_port, worker_port = port(), port(), port()
    endpoint = f"http://127.0.0.1:{api_port}"
    processes, logs, owned = [], [], {}
    report = {
        "engine": engine,
        "deployment": deployment,
        "cuda_graph": graph,
        "tp": tp,
        "endpoint": endpoint,
        "folder": str(folder),
    }
    probe = (
        "import json, sys, xinference; from importlib.metadata import version; "
        "print(json.dumps(dict(python=sys.executable, package=xinference.__file__, "
        f"xinference=xinference.__version__, engine=version({engine!r}))))"
    )
    report["installations"] = {
        name: json.loads(
            subprocess.check_output(
                [python, "-c", probe],
                env={
                    key: value
                    for key, value in os.environ.items()
                    if name == "source"
                    or target_python is None
                    or key not in {"PYTHONPATH", "PYTHONHOME"}
                },
                text=True,
            )
        )
        for name, python in (
            ("source", sys.executable),
            ("target", target_python or sys.executable),
        )
    }
    uid = "seamless-" + engine

    def remember():
        for child in processes:
            with contextlib.suppress(psutil.NoSuchProcess):
                parent = psutil.Process(child.pid)
                for proc in [parent, *parent.children(recursive=True)]:
                    if proc.uids().real == os.getuid():
                        owned[proc.pid] = proc.create_time()

    def start(name, args):
        output = (folder / (name + ".log")).open("w")
        logs.append(output)
        command = [sys.executable, str(root / "bootstrap.py"), name, *args]
        if name == "worker":
            command = [sys.executable, "-m", "xinference.deploy.worker_host", *command]
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(proc)
        print("START", name, proc.pid, flush=True)

    async def until(fn, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = await fn()
            if result:
                return result
            remember()
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Owned test process exited: {proc.pid}: {proc.returncode}"
                    )
            await asyncio.sleep(0.25)
        raise TimeoutError("Acceptance condition timed out")

    def body(tokens=8, stream=False):
        return dict(
            model=uid,
            prompt="Count the integers in order from 1:",
            temperature=0,
            max_tokens=tokens,
            ignore_eos=True,
            stop=[],
            stop_token_ids=[],
            stream=stream,
        )

    stop = asyncio.Event()
    errors, normal, chunks = [], [], []
    tasks = []
    try:
        start(
            "local" if deployment == "local" else "supervisor",
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
            ]
            + (
                ["--supervisor-port", str(sup_port)]
                if deployment == "distributed"
                else []
            ),
        )
        async with httpx.AsyncClient(
            timeout=600, limits=httpx.Limits(max_connections=20)
        ) as http:

            async def ready():
                try:
                    return (await http.get(endpoint + "/v1/models")).status_code == 200
                except httpx.HTTPError:
                    return False

            await until(ready)
            if deployment == "distributed":
                start(
                    "worker",
                    [
                        "--host",
                        "127.0.0.1",
                        "--endpoint",
                        endpoint,
                        "--worker-port",
                        str(worker_port),
                    ],
                )

            async def registered():
                response = await http.get(endpoint + "/v1/workers")
                response.raise_for_status()
                return bool(response.json())

            await until(registered)
            address = (await http.get(endpoint + "/v1/address")).json()
            options = dict(
                gpu_memory_utilization=0.3,
                kv_cache_memory_bytes=(4096 if graph or target_python else 1536)
                * 1024**2,
                max_model_len=32768 if graph or target_python else 8192,
                max_num_batched_tokens=2048,
                max_num_seqs=4,
                enforce_eager=not graph,
                tensor_parallel_size=tp,
            )
            if engine == "sglang":
                options = dict(
                    mem_fraction_static=0.3,
                    max_total_tokens=32768 if graph or target_python else 16384,
                    context_length=32768 if graph or target_python else 8192,
                    max_running_requests=4,
                    disable_cuda_graph=not graph,
                    tp_size=tp,
                    attention_backend="triton",
                )
            if engine == "sglang" and graph:
                options.update(
                    cuda_graph_max_bs_decode=4,
                    cuda_graph_bs_decode=[1, 2, 4],
                    cuda_graph_backend_prefill="disabled",
                )
            launch = await http.post(
                endpoint + "/v1/models",
                json=dict(
                    model_uid=uid,
                    model_name="qwen3",
                    model_size_in_billions="0_6",
                    model_format="pytorch",
                    quantization="none",
                    model_engine="vLLM" if engine == "vllm" else "SGLang",
                    model_path=model_path,
                    n_gpu=tp,
                    enable_virtual_env=False,
                    enable_weight_cache=True,
                    dtype="float16",
                    enable_thinking=False,
                    **options,
                ),
            )
            print("LAUNCH", launch.status_code, launch.text[-1500:], flush=True)
            launch.raise_for_status()
            warm = await http.post(endpoint + "/v1/completions", json=body())
            warm.raise_for_status()
            from xinference.core.supervisor import SupervisorActor

            supervisor = await xo.actor_ref(
                address=address, uid=SupervisorActor.default_uid()
            )
            model = await supervisor.get_model(uid)
            model_pid = await model.get_pid()
            report["before"] = await model.export_model_generation({})
            assert report["before"]["model_uid"] == f"{uid}-rep0"
            assert report["before"]["model_python_executable"] is None
            print("WARM", report["before"]["cache"]["pid"], flush=True)
            started = asyncio.Event()

            async def stream():
                done = finished = False
                async with http.stream(
                    "POST",
                    endpoint + "/v1/completions",
                    json=body(30000 if graph or target_python else 4096, True),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line == "data: [DONE]":
                            done = True
                        elif line.startswith("data: "):
                            item = json.loads(line[6:])
                            if "error" in item:
                                raise RuntimeError(item)
                            finished |= any(
                                choice.get("finish_reason") is not None
                                for choice in item.get("choices", [])
                            )
                            chunks.append(time.time())
                            started.set()
                # The completions route terminates with finish_reason;
                # unlike chat completions, it does not require a [DONE] event.
                assert (
                    finished
                ), "The original stream ended without a terminal finish_reason"
                report["stream_completed"] = True
                report["stream_done_marker"] = done

            async def traffic():
                while not stop.is_set():
                    try:
                        response = await http.post(
                            endpoint + "/v1/completions", json=body()
                        )
                        response.raise_for_status()
                        normal.append(time.time())
                    except Exception as exc:
                        errors.append(str(exc))
                    await asyncio.sleep(0.1)

            tasks = [asyncio.create_task(stream()), asyncio.create_task(traffic())]
            await asyncio.wait_for(started.wait(), 60)

            async def replace(component, round_number, executable, failed=False):
                path = f"/v1/{component}/relaunch"
                if component == "worker":
                    workers = (await http.get(endpoint + "/v1/workers")).json()
                    path = f"/v1/workers/{workers[0]['work-ip']}/relaunch"
                response = await http.post(
                    endpoint + path,
                    json={
                        "request_id": f"{component}-{round_number}",
                        "python_executable": executable,
                    },
                )
                response.raise_for_status()
                operation = response.json()

                async def settled():
                    response = await http.get(
                        endpoint
                        + f"/v1/{component}-relaunches/"
                        + operation["operation_id"]
                    )
                    response.raise_for_status()
                    value = response.json()
                    if failed:
                        assert value["status"] != "succeeded", value
                        return value if value["status"] == "failed" else None
                    if value["status"] == "failed":
                        raise RuntimeError(value)
                    # An API stream still belongs to the original API.
                    return (
                        value
                        if value["status"]
                        in (
                            {"draining", "succeeded"}
                            if component == "api" and round_number == 1
                            else {"succeeded"}
                        )
                        else None
                    )

                value = await until(settled)
                if not failed:
                    expected = report["installations"]["target"]["xinference"]
                    if component in {"api", "supervisor"}:
                        assert value["target_version"] == expected, value
                    else:
                        from xinference.core.worker import WorkerActor

                        worker = await xo.actor_ref(
                            address=value["target_worker"],
                            uid=WorkerActor.default_uid(),
                        )
                        assert (await worker.get_node_metadata())[
                            "software_version"
                        ] == expected
                report.setdefault("component_upgrades", []).append(
                    {"component": component, **value}
                )
                return value

            for component in ("api", "supervisor", "worker"):
                await replace(component, "invalid", "/does/not/exist", failed=True)
            first_api = await replace("api", 1, target_python)
            for component in ("supervisor", "worker"):
                await replace(component, 1, target_python)
            assert not tasks[
                0
            ].done(), "Long stream finished before component migration"
            assert await supervisor.is_local_deployment() is (deployment == "local")
            assert await model.get_pid() == model_pid
            assert (await model.export_model_generation({}))["cache"] == report[
                "before"
            ]["cache"]
            capacity_field = (
                "gpu_memory_utilization" if engine == "vllm" else "mem_fraction_static"
            )
            denied = await http.post(
                endpoint + f"/v1/models/{uid}/reload",
                json={"model_config": {capacity_field: 0.99}},
            )
            denied.raise_for_status()

            async def capacity_denied():
                state = (await http.get(endpoint + f"/v1/models/{uid}/reload")).json()
                return state if state["status"] == "error" else None

            report["capacity_denied"] = await until(capacity_denied)
            assert "Insufficient GPU memory" in report["capacity_denied"]["error"]
            (
                await http.post(endpoint + "/v1/completions", json=body())
            ).raise_for_status()
            print("CAPACITY_DENIED", report["capacity_denied"], flush=True)
            field = "max_num_seqs" if engine == "vllm" else "max_running_requests"
            response = await http.post(
                endpoint + f"/v1/models/{uid}/reload",
                json={
                    "model_config": {field: 8},
                    **({"python_executable": target_python} if target_python else {}),
                },
            )
            print("RELOAD", response.status_code, response.text, flush=True)
            response.raise_for_status()
            report["operation"] = response.json()

            async def finished():
                response = await http.get(endpoint + f"/v1/models/{uid}/reload")
                response.raise_for_status()
                state = response.json()
                if state["status"] == "error":
                    raise RuntimeError(state)
                return state if state["status"] == "ready" else None

            report["result"] = await until(finished)
            report["cutover_at"] = time.time()
            print("CUTOVER", report["result"], flush=True)

            async def new_generation_stream():
                terminal = False
                async with http.stream(
                    "POST", endpoint + "/v1/completions", json=body(64, True)
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            item = json.loads(line[6:])
                            assert "error" not in item, item
                            terminal |= any(
                                choice.get("finish_reason") is not None
                                for choice in item.get("choices", [])
                            )
                assert terminal, "Post-cutover stream did not finish"

            # Release a real new-generation stream through the public REST
            # route while the original generation still owns its long stream.
            await new_generation_stream()
            report["post_cutover_stream_completed"] = True
            report["after"] = await model.export_model_generation({})
            assert report["after"]["model_uid"] == report["before"]["model_uid"]
            assert report["after"]["model_python_executable"] == target_python
            expected_reuse = all(
                report["after"][key] == report["before"][key]
                for key in ("engine_version", "torch_version")
            )
            assert (
                report["after"]["cache"]["pid"] == report["before"]["cache"]["pid"]
            ) is expected_reuse
            if target_python:
                assert (
                    report["after"]["python_executable"]
                    == report["installations"]["target"]["python"]
                )
                print(
                    "VERSIONS",
                    report["before"]["engine_version"],
                    report["after"]["engine_version"],
                    flush=True,
                )
            assert report["after"]["config"][field] == 8
            # Transfer both the serving generation and the old generation whose
            # admitted long stream is still running.
            await replace("worker", 2, target_python)
            assert not tasks[0].done(), "Long stream finished before generation handoff"
            assert processes[-1].poll() is None
            await asyncio.wait_for(tasks[0], 600)

            async def api_drained():
                value = (
                    await http.get(
                        endpoint + "/v1/api-relaunches/" + first_api["operation_id"]
                    )
                ).json()
                return value if value["status"] == "succeeded" else None

            report["first_api_drained"] = await until(api_drained)
            await asyncio.sleep(2)
            for component in ("supervisor", "api"):
                await replace(component, 2, target_python)
            # A later parameter-only replacement must keep the committed
            # interpreter and cache, even after the original engine has retired
            # and the Worker has been replaced.
            again = await http.post(
                endpoint + f"/v1/models/{uid}/reload", json={"model_config": {field: 4}}
            )
            again.raise_for_status()
            report["second_reconfiguration"] = await until(finished)
            await new_generation_stream()
            report["second_generation_stream_completed"] = True
            latest = await model.export_model_generation({})
            assert latest["python_executable"] == report["after"]["python_executable"]
            assert latest["cache"]["pid"] == report["after"]["cache"]["pid"]
            stop.set()
            await tasks[1]
            report.update(
                errors=errors,
                ordinary_requests=len(normal),
                stream_chunks=len(chunks),
                ordinary_after_cutover=sum(at > report["cutover_at"] for at in normal),
                chunks_after_cutover=sum(at > report["cutover_at"] for at in chunks),
            )
            assert not errors, errors
            assert report["ordinary_after_cutover"] > 0
            assert report["chunks_after_cutover"] > 0
            invalid = await http.post(
                endpoint + f"/v1/models/{uid}/reload",
                json={"python_executable": "/does/not/exist"},
            )
            invalid.raise_for_status()

            async def failed():
                state = (await http.get(endpoint + f"/v1/models/{uid}/reload")).json()
                return state if state["status"] == "error" else None

            report["invalid_environment"] = await until(failed)
            (
                await http.post(endpoint + "/v1/completions", json=body())
            ).raise_for_status()
            (await http.delete(endpoint + f"/v1/models/{uid}")).raise_for_status()
            report["passed"] = True
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        remember()
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and any(
            proc.poll() is None for proc in processes
        ):
            await asyncio.sleep(0.1)
        survivors = []
        for pid, created in owned.items():
            with contextlib.suppress(psutil.NoSuchProcess):
                proc = psutil.Process(pid)
                if (
                    proc.uids().real == os.getuid()
                    and proc.create_time() == created
                    and proc.status() != psutil.STATUS_ZOMBIE
                ):
                    survivors.append(pid)
                    proc.kill()
        for process in processes:
            process.wait(timeout=10)
        report["cleanup_survivors"] = survivors
        for log in logs:
            log.close()
        (folder / "result.json").write_text(json.dumps(report, indent=2, default=str))
        print(
            "RESULT",
            json.dumps(
                {k: v for k, v in report.items() if k not in {"before", "after"}}
            ),
            flush=True,
        )
        if report.get("passed"):
            assert not survivors, "Service shutdown left owned processes running"
