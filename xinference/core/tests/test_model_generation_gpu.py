# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in GPU acceptance with continuous HTTP requests and an in-flight stream.

Set XINFERENCE_TEST_GENERATION_MODEL_PATH to a local Qwen3-0.6B directory.
Optionally select prepared target environments with
XINFERENCE_TEST_GENERATION_VLLM_PYTHON / XINFERENCE_TEST_GENERATION_SGLANG_PYTHON.
Both environments must contain this Xinference feature and a supported engine.
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
if role == "supervisor":
    from xinference.deploy.cmdline import supervisor
    supervisor()
else:
    from xinference.core import worker as module
    from xinference.core.worker import WorkerActor
    from xinference.deploy.cmdline import worker
    async def no_reap(*args, **kwargs):
        return []
    # A GPU acceptance test must not reclaim another service's processes.
    module._kill_gpu_orphans_by_ppid = no_reap
    WorkerActor._cleanup_gpu_orphans_on_startup = no_reap
    worker()
"""


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
async def test_gpu_upgrade_keeps_streams_and_requests_serving(engine, tmp_path):
    pytest.importorskip(engine)
    root = tmp_path
    model_path = os.environ["XINFERENCE_TEST_GENERATION_MODEL_PATH"]
    (root / "bootstrap.py").write_text(BOOTSTRAP)
    target_python = os.getenv(f"XINFERENCE_TEST_GENERATION_{engine.upper()}_PYTHON")
    hosted = True
    graph = os.getenv("XINFERENCE_TEST_GENERATION_CUDA_GRAPH") == "1"
    tp = int(os.getenv("XINFERENCE_TEST_GENERATION_TP", "1"))
    folder = root / (engine + "-" + time.strftime("%Y%m%d-%H%M%S"))
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
    api_port, sup_port, worker_port = port(), port(), port()
    endpoint = f"http://127.0.0.1:{api_port}"
    processes, logs, owned = [], [], {}
    report = {
        "engine": engine,
        "cuda_graph": graph,
        "tp": tp,
        "endpoint": endpoint,
        "folder": str(folder),
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
        if name == "worker" and hosted:
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
            "supervisor",
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
                "--supervisor-port",
                str(sup_port),
            ],
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
                return (
                    f"127.0.0.1:{worker_port}"
                    in (await http.get(endpoint + "/v1/workers")).text
                )

            await until(registered)
            options = dict(
                gpu_memory_utilization=0.3,
                kv_cache_memory_bytes=(
                    4096 if graph else (2048 if target_python else 1536)
                )
                * 1024**2,
                max_model_len=32768 if graph else (16384 if target_python else 8192),
                max_num_batched_tokens=2048,
                max_num_seqs=4,
                enforce_eager=not graph,
                tensor_parallel_size=tp,
            )
            if engine == "sglang":
                options = dict(
                    mem_fraction_static=0.3,
                    max_total_tokens=32768 if graph else 16384,
                    context_length=(
                        32768 if graph else (16384 if target_python else 8192)
                    ),
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
                address=f"127.0.0.1:{sup_port}", uid=SupervisorActor.default_uid()
            )
            model = await supervisor.get_model(uid)
            report["before"] = await model.export_model_generation({})
            print("WARM", report["before"]["cache"]["pid"], flush=True)
            started = asyncio.Event()

            async def stream():
                done = finished = False
                async with http.stream(
                    "POST",
                    endpoint + "/v1/completions",
                    json=body(
                        30000 if graph else (12000 if target_python else 4096), True
                    ),
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
            report["after"] = await model.export_model_generation({})
            expected_reuse = all(
                report["after"][key] == report["before"][key]
                for key in ("engine_version", "torch_version")
            )
            assert (
                report["after"]["cache"]["pid"] == report["before"]["cache"]["pid"]
            ) is expected_reuse
            if target_python:
                assert report["after"]["python_executable"] == target_python
                print(
                    "VERSIONS",
                    report["before"]["engine_version"],
                    report["after"]["engine_version"],
                    flush=True,
                )
            if hosted:
                host = processes[1]
                before_pid = host.pid
                handoff = await http.post(
                    endpoint + f"/v1/workers/127.0.0.1:{worker_port}/relaunch",
                    json={"request_id": "seamless-upgrade"},
                )
                handoff.raise_for_status()
                operation = handoff.json()

                async def handoff_done():
                    value = (
                        await http.get(
                            endpoint
                            + "/v1/worker-relaunches/"
                            + operation["operation_id"]
                        )
                    ).json()
                    if value["status"] == "failed":
                        raise RuntimeError(value)
                    return value if value["status"] == "succeeded" else None

                report["worker_relaunch"] = await until(handoff_done)
                assert host.poll() is None and host.pid == before_pid
                print("WORKER_RELAUNCH", report["worker_relaunch"], flush=True)
            assert report["after"]["config"][field] == 8
            await asyncio.wait_for(tasks[0], 600)
            await asyncio.sleep(2)
            # A later parameter-only replacement must keep the committed
            # interpreter and cache, even after the original engine has retired
            # and the Worker has been replaced.
            again = await http.post(
                endpoint + f"/v1/models/{uid}/reload", json={"model_config": {field: 4}}
            )
            again.raise_for_status()
            report["second_reconfiguration"] = await until(finished)
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
        await asyncio.sleep(2)
        for pid, created in owned.items():
            with contextlib.suppress(psutil.NoSuchProcess):
                proc = psutil.Process(pid)
                if proc.uids().real == os.getuid() and proc.create_time() == created:
                    proc.kill()
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
