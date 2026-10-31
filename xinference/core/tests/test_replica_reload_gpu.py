# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in two-GPU acceptance for replica reload, partial failure and scale-out.

Set XINFERENCE_TEST_REPLICA_RELOAD_MODEL_PATH to a local Qwen1.5-0.5B-Chat
checkpoint. Both GPUs must have room for a serving and a replacement engine.
The interpreter must have this checkout installed for replacement subprocesses.
Shared-GPU cleanup protection is reused from the generation acceptance suite.
"""

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time

import httpx
import psutil
import pytest
import xoscar as xo

from .test_model_generation_gpu import BOOTSTRAP, port, protected_gpu_environment

pytestmark = pytest.mark.skipif(
    not os.getenv("XINFERENCE_TEST_REPLICA_RELOAD_MODEL_PATH"),
    reason="Set XINFERENCE_TEST_REPLICA_RELOAD_MODEL_PATH for two-GPU acceptance",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
async def test_two_cached_replicas_reload_and_scale_under_traffic(engine, tmp_path):
    import torch

    pytest.importorskip(engine)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA/ROCm GPUs required")
    root = tmp_path
    (root / "bootstrap.py").write_text(BOOTSTRAP)
    fault = root / "reject-second"
    env = dict(
        os.environ,
        XINFERENCE_HOME=str(root / "home"),
        XINFERENCE_ENABLE_VIRTUAL_ENV="0",
        XINFERENCE_DISABLE_METRICS="1",
        XINFERENCE_TOKEN_ROUTER_ENABLED="0",
        XINFERENCE_AUTH_ADVANCED="false",
        CUDA_VISIBLE_DEVICES=os.getenv("XINFERENCE_TEST_REPLICA_RELOAD_GPU", "0,1"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        SGLANG_PLUGINS="__replica_reload_test__",
        OMP_NUM_THREADS="2",
        MAX_JOBS="2",
        PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"],
    )
    env, _ = protected_gpu_environment(root, env, None)
    policy = root / "gpu-test-policy" / "sitecustomize.py"
    policy.write_text(
        policy.read_text()
        + f"""
from pathlib import Path
from xinference.core import model_generation
_original_capacity_check = model_generation._check_generation_capacity

def _replica_capacity_check(model, *args, **kwargs):
    if model.raw_model_uid.endswith('-rep1') and Path({str(fault)!r}).exists():
        raise ValueError('Injected second-replica capacity rejection')
    return _original_capacity_check(model, *args, **kwargs)

model_generation._check_generation_capacity = _replica_capacity_check
"""
    )
    endpoint = f"http://127.0.0.1:{port()}"
    uid = "replica-reload-" + engine
    owned = {}
    requests, chunks, errors = [], [], []
    stop = asyncio.Event()
    tasks = []
    report = {"engine": engine, "endpoint": endpoint}
    output = (root / "service.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            str(root / "bootstrap.py"),
            "local",
            "--host",
            "127.0.0.1",
            "--port",
            endpoint.rsplit(":", 1)[1],
        ],
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def remember():
        with contextlib.suppress(psutil.NoSuchProcess):
            parent = psutil.Process(process.pid)
            for child in [parent, *parent.children(recursive=True)]:
                if child.uids().real == os.getuid():
                    owned[child.pid] = child.create_time()

    async def until(check, timeout=600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remember()
            if process.poll() is not None:
                raise RuntimeError((root / "service.log").read_text()[-6000:])
            value = await check()
            if value:
                return value
            await asyncio.sleep(0.1)
        raise TimeoutError("Replica acceptance condition timed out")

    def body(tokens=8, stream=False):
        return dict(
            model=uid,
            prompt="Count the integers from 1:",
            max_tokens=tokens,
            temperature=0,
            ignore_eos=True,
            stream=stream,
            stop=[],
            stop_token_ids=[],
        )

    async with httpx.AsyncClient(
        timeout=600, limits=httpx.Limits(max_connections=20)
    ) as http:
        try:

            async def ready():
                try:
                    return (await http.get(endpoint + "/v1/models")).status_code == 200
                except httpx.HTTPError:
                    return False

            await until(ready)
            options = dict(
                max_model_len=32768,
                max_num_seqs=2,
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.3,
                kv_cache_memory_bytes=4096 * 1024**2,
                enforce_eager=True,
            )
            field = "max_num_seqs"
            if engine == "sglang":
                field = "max_running_requests"
                options = dict(
                    context_length=32768,
                    max_running_requests=2,
                    max_total_tokens=32768,
                    mem_fraction_static=0.3,
                    disable_cuda_graph=True,
                    attention_backend="triton",
                )
            response = await http.post(
                endpoint + "/v1/models",
                json=dict(
                    model_uid=uid,
                    model_name="qwen1.5-chat",
                    model_size_in_billions="0_5",
                    model_format="pytorch",
                    quantization="none",
                    model_engine=engine,
                    model_path=os.environ["XINFERENCE_TEST_REPLICA_RELOAD_MODEL_PATH"],
                    replica=2,
                    n_gpu=1,
                    enable_virtual_env=False,
                    dtype="float16",
                    **options,
                ),
            )
            response.raise_for_status()
            from ..supervisor import SupervisorActor

            address = (await http.get(endpoint + "/v1/address")).json()
            supervisor = await xo.actor_ref(
                address=address, uid=SupervisorActor.default_uid()
            )
            members, before = {}, {}
            for _ in range(2):
                ref = await supervisor.get_model(uid)
                snapshot = await ref.export_model_generation({})
                members[snapshot["model_uid"]] = ref
                before[snapshot["model_uid"]] = snapshot
            assert len(members) == 2
            assert len({snapshot["cache"]["pid"] for snapshot in before.values()}) == 2
            metadata = (
                await http.get(endpoint + f"/v1/models/{uid}/reload/config")
            ).json()
            assert metadata["enabled"] and len(metadata["replicas"]) == 2
            report["before"] = before
            print("TWO CACHED REPLICAS", list(members), flush=True)
            for _ in range(2):
                (
                    await http.post(endpoint + "/v1/completions", json=body())
                ).raise_for_status()
            started = asyncio.Event()

            async def stream():
                finished = False
                async with http.stream(
                    "POST", endpoint + "/v1/completions", json=body(16000, True)
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            item = json.loads(line[6:])
                            assert "error" not in item, item
                            finished |= any(
                                choice.get("finish_reason") is not None
                                for choice in item.get("choices", [])
                            )
                            chunks.append(time.time())
                            started.set()
                assert finished, "The original stream must finish normally"

            async def traffic():
                while not stop.is_set():
                    try:
                        (
                            await http.post(endpoint + "/v1/completions", json=body())
                        ).raise_for_status()
                        requests.append(time.time())
                    except Exception as exc:
                        errors.append(str(exc))
                    await asyncio.sleep(0.1)

            stream_task = asyncio.create_task(stream())
            traffic_task = asyncio.create_task(traffic())
            tasks.extend([stream_task, traffic_task])
            await asyncio.wait_for(started.wait(), 60)
            fault.touch()
            first_commit = []

            async def poll():
                response = await http.get(endpoint + f"/v1/models/{uid}/reload")
                response.raise_for_status()
                status = response.json()
                if status.get("completed_replicas") and not first_commit:
                    first_commit.append(time.time())
                return status if status["status"] in ("ready", "error") else None

            async def reload(value):
                response = await http.post(
                    endpoint + f"/v1/models/{uid}/reload",
                    json={"model_config": {field: value}},
                )
                assert response.status_code == 202, response.text
                return await until(poll)

            partial = await reload(4)
            report["partial"] = partial
            assert partial["status"] == "error" and partial["partial"], partial
            assert partial["completed_replicas"] == 1, partial
            assert partial["replicas"][1]["restored"], partial
            for member_uid, ref in members.items():
                snapshot = await ref.export_model_generation({})
                assert snapshot["cache"]["pid"] == before[member_uid]["cache"]["pid"]
                assert snapshot["config"][field] == (
                    4 if member_uid.endswith("rep0") else 2
                )
            print("PARTIAL FAILURE PRESERVED SERVING", flush=True)
            await asyncio.wait_for(stream_task, 300)
            assert first_commit and chunks[-1] > first_commit[0]
            fault.unlink()
            complete = await reload(4)
            report["retry"] = complete
            assert (
                complete["status"] == "ready" and complete["completed_replicas"] == 2
            ), complete
            assert complete["weights_reused"], complete
            for member_uid, ref in members.items():
                snapshot = await ref.export_model_generation({})
                assert snapshot["cache"]["pid"] == before[member_uid]["cache"]["pid"]
                assert snapshot["config"][field] == 4
            stop.set()
            await traffic_task
            (
                await http.delete(endpoint + f"/v1/models/{uid}/replicas/0")
            ).raise_for_status()
            added = await http.post(
                endpoint + f"/v1/models/{uid}/replicas", json={"replica": 1}
            )
            added.raise_for_status()
            metadata = (
                await http.get(endpoint + f"/v1/models/{uid}/reload/config")
            ).json()
            assert metadata["enabled"] and len(metadata["replicas"]) == 2, metadata
            assert metadata["model_config"][field] == 4, metadata
            report["scale_out"] = added.json()
            stop = asyncio.Event()
            traffic_task = asyncio.create_task(traffic())
            tasks.append(traffic_task)
            final = await reload(6)
            report["final"] = final
            assert (
                final["status"] == "ready" and final["completed_replicas"] == 2
            ), final
            stop.set()
            await traffic_task
            assert not errors, errors
            assert len(requests) > 5 and len(chunks) > 5
            report.update(
                passed=True,
                requests=len(requests),
                chunks=len(chunks),
                chunks_after_cutover=sum(at > first_commit[0] for at in chunks),
            )
            print("RETRY AND SCALE-OUT PASSED", len(requests), "requests", flush=True)
        finally:
            stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await http.delete(endpoint + f"/v1/models/{uid}")
            remember()
            if process.poll() is None:
                process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)
            for pid, created in owned.items():
                with contextlib.suppress(psutil.NoSuchProcess):
                    child = psutil.Process(pid)
                    if (
                        child.uids().real == os.getuid()
                        and child.create_time() == created
                        and child.status() != psutil.STATUS_ZOMBIE
                    ):
                        child.kill()
            output.close()
            report.update(errors=errors, requests=len(requests), chunks=len(chunks))
            (root / "result.json").write_text(json.dumps(report, indent=2, default=str))
