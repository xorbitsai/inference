# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Real CUDA/two-replica LMCache regression; opt in explicitly.

Set XINFERENCE_TEST_LMCACHE_GPU_INDICES=0,1 to exercise separate GPUs.
"""

import asyncio
import json
import multiprocessing
import os
import time
from copy import deepcopy

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_LMCACHE_GPU") != "1",
    reason="requires Linux CUDA, LMCache 0.5.6 and locally available model weights",
)


@pytest.mark.parametrize("deployment_mode", ["local", "supervisor-workers"])
def test_lmcache_gpu_prefix_reuse(monkeypatch, tmp_path, deployment_mode):
    import torch
    import xoscar as xo

    from xinference.conftest import TEST_FILE_LOGGING_CONF
    from xinference.deploy.local import health_check, run_in_subprocess

    assert torch.cuda.is_available()
    multiprocessing.set_start_method("spawn", force=True)
    monkeypatch.setenv("XINFERENCE_AUTH_ADVANCED", "false")
    monkeypatch.setenv("XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU", "1")
    logging_config = deepcopy(TEST_FILE_LOGGING_CONF)
    logging_config["handlers"]["file_handler"].update(
        filename=str(tmp_path / "cluster.log"), maxBytes=0, backupCount=0
    )
    address = f"127.0.0.1:{xo.utils.get_next_port()}"
    processes = []
    try:
        worker_addresses = None
        endpoint = None
        if deployment_mode == "local":
            cluster = run_in_subprocess(address, None, None, logging_config)
        else:
            from xinference.deploy.supervisor import run_in_subprocess as run_supervisor

            cluster = run_supervisor(address, logging_config)
        processes.append(cluster)
        assert health_check(address=address, max_attempts=30, sleep_interval=1)
        if deployment_mode == "supervisor-workers":
            worker_addresses = [
                f"127.0.0.1:{xo.utils.get_next_port()}" for _ in range(2)
            ]
            for worker_address in worker_addresses:
                process = multiprocessing.Process(
                    target=_run_test_worker,
                    args=(worker_address, address, logging_config),
                )
                process.start()
                processes.append(process)
            import requests

            port = xo.utils.get_next_port()
            api = multiprocessing.Process(
                target=_run_test_api, args=(address, port, logging_config)
            )
            api.start()
            processes.append(api)
            endpoint = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while True:
                try:
                    response = requests.get(f"{endpoint}/status", timeout=1)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                assert time.monotonic() < deadline, "REST API did not start"
                time.sleep(0.1)
        asyncio.run(_exercise_cluster(address, tmp_path, worker_addresses, endpoint))
    finally:
        for process in reversed(processes):
            if process.is_alive():
                process.kill()
            process.join(timeout=15)


def _run_test_worker(address, supervisor_address, logging_config):
    import logging.config

    from xinference.deploy.worker import _start_worker

    logging.config.dictConfig(logging_config)
    asyncio.run(_start_worker(address, supervisor_address, logging_conf=logging_config))


def _run_test_api(supervisor_address, port, logging_config):
    from xinference.api import restful_api

    restful_api.run(
        supervisor_address=supervisor_address,
        host="127.0.0.1",
        port=port,
        logging_conf=logging_config,
        api_role="supervisor",
    )


async def _exercise_cluster(address, tmp_path, worker_addresses=None, endpoint=None):
    import xoscar as xo

    from xinference.core.replica_config import DeviceConfig, ReplicaConfig
    from xinference.core.supervisor import SupervisorActor
    from xinference.core.worker import WorkerActor
    from xinference.model.llm.vllm.xavier.transfer import TransferActor

    supervisor = await xo.actor_ref(address=address, uid=SupervisorActor.default_uid())
    if worker_addresses is not None:
        deadline = asyncio.get_running_loop().time() + 30
        while len(await supervisor.get_workers_info()) != 2:
            assert asyncio.get_running_loop().time() < deadline, "Workers did not join"
            await asyncio.sleep(0.1)
    worker_addresses = worker_addresses or [address, address]
    workers = [
        await xo.actor_ref(address=worker_address, uid=WorkerActor.default_uid())
        for worker_address in worker_addresses
    ]
    uid = "lmcache-gpu-regression"
    model_path = os.environ["XINFERENCE_TEST_LMCACHE_MODEL_PATH"]
    gpu_indices = [
        int(index)
        for index in os.environ.get("XINFERENCE_TEST_LMCACHE_GPU_INDICES", "0,0").split(
            ","
        )
    ]
    assert len(gpu_indices) == 2
    disk = tmp_path / "kv"
    launch_config = dict(
        model_uid=uid,
        model_name="qwen2.5-instruct",
        model_engine="vLLM",
        model_type="LLM",
        model_size_in_billions="0_5",
        model_path=model_path,
        model_format="pytorch",
        quantization="none",
        replica=2,
        enable_virtual_env=False,
        enable_xavier=True,
        xavier_cache_backend="lmcache",
        xavier_cache_backend_config={
            "cpu_bytes": 8 << 20,
            "disk_path": str(disk),
            "disk_bytes": 128 << 20,
            "read_timeout": 10,
        },
        max_model_len=1024,
        gpu_memory_utilization=float(
            os.environ.get("XINFERENCE_TEST_LMCACHE_GPU_MEMORY_UTILIZATION", "0.2")
        ),
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_num_seqs=4,
        replica_config=[
            ReplicaConfig(
                devices=[
                    DeviceConfig(worker_ip=worker_address, n_gpu=1, gpu_idx=[gpu_index])
                ]
            )
            for worker_address, gpu_index in zip(worker_addresses, gpu_indices)
        ],
    )
    if endpoint is None:
        await supervisor.launch_builtin_model(**launch_config)
    else:
        import requests

        body = {
            **launch_config,
            "replica_config": [
                config.dict() for config in launch_config["replica_config"]
            ],
        }
        response = await asyncio.to_thread(
            requests.post, f"{endpoint}/v1/models", json=body, timeout=600
        )
        response.raise_for_status()
        assert response.json()["model_uid"] == uid
    try:
        producer = await workers[0].get_model(f"{uid}-rep0")
        consumer = await workers[1].get_model(f"{uid}-rep1")
        transfer = await xo.actor_ref(
            address=producer.address, uid=f"{TransferActor.default_uid()}-1"
        )
        verification_number = "73814926"
        target = (
            f"The verification number is {verification_number}.\n"
            + " ".join(["signal"] * 420)
            + ". What is the verification number from the first line? "
            "Reply only with the eight digits."
        )

        async def chat(model, prompt, salt):
            result = await model.chat(
                [{"role": "user", "content": prompt}],
                {"temperature": 0, "max_tokens": 16, "cache_salt": salt},
            )
            if isinstance(result, (bytes, str)):
                result = json.loads(result)
            return result

        original = await chat(producer, target, "scope-a")
        # New, unrelated prefixes on the same producer exceed its DRAM pool.
        # The old prefix remains on disk; local vLLM prefix reuse is disabled.
        for word in ("pressureone", "pressuretwo", "pressurethree"):
            await chat(producer, " ".join([word] * 220), "pressure")
        # LMCache 0.5.6 checks L1 eviction on a one-second background tick.
        # Allow that tick to run before claiming this is a disk-cache test.
        await asyncio.sleep(1.5)
        before = await transfer.get_cache_backend_stats_v1()
        restored = await chat(consumer, target, "scope-a")
        after = await transfer.get_cache_backend_stats_v1()
        isolated = await chat(consumer, target, "scope-b")
        after_isolated = await transfer.get_cache_backend_stats_v1()
        content = lambda result: result["choices"][0]["message"]["content"]
        evidence = {
            "gpu_indices": gpu_indices,
            "supervisor_address": address,
            "worker_addresses": worker_addresses,
            "launch_transport": "rest" if endpoint else "actor",
            "producer_before": before,
            "producer_after": after,
            "producer_after_different_salt": after_isolated,
            "original": original,
            "restored": restored,
            "different_salt": isolated,
            "disk_files": len(list(disk.rglob("*.data"))),
        }
        (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
        print(json.dumps(evidence, indent=2), flush=True)
        assert verification_number in content(original)
        assert content(original) == content(restored) == content(isolated)
        assert restored["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        assert isolated["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert after["l2_hits"] > before["l2_hits"]
        assert after_isolated["l2_hits"] == after["l2_hits"]
        assert after["resident_pages"] == 0
    finally:
        await supervisor.terminate_model(uid)
    assert not list(disk.rglob("*.data"))
