# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Real CUDA/two-replica LMCache regression; opt in explicitly.

Set XINFERENCE_TEST_LMCACHE_GPU_INDICES=0,1 to exercise separate GPUs.
"""

import asyncio
import json
import multiprocessing
import os
from copy import deepcopy

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_LMCACHE_GPU") != "1",
    reason="requires Linux CUDA, LMCache 0.5.6 and locally available model weights",
)


def test_lmcache_gpu_prefix_reuse(monkeypatch, tmp_path):
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
    cluster = run_in_subprocess(address, None, None, logging_config)
    try:
        assert health_check(address=address, max_attempts=30, sleep_interval=1)
        asyncio.run(_exercise_cluster(address, tmp_path))
    finally:
        cluster.kill()
        cluster.join(timeout=15)


async def _exercise_cluster(address, tmp_path):
    import xoscar as xo

    from xinference.core.replica_config import DeviceConfig, ReplicaConfig
    from xinference.core.supervisor import SupervisorActor
    from xinference.core.worker import WorkerActor
    from xinference.model.llm.vllm.xavier.transfer import TransferActor

    supervisor = await xo.actor_ref(address=address, uid=SupervisorActor.default_uid())
    worker = await xo.actor_ref(address=address, uid=WorkerActor.default_uid())
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
    await supervisor.launch_builtin_model(
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
                devices=[DeviceConfig(worker_ip=address, n_gpu=1, gpu_idx=[gpu_index])]
            )
            for gpu_index in gpu_indices
        ],
    )
    try:
        producer = await worker.get_model(f"{uid}-rep0")
        consumer = await worker.get_model(f"{uid}-rep1")
        transfer = await xo.actor_ref(
            address=producer.address, uid=f"{TransferActor.default_uid()}-1"
        )
        target = (
            " ".join(["original"] * 440)
            + ". Which word was repeated above? Reply with that word only."
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
        assert content(original) == content(restored) == content(isolated)
        assert restored["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        assert isolated["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert after["l2_hits"] > before["l2_hits"]
        assert after_isolated["l2_hits"] == after["l2_hits"]
        assert after["resident_pages"] == 0
    finally:
        await supervisor.terminate_model(uid)
    assert not list(disk.rglob("*.data"))
