# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in fresh-process CUDA P/D correctness with forced LMCache disk reuse.

Set XINFERENCE_TEST_PD_HISTORY=1, XINFERENCE_TEST_PD_MODEL_PATH,
XINFERENCE_TEST_PD_HISTORY_GPUS (P,D), and XINFERENCE_TEST_PD_HISTORY_PYTHONS
(JSON mapping vllm/sglang/trt-llm to their Python executables).
"""

import asyncio
import json
import os
import signal

import pytest
import xoscar as xo

from ...sglang.xavier.directory import XavierPDDirectory
from ...trt_llm.tests.test_cross_engine_live import SCRIPT
from ..history import XavierHistoryProxy
from .utils import DiskHistoryCacheActor

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_PD_HISTORY") != "1",
    reason="Requires reserved GPUs, LMCache and separate engine environments",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefill,decode,restore_prefill",
    [
        ("vllm", "sglang", "vllm"),
        ("sglang", "vllm", "sglang"),
        ("trt-llm", "vllm", "trt-llm"),
        ("vllm", "vllm", "sglang"),
        ("sglang", "vllm", "vllm"),
        ("trt-llm", "vllm", "vllm"),
        ("vllm", "vllm", "trt-llm"),
    ],
)
async def test_live_pd_restores_history_after_engines_restart(
    tmp_path, prefill, decode, restore_prefill
):
    pythons = json.loads(os.environ["XINFERENCE_TEST_PD_HISTORY_PYTHONS"])
    gpus = os.environ["XINFERENCE_TEST_PD_HISTORY_GPUS"].split(",")
    assert len(gpus) == 2
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        store = await xo.create_actor(
            DiskHistoryCacheActor,
            dict(
                name="lmcache",
                namespace="live-history",
                config=dict(
                    cpu_bytes=64 << 20,
                    disk_bytes=128 << 20,
                    disk_path=str(tmp_path / "kv"),
                    read_timeout=10,
                ),
            ),
            address=pool.external_address,
        )
        try:
            reports = []
            for cycle in range(2):
                p_engine = restore_prefill if cycle else prefill
                root = tmp_path / str(cycle)
                root.mkdir()
                directory = await xo.create_actor(
                    XavierPDDirectory,
                    address=pool.external_address,
                    uid="trt-cross-live",
                )
                proxy = await xo.create_actor(
                    XavierHistoryProxy,
                    dict(address=store.address, uid=store.uid),
                    64 << 20,
                    read_timeout=10,
                    address=pool.external_address,
                    uid=f"history-proxy-{cycle}",
                )
                (root / "config.json").write_text(
                    json.dumps(
                        dict(
                            address=pool.external_address,
                            model=os.environ["XINFERENCE_TEST_PD_MODEL_PATH"],
                            logical_dtype=(
                                "bfloat16"
                                if "trt-llm" in (prefill, decode, restore_prefill)
                                else "float16"
                            ),
                            baseline_texts=(
                                [
                                    record["baseline"]
                                    for record in json.loads(
                                        (tmp_path / "0" / "results.json").read_text()
                                    )[::2]
                                ]
                                if cycle
                                else None
                            ),
                            history=dict(
                                address=proxy.address,
                                uid=f"history-proxy-{cycle}",
                                capacity_bytes=64 << 20,
                                shared_prefix=True,
                                write_behind=True,
                                read_timeout=10,
                            ),
                        )
                    )
                )
                processes, logs = [], []

                async def start(role, engine, gpu):
                    log = (root / (role + ".log")).open("w")
                    logs.append(log)
                    processes.append(
                        await asyncio.create_subprocess_exec(
                            pythons[engine],
                            "-c",
                            SCRIPT,
                            str(root),
                            role,
                            engine,
                            stdout=log,
                            stderr=asyncio.subprocess.STDOUT,
                            start_new_session=True,
                            env=dict(
                                os.environ,
                                CUDA_VISIBLE_DEVICES=gpu,
                                SGLANG_PLUGINS="__xinference_test__",
                                TOKENIZERS_PARALLELISM="false",
                            ),
                        )
                    )

                async def wait(name):
                    for _ in range(3000):
                        assert all(
                            p.returncode in (None, 0) for p in processes
                        ), f"Engine failed; inspect {root}"
                        if (root / name).exists():
                            return
                        await asyncio.sleep(0.1)
                    raise TimeoutError(f"Missing {name}; inspect {root}")

                try:
                    await start("decode", decode, gpus[1])
                    await wait("ready")
                    await start("prefill", p_engine, gpus[0])
                    await wait("decode.done")
                    await wait("prefill.done")
                    (root / "release").touch()
                    await asyncio.wait_for(
                        asyncio.gather(*(p.wait() for p in processes)), 60
                    )
                    assert all(p.returncode == 0 for p in processes)
                    await proxy.flush()
                    stats = await proxy.get_stats()
                    pd = await directory.get_stats()
                    (root / "metrics.json").write_text(
                        json.dumps(
                            dict(proxy=stats, pd=pd, storage=await store.get_stats()),
                            indent=2,
                        )
                    )
                    assert pd["completed_requests"] == 6 and pd["gpu_bytes"] > 0
                    assert pd["host_bytes"] == 0
                    remaining = 0
                    for room in range(100, 106):
                        try:
                            await directory.request_info(room)
                        except RuntimeError as error:
                            assert "unavailable" in str(error)
                            continue
                        # Completed metadata can outlive the final P scheduler
                        # poll until the deployment directory is destroyed.
                        assert await directory.check(room)
                        remaining += 1
                    assert pd["active_handoffs"] == remaining
                    assert stats["enqueued_pages"] > 0 or stats["imported_pages"] > 0
                    if cycle:
                        if p_engine == "sglang":
                            # SGLang's storage adapter copies into its host pool;
                            # native usage reports the subsequent GPU reuse.
                            assert stats["read_pages"] > 0
                            cold = json.loads(
                                (tmp_path / "0" / "prefill-results.json").read_text()
                            )
                            warm = json.loads(
                                (root / "prefill-results.json").read_text()
                            )

                            def cached(record):
                                return record["usage"]["prompt_tokens_details"][
                                    "cached_tokens"
                                ]

                            if prefill == restore_prefill:
                                assert cached(warm[4]) > cached(cold[4])
                            else:
                                assert cached(warm[4]) >= 128
                        else:
                            assert stats["imported_pages"] > 0
                        assert (await store.get_stats())["backend_stats"][
                            "l2_hits"
                        ] > reports[0]["storage"]["backend_stats"]["l2_hits"]
                    reports.append(
                        dict(proxy=stats, pd=pd, storage=await store.get_stats())
                    )
                    (tmp_path / "history-results.json").write_text(
                        json.dumps(reports, indent=2)
                    )
                finally:
                    (root / "release").touch()
                    for p in processes:
                        if p.returncode is None:
                            os.killpg(p.pid, signal.SIGTERM)
                            try:
                                await asyncio.wait_for(p.wait(), 20)
                            except asyncio.TimeoutError:
                                os.killpg(p.pid, signal.SIGKILL)
                                await p.wait()
                    for log in logs:
                        log.close()
                    await xo.destroy_actor(proxy)
                    await xo.destroy_actor(directory)
                await store.force_disk_reads()
        finally:
            await xo.destroy_actor(store)
