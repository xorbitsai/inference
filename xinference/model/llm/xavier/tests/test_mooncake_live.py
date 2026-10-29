# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
"""Opt-in native Mooncake tests against an explicitly configured test pool."""

import asyncio
import os
import sys
from uuid import uuid4

import pytest
import torch

from ..backends.torch.backend import create_snapshot_backend
from .utils import mooncake_backend_config

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not os.environ.get("XINFERENCE_TEST_MOONCAKE_CONFIG"),
    reason="requires Linux and XINFERENCE_TEST_MOONCAKE_CONFIG JSON file",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_mode", ["normal", "timeout", "cancel"])
async def test_real_mooncake_roundtrip_and_namespace_cleanup(read_mode):
    config = mooncake_backend_config(cpu_bytes=16 << 20)
    pytest.importorskip("lmcache.lmcache_mooncake")
    stores = []
    payload = torch.arange(8192, dtype=torch.float32).reshape(2, 4096).half()
    for suffix in ("first", "other"):
        stores.append(
            create_snapshot_backend(
                8,
                "lmcache",
                config,
                namespace=uuid4().hex + suffix,
                schema={"KV": ((4096,), torch.float16)},
            )
        )
    first, other = stores

    async def clear_l1(store):
        deadline = asyncio.get_running_loop().time() + 30
        while any(
            state["pending_keys_count"] or state["in_flight_task_count"]
            for state in store._storage.manager.report_status()[
                "store_controllers"
            ].values()
        ):
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.05)
        store._storage.manager.clear()

    try:
        for store in stores:
            store.stage("KV", [1, 2], payload)
            assert store.publish([1, 2], {"KV"}) == [1, 2]
            await clear_l1(store)
        if read_mode == "timeout":
            first._config["read_timeout"] = 1e-9
            assert not await first.prefetch([1, 2])
            assert first.get_stats()["timeouts"] == 1
        elif read_mode == "cancel":
            task = asyncio.create_task(first.prefetch([1, 2]))
            await asyncio.sleep(0)
            assert first._pending is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        if first._pending is not None:
            await asyncio.shield(first._pending)
        await clear_l1(first)
        first._config["read_timeout"] = 10
        assert await first.prefetch([1, 2])
        assert first.reserve("1:remote", [1, 2])
        assert torch.equal(first.read("KV", [1, 2]), payload)
        first.release("1:remote")
        assert first.get_stats()["l2_hits"] >= 2
        assert not first.blocks and not first.leases
        await first.close()
        stores.remove(first)
        await clear_l1(other)
        assert await other.prefetch([1, 2])
        assert other.reserve("2:remote", [1, 2])
        assert torch.equal(other.read("KV", [1, 2]), payload)
        other.release("2:remote")
        assert other.get_stats()["l2_hits"] == 2
        print(
            {
                "read_mode": read_mode,
                "first": first.get_stats(),
                "other": other.get_stats(),
            },
            flush=True,
        )
    finally:
        for store in stores:
            await store.close()
