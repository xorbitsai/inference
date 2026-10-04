# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def gpu_module(monkeypatch):
    # Exercise adapter lifetime without installing the optional CUDA engine.
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.disaggregation.base.conn",
        SimpleNamespace(
            **{
                name: object
                for name in (
                    "BaseKVManager",
                    "BaseKVReceiver",
                    "BaseKVSender",
                    "KVTransferMetric",
                )
            },
            KVPoll=SimpleNamespace(),
            KVTransferDestination=SimpleNamespace(DEVICE="device"),
        ),
    )
    name = "xinference.model.llm.sglang.xavier._gpu_lifecycle_test"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "xavier/gpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_gpu_metadata_pool_rejected_before_borrow(gpu_module, monkeypatch):
    monkeypatch.setenv("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "INTRA_NODE_NVLINK")
    with pytest.raises(ValueError, match="CPU first-token"):
        gpu_module._buffers(None, None)


def test_failed_manager_start_stops_thread(gpu_module, monkeypatch):
    from ...xavier.contract import KVCacheContract

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "float16"
    )
    import json

    monkeypatch.setenv(
        gpu_module.GPU_CONFIG_ENV, json.dumps(dict(contract=contract.to_dict()))
    )
    managers = []

    async def fail(manager, *args):
        managers.append(manager)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(gpu_module.XavierKVManager, "_start", fail)
    with pytest.raises(RuntimeError, match="startup failed"):
        gpu_module.XavierKVManager(None, None, None)
    assert managers[0]._closed
    assert not managers[0]._thread.is_alive()
    assert managers[0]._loop.is_closed()


@pytest.mark.asyncio
async def test_abort_drains_destination_and_source_before_return(gpu_module):
    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    cancelled, drained = asyncio.Event(), asyncio.Event()

    async def receive():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await drained.wait()
            raise

    incoming = asyncio.create_task(receive())
    actor.tasks[1] = incoming
    actor.transfer = SimpleNamespace(send_lock=asyncio.Lock(), release_direct=Mock())
    actor.rooms[1] = dict(chunks=[dict(ticket="1:0")])
    await actor.transfer.send_lock.acquire()
    await asyncio.sleep(0)
    abort = asyncio.create_task(actor.abort(1))
    await cancelled.wait()
    assert not abort.done()
    drained.set()
    await asyncio.sleep(0)
    assert not abort.done()
    assert not actor.transfer.release_direct.called
    actor.transfer.send_lock.release()
    await abort
    assert incoming.cancelled()
    assert actor.rooms[1]["aborted"]
    actor.transfer.release_direct.assert_called_once_with("1:0")
