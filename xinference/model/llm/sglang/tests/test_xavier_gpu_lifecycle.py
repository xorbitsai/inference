# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import concurrent.futures
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
    actor.rooms[1] = dict(chunks=[dict(ticket="1:0")], completed=asyncio.Event())
    source_completion = asyncio.create_task(actor.wait_done(1))
    await actor.transfer.send_lock.acquire()
    await asyncio.sleep(0)
    abort = asyncio.create_task(actor.abort(1))
    await cancelled.wait()
    assert not abort.done()
    drained.set()
    await asyncio.sleep(0)
    assert not abort.done()
    assert not actor.transfer.release_direct.called
    assert not source_completion.done()
    actor.transfer.send_lock.release()
    await abort
    assert incoming.cancelled()
    assert actor.rooms[1]["aborted"]
    actor.transfer.release_direct.assert_called_once_with("1:0")
    with pytest.raises(RuntimeError, match="cancelled"):
        await source_completion


@pytest.mark.asyncio
async def test_source_completion_waits_until_last_gpu_chunk_releases(gpu_module):
    import torch

    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor.directory = SimpleNamespace(publish_source=AsyncMock())
    actor.transfer = SimpleNamespace(
        register_direct=Mock(), release_direct=Mock(), poll_direct=Mock()
    )
    actor.aux = [torch.zeros((1, 1), dtype=torch.uint8)]
    actor.chunk_capacity = 1
    await actor.open(1)
    actor.init(1, 2, 0)
    completion = asyncio.create_task(actor.wait_done(1))
    await actor.add_chunk(1, [0])
    actor.release_chunk("1:0")
    await asyncio.sleep(0)
    assert not completion.done()
    await actor.add_chunk(1, [1])
    await asyncio.sleep(0)
    assert not completion.done()
    actor.release_chunk("1:1")
    assert await asyncio.wait_for(completion, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [1, 4])
async def test_cached_prefix_coalesces_until_final_or_slab_capacity(
    gpu_module, capacity
):
    import torch

    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor.directory = SimpleNamespace(publish_source=AsyncMock())
    actor.transfer = SimpleNamespace(register_direct=Mock())
    actor.aux = [torch.tensor([[7]], dtype=torch.uint8)]
    actor.chunk_capacity = capacity
    await actor.open(1)
    actor.init(1, 3, 0)
    await actor.add_chunk(1, [0, 1])
    if capacity == 4:
        assert actor.chunk(1, 0) is None
        actor.transfer.register_direct.assert_not_called()
    else:
        assert actor.chunk(1, 0)["pages"] == [0, 1]
        assert not actor.chunk(1, 0)["final"]
    await actor.add_chunk(1, [2])
    chunks = actor.rooms[1]["chunks"]
    assert [page for chunk in chunks for page in chunk["pages"]] == [0, 1, 2]
    assert chunks[-1]["final"] and chunks[-1]["aux"] == [b"\x07"]
    assert actor.transfer.register_direct.call_count == (1 if capacity == 4 else 2)
    assert not actor.rooms[1]["pending"]


@pytest.mark.parametrize("failed", [False, True])
def test_sender_poll_never_blocks_scheduler_on_actor(gpu_module, failed):
    gpu_module.KVPoll = SimpleNamespace(
        WaitingForInput="waiting",
        Transferring="transferring",
        Failed="failed",
        Success="success",
    )
    sender = object.__new__(gpu_module.XavierKVSender)
    sender.kv_mgr = SimpleNamespace(call=Mock(side_effect=AssertionError("blocked")))
    sender.inited, sender.aborted = False, False
    sender.future = concurrent.futures.Future()
    assert sender.poll() == "waiting"
    sender.inited = True
    for _ in range(3):
        assert sender.poll() == "transferring"
    if failed:
        sender.future.set_exception(RuntimeError("transfer failed"))
        assert sender.poll() == "failed"
        with pytest.raises(RuntimeError, match="transfer failed"):
            sender.failure_exception()
    else:
        sender.future.set_result(True)
        assert sender.poll() == "success"
    sender.kv_mgr.call.assert_not_called()
