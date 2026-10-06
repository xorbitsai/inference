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


@pytest.mark.asyncio
async def test_sglang_chunk_uses_configured_lease_for_delayed_gpu_pull(
    gpu_module, monkeypatch
):
    import torch

    from ...vllm.xavier.test.test_direct_handoff import direct_runtime, peer_for
    from ...xavier.backends.torch import direct_handoff
    from ..xavier.settings import TRANSFER_TIMEOUT_ENV

    clock = [0.0]
    monkeypatch.setattr(
        direct_handoff, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, "600")
    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    source.caches["K"].fill_(7)
    source.register_direct("legacy", "vllm-default", [2])
    assert source.direct_requests["legacy"].deadline == 120
    actor = gpu_module.XavierGPUActor(
        None, None, SimpleNamespace(publish_source=AsyncMock()), "ns", 0
    )
    actor.transfer = source
    actor.aux = [torch.zeros((1, 1), dtype=torch.uint8)]
    actor.chunk_capacity = 1
    await actor.open(1)
    actor.init(1, 1, 0)
    await actor.add_chunk(1, [1])
    assert source.direct_requests["1:0"].deadline == 600
    clock[0] = 121
    assert source.poll_direct() == {"vllm-default"}
    monkeypatch.setattr("xoscar.actor_ref", AsyncMock(return_value=peer_for(source)))

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr("xoscar.copy_to", copy)
    assert not await dest.run(dest.load_direct, [{0: {"K": {1: 5}}}], ["1:0"])
    assert torch.equal(source.caches["K"][1], dest.caches["K"][5])


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["sender", "receiver"])
async def test_abort_marks_failure_but_retains_slot_drain_fence(gpu_module, role):
    loop = asyncio.get_running_loop()
    entered, drained = asyncio.Event(), asyncio.Event()

    async def abort(room):
        entered.set()
        await drained.wait()

    manager = SimpleNamespace(
        actor=SimpleNamespace(open=AsyncMock()),
        abort=abort,
        submit=lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop),
        call=lambda coroutine: asyncio.run_coroutine_threadsafe(coroutine, loop).result(
            timeout=2
        ),
    )
    adapter = (
        gpu_module.XavierKVSender(manager, "unused", 1, [], 0)
        if role == "sender"
        else gpu_module.XavierKVReceiver(manager, "unused", 1)
    )
    task = asyncio.create_task(asyncio.to_thread(adapter.abort))
    await entered.wait()
    assert adapter.aborted
    # Native SGLang can reuse engine slots on return, so abort must not finish
    # until the actor's actual GPU/metadata writes and source reads drain.
    assert not task.done()
    drained.set()
    await task


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["sender", "receiver"])
async def test_clear_submits_cleanup_without_scheduler_rpc_wait(gpu_module, role):
    futures = []

    def submit(coroutine):
        task = asyncio.create_task(coroutine)
        futures.append(task)
        return task

    manager = SimpleNamespace(
        actor=SimpleNamespace(clear=AsyncMock()),
        submit=submit,
        call=Mock(side_effect=AssertionError("clear must not block scheduler")),
    )
    if role == "sender":
        adapter = object.__new__(gpu_module.XavierKVSender)
        adapter.kv_mgr, adapter.room = manager, 1
        adapter._operation = concurrent.futures.Future()
    else:
        adapter = gpu_module.XavierKVReceiver(manager, "unused", 1)
    adapter.clear()
    manager.call.assert_not_called()
    if role == "sender":
        await asyncio.sleep(0)
        manager.actor.clear.assert_not_awaited()
        adapter._operation.set_result(None)
    await asyncio.gather(*futures)
    manager.actor.clear.assert_awaited_once_with(1)


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
    actor.rooms[1] = dict(
        chunks=[dict(ticket="1:0")],
        completed=asyncio.Event(),
        changed=asyncio.Event(),
        deadline=gpu_module.time.monotonic() + 600,
    )
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
async def test_missing_decode_times_out_and_drains_source_before_failure(gpu_module):
    import time

    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor.directory = SimpleNamespace(publish_source=AsyncMock())
    actor.transfer = SimpleNamespace(send_lock=asyncio.Lock(), release_direct=Mock())
    await actor.open(1)
    state = actor.rooms[1]
    state["chunks"].append(dict(ticket="1:0"))
    state["deadline"] = time.monotonic() - 1
    await actor.transfer.send_lock.acquire()
    completion = asyncio.create_task(actor.wait_done(1))
    await asyncio.sleep(0.02)
    assert not completion.done()
    actor.transfer.release_direct.assert_not_called()
    actor.transfer.send_lock.release()
    with pytest.raises(TimeoutError, match="timed out"):
        await asyncio.wait_for(completion, 1)
    assert state["aborted"] and state["completed"].is_set()
    actor.transfer.release_direct.assert_called_once_with("1:0")


@pytest.mark.asyncio
async def test_source_progress_renews_completion_deadline(gpu_module, monkeypatch):
    import torch

    monkeypatch.setattr(gpu_module, "transfer_timeout", lambda: 0.2)
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
    await asyncio.sleep(0.12)
    await actor.add_chunk(1, [0])
    actor.release_chunk("1:0")
    await asyncio.sleep(0.12)
    assert not completion.done()  # Beyond the original deadline, with progress.
    await actor.add_chunk(1, [1])
    actor.release_chunk("1:1")
    assert await asyncio.wait_for(completion, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("aborted", [False, True])
async def test_chunk_wait_wakes_on_publication_or_abort(gpu_module, aborted):
    import torch

    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor.directory = SimpleNamespace(publish_source=AsyncMock())
    actor.transfer = SimpleNamespace(send_lock=asyncio.Lock(), register_direct=Mock())
    actor.aux = [torch.zeros((1, 1), dtype=torch.uint8)]
    actor.chunk_capacity = 1
    await actor.open(1)
    actor.init(1, 1, 0)
    waiting = asyncio.create_task(actor.wait_chunk(1, 0))
    await asyncio.sleep(0.02)
    assert not waiting.done()
    if aborted:
        await actor.abort(1)
        with pytest.raises(RuntimeError, match="cancelled"):
            await asyncio.wait_for(waiting, 1)
    else:
        await actor.add_chunk(1, [0])
        chunk = await asyncio.wait_for(waiting, 1)
        assert chunk["final"] and chunk["pages"] == [0]


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
    sender._operation = sender._error = None
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["open", "init", "send"])
async def test_cross_engine_late_cancel_fails_request_without_stopping_scheduler(
    gpu_module, monkeypatch, phase
):
    loop = asyncio.get_running_loop()
    tasks = []
    error = RuntimeError("cancelled room")

    async def operation(method):
        if method == phase:
            raise error

    monkeypatch.setattr(gpu_module, "KVPoll", SimpleNamespace(Failed="failed"))
    actor = SimpleNamespace(wait_done=AsyncMock())
    calls = []

    async def invoke(method, *args):
        calls.append(method)
        await operation(method)

    actor.open = lambda *args: invoke("open", *args)
    actor.init = lambda *args: invoke("init", *args)
    actor.add_chunk = lambda *args: invoke("send", *args)

    def submit(coroutine):
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        tasks.append(future)
        return future

    manager = SimpleNamespace(
        config={"heterogeneous": True},
        actor=actor,
        call=Mock(side_effect=AssertionError("scheduler RPC wait")),
        submit=submit,
        kv_args=SimpleNamespace(gpu_id=0),
        aux=[],
    )
    monkeypatch.setattr(gpu_module.torch.cuda, "synchronize", Mock())
    sender = gpu_module.XavierKVSender(manager, "host", 123, [0], 0)
    sender.init(2, 0)
    sender.send(SimpleNamespace(tolist=lambda: [1, 2]))
    await asyncio.gather(
        *(asyncio.wrap_future(f) for f in tasks), return_exceptions=True
    )
    assert sender.poll() == gpu_module.KVPoll.Failed
    with pytest.raises(RuntimeError, match="cancelled room"):
        sender.failure_exception()
    count = len(calls)
    sender.send(SimpleNamespace(tolist=lambda: [1, 2]))
    await asyncio.gather(
        *(asyncio.wrap_future(f) for f in tasks), return_exceptions=True
    )
    assert len(calls) == count
    manager.call.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("host", [False, True])
async def test_manager_commits_engine_metadata_before_completion(gpu_module, host):
    import numpy as np
    import torch

    backing = np.zeros((3, 2), dtype=np.uint8)
    manager = object.__new__(gpu_module.XavierKVManager)
    manager._receive_tasks = {}
    manager.config = {}
    manager.config = dict(host_handoff=host)
    manager.aux = [torch.from_numpy(backing)]
    manager.actor = SimpleNamespace(
        receive=AsyncMock(return_value=(1024, [b"\x07\x09"]))
    )

    async def complete(room, nbytes, host_bytes=0):
        assert room == 1
        assert (nbytes, host_bytes) == ((0, 1024) if host else (1024, 0))
        assert backing.tolist() == [[0, 0], [7, 9], [0, 0]]

    manager.directory = SimpleNamespace(complete=AsyncMock(side_effect=complete))
    await manager.receive(1, [2], 1)
    manager.directory.complete.assert_awaited_once()
    assert not manager._receive_tasks


@pytest.mark.asyncio
async def test_manager_abort_drains_delayed_metadata_reply(gpu_module):
    import torch

    started, reply = asyncio.Event(), asyncio.Event()

    async def receive(*args):
        started.set()
        await reply.wait()
        return 1024, [b"\x07"]

    manager = object.__new__(gpu_module.XavierKVManager)
    manager._receive_tasks = {}
    manager.config = {}
    manager.aux = [torch.zeros((1, 1), dtype=torch.uint8)]
    manager.actor = SimpleNamespace(receive=receive, abort=AsyncMock())
    manager.directory = SimpleNamespace(complete=AsyncMock())
    incoming = asyncio.create_task(manager.receive(1, [2], 0))
    await started.wait()
    # GPU work may have finished, but its reply has not reached the manager.
    await manager.abort(1)
    with pytest.raises(asyncio.CancelledError):
        await incoming
    reply.set()
    await asyncio.sleep(0)
    manager.actor.abort.assert_awaited_once_with(1)
    assert not manager.aux[0].count_nonzero()
    manager.directory.complete.assert_not_awaited()
    assert not manager._receive_tasks


@pytest.mark.asyncio
async def test_manager_cancel_keeps_metadata_commit_owned(gpu_module):
    import torch

    started, drained = asyncio.Event(), asyncio.Event()

    async def receive(*args):
        started.set()
        await drained.wait()
        return 1024, [b"\x07"]

    manager = object.__new__(gpu_module.XavierKVManager)
    manager._receive_tasks = {}
    manager.config = {}
    manager.aux = [torch.zeros((1, 1), dtype=torch.uint8)]
    manager.actor = SimpleNamespace(receive=receive)
    manager.directory = SimpleNamespace(complete=AsyncMock())
    incoming = asyncio.create_task(manager.receive(1, [2], 0))
    await started.wait()
    for _ in range(2):
        incoming.cancel()
        await asyncio.sleep(0)
        assert not incoming.done()
        assert manager._receive_tasks[1] is not None
    drained.set()
    with pytest.raises(asyncio.CancelledError):
        await incoming
    assert manager.aux[0].item() == 7
    manager.directory.complete.assert_awaited_once_with(1, 1024)
    assert not manager._receive_tasks


def test_manager_close_retains_exports_until_importer_stops(gpu_module, monkeypatch):
    manager = object.__new__(gpu_module.XavierKVManager)
    manager._closed = False
    manager._receive_tasks = {}
    manager.config = {}
    manager._ipc_caches = {"0": object()}
    manager.aux = [object()]
    manager.actor = object()
    manager.directory = None
    order = []

    async def destroy(actor):
        assert actor is manager.actor and manager._ipc_caches
        order.append("destroy")

    async def stop():
        assert manager._ipc_caches and manager.aux
        order.append("stop")

    manager.pool = SimpleNamespace(stop=stop)
    manager.call = asyncio.run
    manager._loop = SimpleNamespace(
        call_soon_threadsafe=Mock(), stop=Mock(), close=Mock()
    )
    manager._thread = SimpleNamespace(join=Mock(), is_alive=lambda: False)
    monkeypatch.setattr(gpu_module.xo, "destroy_actor", destroy)
    manager.close()
    assert order == ["destroy", "stop"]
    assert not manager._ipc_caches and not manager.aux
    manager._loop.close.assert_called_once()


@pytest.mark.asyncio
async def test_ipc_actor_never_borrows_engine_addresses(gpu_module, monkeypatch):
    import torch
    from torch.multiprocessing import reductions

    cache = torch.zeros((8, 4), dtype=torch.uint8)
    rebuild = Mock(return_value=cache)
    monkeypatch.setattr(reductions, "rebuild_cuda_tensor", rebuild)
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    monkeypatch.setattr(
        gpu_module, "_buffers", Mock(side_effect=AssertionError("raw engine pointer"))
    )
    transfer = SimpleNamespace(slab_bytes=64 * 1024**2, add_slab_views=Mock())
    monkeypatch.setattr(gpu_module, "DirectGPUTransfer", Mock(return_value=transfer))
    actor = gpu_module.XavierGPUActor(
        SimpleNamespace(gpu_id=0, aux_item_lens=[2]),
        None,
        None,
        "ns",
        0,
        ipc_descriptors={"0": ("descriptor",)},
    )
    actor._gc_freeze = SimpleNamespace(start=Mock(), close=Mock())
    await actor.__post_create__()
    rebuild.assert_called_once_with("descriptor")
    assert actor.caches["0"] is cache and actor.aux is None
    actor._gc_freeze.start.assert_called_once()


@pytest.mark.asyncio
async def test_transfer_shutdown_restores_gc_after_drain_failure(gpu_module):
    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor._gc_freeze = SimpleNamespace(start=Mock(), close=Mock())
    actor.transfer = SimpleNamespace(close=AsyncMock(side_effect=RuntimeError("drain")))
    with pytest.raises(RuntimeError, match="drain"):
        await actor.__pre_destroy__()
    actor._gc_freeze.close.assert_called_once()


@pytest.mark.asyncio
async def test_borrowed_actor_does_not_freeze_engine_process(gpu_module, monkeypatch):
    import torch

    cache = torch.zeros((8, 4), dtype=torch.uint8)
    monkeypatch.setattr(gpu_module, "_buffers", Mock(return_value=({"0": cache}, [])))
    transfer = SimpleNamespace(slab_bytes=64 * 1024**2, add_slab_views=Mock())
    monkeypatch.setattr(gpu_module, "DirectGPUTransfer", Mock(return_value=transfer))
    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor._gc_freeze = SimpleNamespace(start=Mock(), close=Mock())
    await actor.__post_create__()
    actor._gc_freeze.start.assert_not_called()


@pytest.mark.asyncio
async def test_manager_exports_once_and_disables_descriptor_replay(
    gpu_module, monkeypatch
):
    from torch.multiprocessing import reductions
    from xoscar.backends.allocate_strategy import ProcessIndex

    from ...xavier import transport
    from ...xavier.contract import KVCacheContract

    contract = KVCacheContract(
        "a" * 64, "b" * 64, "c" * 64, "d" * 64, 2, 2, 4, 4, "float16"
    )
    manager = object.__new__(gpu_module.XavierKVManager)
    manager.config = dict(
        host="127.0.0.1", address="directory", uid="directory", rank=0
    )
    args = SimpleNamespace(gpu_id=0, kv_item_lens=[64] * 4, aux_item_lens=[2])
    cache = object()
    aux = object()
    monkeypatch.setattr(
        gpu_module, "_buffers", Mock(return_value=({"0": cache}, [aux]))
    )
    reduce = Mock(return_value=(None, ("descriptor",)))
    monkeypatch.setattr(reductions, "reduce_tensor", reduce)
    monkeypatch.setattr(gpu_module.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(gpu_module.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(
        transport,
        "gpu_pool_options",
        Mock(return_value={"external_address": "nixl://127.0.0.1:0"}),
    )
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.5.21")
    directory = SimpleNamespace(configure=AsyncMock(), register_peer=AsyncMock())
    pool = SimpleNamespace(external_address="nixl://127.0.0.1:1", start=AsyncMock())
    actor = SimpleNamespace(address="nixl://127.0.0.1:2")
    create_pool = AsyncMock(return_value=pool)
    create_actor = AsyncMock(return_value=actor)
    monkeypatch.setattr(gpu_module.xo, "actor_ref", AsyncMock(return_value=directory))
    monkeypatch.setattr(gpu_module.xo, "create_actor_pool", create_pool)
    monkeypatch.setattr(gpu_module.xo, "create_actor", create_actor)

    await manager._start(args, contract)
    reduce.assert_called_once_with(cache)
    assert manager._ipc_caches["0"] is cache and manager.aux == [aux]
    assert create_pool.await_args.kwargs == dict(
        n_process=1, subprocess_start_method="spawn", auto_recover=False
    )
    created_args, created_kwargs = create_actor.await_args
    assert not hasattr(created_args[1], "kv_data_ptrs")
    assert not hasattr(created_args[1], "aux_data_ptrs")
    assert created_kwargs["ipc_descriptors"] == {"0": ("descriptor",)}
    assert isinstance(created_kwargs["allocate_strategy"], ProcessIndex)
    directory.register_peer.assert_awaited_once_with(0, actor.address)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_method", [None, "open", "init", "add_chunk"])
async def test_sender_queues_rpc_without_blocking_and_reports_failures(
    gpu_module, monkeypatch, failed_method
):
    import torch

    gate = asyncio.Event()
    order = []

    async def operation(method):
        if method == "open":
            await gate.wait()
        order.append(method)
        if method == failed_method:
            raise RuntimeError(method + " failed")

    actor = SimpleNamespace(
        **{
            name: (lambda *args, name=name: operation(name))
            for name in ("open", "init", "add_chunk", "wait_done")
        }
    )
    submitted = []

    def submit(coroutine):
        result = concurrent.futures.Future()
        task = asyncio.create_task(coroutine)
        submitted.append(task)

        def finish(task):
            if task.exception() is not None:
                result.set_exception(task.exception())
            else:
                result.set_result(task.result())

        task.add_done_callback(finish)
        return result

    mgr = SimpleNamespace(
        actor=actor,
        submit=submit,
        aux=[],
        kv_args=SimpleNamespace(gpu_id=0),
        call=Mock(side_effect=AssertionError("Scheduler must not call actor RPC")),
    )
    for name in ("Failed", "Success", "Transferring", "WaitingForInput"):
        setattr(gpu_module.KVPoll, name, name)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    sender = gpu_module.XavierKVSender(mgr, "host", 1, [], 0)
    sender.init(1, 0)
    sender.send(torch.tensor([2]))
    assert not gate.is_set() and not order
    mgr.call.assert_not_called()
    gate.set()
    await asyncio.gather(*submitted, return_exceptions=True)
    await asyncio.sleep(0)
    if failed_method:
        assert sender.poll() == "Failed"
        with pytest.raises(RuntimeError, match=failed_method):
            sender.failure_exception()
    else:
        assert sender.poll() == "Success"
        assert order.index("open") < order.index("init") < order.index("add_chunk")


@pytest.mark.asyncio
async def test_failed_open_does_not_leak_gpu_room(gpu_module):
    actor = gpu_module.XavierGPUActor(None, None, None, "ns", 0)
    actor.rooms = {}
    actor.address, actor.rank = "worker:1234", 0
    actor.directory = SimpleNamespace(
        publish_source=AsyncMock(side_effect=RuntimeError("directory unavailable"))
    )
    with pytest.raises(RuntimeError, match="directory unavailable"):
        await actor.open(1)
    assert not actor.rooms
