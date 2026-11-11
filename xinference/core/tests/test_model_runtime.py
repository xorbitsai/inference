# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import xoscar as xo

from ...model.llm.vllm.core import VLLMChatModel
from ..exceptions import ModelNotReadyError
from ..model import ModelActor
from ..model_runtime import get_model_runtime_info
from .test_model_generation import ServingGeneration


class RuntimeGeneration(ServingGeneration):
    def __init__(self, name, length):
        super().__init__(name)
        self.length = length

    def get_runtime_context_length(self):
        return self.length


@pytest.mark.asyncio
async def test_runtime_metadata_follows_generation_cutover_over_actor_transport():
    async with await xo.create_actor_pool("127.0.0.1", n_process=0) as pool:
        anchor = await xo.create_actor(
            RuntimeGeneration,
            "anchor",
            131072,
            uid="anchor",
            address=pool.external_address,
        )
        target = await xo.create_actor(
            RuntimeGeneration,
            "target",
            4096,
            uid="target",
            address=pool.external_address,
        )
        assert await anchor.get_runtime_context_length() == 131072
        try:
            await anchor.commit_model_generation(target, "reload", True, "cache")
            assert await anchor.get_runtime_context_length() == 4096
        finally:
            await anchor.close_model_generations()


def make_supervisor(lengths):
    workers = {}
    for index, length in enumerate(lengths):
        model = SimpleNamespace(
            get_runtime_context_length=AsyncMock(return_value=length)
        )
        if isinstance(length, Exception):
            model.get_runtime_context_length.side_effect = length
        workers[f"demo-{index}"] = SimpleNamespace(
            get_model=AsyncMock(return_value=model)
        )
    return SimpleNamespace(
        _replica_model_uid_to_worker=workers,
        _iter_active_replica_model_uids=lambda uid: list(workers),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lengths,expected",
    [
        ([8192, 4096], 4096),
        ([8192, None], None),
        ([8192, RuntimeError("loading")], None),
        ([True], None),
        ([], None),
    ],
)
async def test_aggregate_uses_smallest_confirmed_limit(lengths, expected):
    result = await get_model_runtime_info(make_supervisor(lengths), "demo")
    assert result["runtime_context_length"] == expected
    assert len(result["replicas"]) == len(lengths)
    assert result["model_uid"] == "demo"


@pytest.mark.asyncio
async def test_actor_observes_current_adapter_and_readiness():
    actor = ModelActor.__new__(ModelActor)
    actor._model_state = "ready"
    actor._model = VLLMChatModel.__new__(VLLMChatModel)
    actor._model._context_length = 4096
    assert await actor.get_runtime_context_length() == 4096
    actor._model._context_length = 8192
    assert await actor.get_runtime_context_length() == 8192
    actor._model_state = "loading"
    with pytest.raises(ModelNotReadyError):
        await actor.get_runtime_context_length()


@pytest.mark.asyncio
async def test_missing_worker_and_distributed_driver():
    supervisor = make_supervisor([8192])
    worker = supervisor._replica_model_uid_to_worker["demo-0"]
    supervisor._replica_model_uid_to_worker["demo-0"] = [worker, None]
    assert (await get_model_runtime_info(supervisor, "demo"))[
        "runtime_context_length"
    ] == 8192
    supervisor._replica_model_uid_to_worker["demo-0"] = None
    assert (await get_model_runtime_info(supervisor, "demo"))[
        "runtime_context_length"
    ] is None


@pytest.mark.asyncio
async def test_dead_runtime_is_bounded_and_topology_change_is_unknown(monkeypatch):
    from .. import model_runtime

    wait_for = asyncio.wait_for
    monkeypatch.setattr(
        model_runtime.asyncio, "wait_for", lambda coro, timeout: wait_for(coro, 0.01)
    )
    supervisor = make_supervisor([8192])
    worker = supervisor._replica_model_uid_to_worker["demo-0"]

    async def hang(**kwargs):
        await asyncio.sleep(10)

    worker.get_model.side_effect = hang
    result = await get_model_runtime_info(supervisor, "demo")
    assert result["runtime_context_length"] is None
    supervisor = make_supervisor([8192])
    # The lookup accepts a model UID; keep the changing snapshots explicit.
    snapshots = iter([["demo-0"], ["demo-0", "demo-1"]])
    supervisor._iter_active_replica_model_uids = lambda uid: next(snapshots)
    assert (await get_model_runtime_info(supervisor, "demo"))[
        "runtime_context_length"
    ] is None


@pytest.mark.asyncio
async def test_runtime_timeout_does_not_wait_for_actor_cancellation(monkeypatch):
    from xoscar.backends.context import IndigenActorContext
    from xoscar.backends.message import CancelMessage, ResultMessage

    from .. import model_runtime

    reply = asyncio.get_running_loop().create_future()
    cancellation_started = asyncio.Event()
    release_cancellation = asyncio.Event()

    class StalledContext(IndigenActorContext):
        async def _call(self, address, message, **kwargs):
            if isinstance(message, CancelMessage):
                cancellation_started.set()
                await release_cancellation.wait()
                reply.set_result(ResultMessage(message.message_id, result=8192))
                return ResultMessage(message.message_id, result=True)
            return reply

    context = StalledContext()
    model = SimpleNamespace(
        get_runtime_context_length=lambda: context.send(
            xo.ActorRef("stalled:1234", "demo-rep0"),
            ("get_runtime_context_length", 0, (), {}),
        )
    )
    supervisor = make_supervisor([8192])
    supervisor._replica_model_uid_to_worker["demo-0"].get_model.return_value = model
    wait_for = asyncio.wait_for
    monkeypatch.setattr(
        model_runtime.asyncio, "wait_for", lambda coro, timeout: wait_for(coro, 0.01)
    )
    task = asyncio.create_task(get_model_runtime_info(supervisor, "demo"))
    try:
        await wait_for(cancellation_started.wait(), timeout=1)
        done, _ = await asyncio.wait({task}, timeout=0.1)
        assert (
            task in done
        ), "Runtime metadata waited for a stalled actor's cancellation"
        assert task.result()["runtime_context_length"] is None
    finally:
        release_cancellation.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
