# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...model.image.docanalyze.mineru25 import Mineru2_5Model
from ...model.image.engine import (
    DiffusersImageModel,
    SGLangImageModel,
    VLLMImageModel,
    register_builtin_image_engines,
)
from ...model.image.engine_family import SUPPORTED_ENGINES as IMAGE_ENGINES
from ...model.utils import _clean_engine_param
from ...model.video.engine import (
    DiffusersVideoEngineModel,
    MLXVideoEngineModel,
    register_builtin_video_engines,
)
from ...model.video.engine_family import SUPPORTED_ENGINES as VIDEO_ENGINES
from ..exceptions import RequestQueueFullError
from ..model import ModelActor
from ..replica_load import ReplicaLoad, SerializedReplicaAdmission


class Pipeline:
    replica_autoscaling = "serialized"

    def __init__(self, model_type):
        self.model_family = SimpleNamespace(
            to_description=lambda: {"model_type": model_type, "model_name": "pipeline"}
        )
        self.calls = []
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def text_to_image(self, prompt, *_args, **_kwargs):
        self.calls.append(prompt)
        self.started.set()
        await self.finish.wait()
        return {"prompt": prompt}

    text_to_video = text_to_image


async def new_replica(model_type, model=None, timeout_ms=1000):
    model = model or Pipeline(model_type)
    actor = ModelActor(
        "supervisor",
        "worker",
        model,
        "pipeline-0",
        replica_admission={
            "concurrency_per_replica": 8,
            "queue_capacity": 1,
            "queue_timeout_ms": timeout_ms,
        },
    )
    actor.record_metrics = AsyncMock()
    actor._get_progressor = AsyncMock(side_effect=lambda _: nullcontext())
    await actor.wait_for_load()
    return actor


async def wait_for_queue(actor):
    async def poll():
        while (await actor.get_replica_load()).queued != 1:
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["image", "video"])
async def test_only_waiting_media_request_moves_to_idle_replica(
    model_type, monkeypatch
):
    async def local_actor_call(target, method, *args, **kwargs):
        kwargs.pop("_rpc_parent_metadata")
        return await getattr(target, method)(*args, **kwargs)

    monkeypatch.setattr("xinference.core.model.actor_call", local_actor_call)
    donor, receiver = [await new_replica(model_type) for _ in range(2)]
    receiver._model.finish.set()
    generate = getattr(donor, f"text_to_{model_type}")
    running = asyncio.create_task(generate("running", request_id="request-1"))
    waiting = None
    try:
        await asyncio.wait_for(donor._model.started.wait(), 1)
        assert await donor.get_replica_load() == ReplicaLoad(0, 0, 0, engine_running=1)
        waiting = asyncio.create_task(generate("waiting", request_id="request-2"))
        await wait_for_queue(donor)
        with pytest.raises(RequestQueueFullError, match="queue is full"):
            await generate("excess", request_id="request-3")
        await asyncio.sleep(0.02)
        assert (await donor.get_replica_load()).oldest_wait > 0
        assert not await donor.begin_replica_drain(0)
        assert donor.redistribute_replica_requests(receiver, 1) == 1
        assert json.loads(await asyncio.wait_for(waiting, 1)) == {"prompt": "waiting"}
        assert donor._model.calls == ["running"]
        assert receiver._model.calls == ["waiting"]
        assert donor._get_progressor.call_count == 1
        donor._model.finish.set()
        await running
        assert await donor.get_replica_load() == ReplicaLoad(0, 0, 1)
        assert await donor.begin_replica_drain(0)
    finally:
        donor._model.finish.set()
        await asyncio.gather(
            *(task for task in (running, waiting) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_queued_timeout_and_cancellation_leave_running_pipeline_owned():
    actor = await new_replica("video", timeout_ms=50)
    running = asyncio.create_task(actor.text_to_video("running", request_id="running"))
    try:
        await asyncio.wait_for(actor._model.started.wait(), 1)
        waiting = asyncio.create_task(
            actor.text_to_video("cancelled", request_id="cancelled")
        )
        await wait_for_queue(actor)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        with pytest.raises(RequestQueueFullError, match="Timed out"):
            await actor.text_to_video("expires", request_id="expires")
        running.cancel()
        # The engine retains the slot until the owned operation has exited.
        await asyncio.sleep(0)
        assert await actor.get_replica_load() == ReplicaLoad(0, 0, 0, engine_running=1)
        assert not await actor.begin_replica_drain(0)
        assert actor._model.calls == ["running"]
    finally:
        actor._model.finish.set()
        await asyncio.gather(running, return_exceptions=True)
    assert running.cancelled()
    assert await actor.get_replica_load() == ReplicaLoad(0, 0, 1)


@pytest.mark.asyncio
async def test_cancelled_waiter_releases_slot_after_handoff():
    admission = SerializedReplicaAdmission(capacity=2, timeout=1)

    async def run():
        async with admission:
            pass

    async with admission:
        first = asyncio.create_task(run())
        second = asyncio.create_task(run())
        await asyncio.sleep(0)
        assert admission.queue.qsize() == 2
    first.cancel()
    await asyncio.wait_for(second, 1)
    assert first.cancelled()
    assert not admission.running
    assert admission.queue.empty()


@pytest.mark.asyncio
async def test_serialized_mode_cannot_be_inherited_or_used_with_an_engine_scheduler():
    class Unadapted(Pipeline):
        pass

    class Scheduled(Pipeline):
        replica_autoscaling = "serialized"
        allow_batch = True

    for model_class in (Unadapted, Scheduled):
        with pytest.raises(ValueError, match="does not support replica autoscaling"):
            await new_replica("image", model=model_class("image"))


@pytest.mark.parametrize(
    "model_class, class_field, supported",
    [
        (DiffusersImageModel, "image_class", True),
        (VLLMImageModel, "image_class", True),
        (DiffusersVideoEngineModel, "video_class", True),
        (MLXVideoEngineModel, "video_class", True),
        (Pipeline, "ocr_class", False),
    ],
)
def test_discovery_advertises_only_opted_in_media_adapters(
    model_class, class_field, supported
):
    entry = _clean_engine_param({class_field: model_class}, class_field)
    assert (entry.get("replica_autoscaling") is True) == supported


@pytest.mark.parametrize(
    "register, registry",
    [
        (register_builtin_image_engines, IMAGE_ENGINES),
        (register_builtin_video_engines, VIDEO_ENGINES),
    ],
)
def test_registered_generation_adapters_declare_autoscaling(register, registry):
    register()
    assert registry
    for adapters in registry.values():
        for adapter in adapters:
            if not issubclass(adapter, Mineru2_5Model):
                assert vars(adapter).get("replica_autoscaling") in (
                    "native",
                    "serialized",
                ), adapter


@pytest.mark.parametrize(
    "options",
    [
        {"batching_max_size": 4},
        {"disagg_mode": True},
        {"disagg_role": "encoder"},
        {"dp_size": 2},
    ],
)
def test_sglang_requires_an_observable_scheduler_configuration(options):
    adapter = SGLangImageModel.__new__(SGLangImageModel)
    adapter._kwargs = options
    with pytest.raises(ValueError, match="monolithic singleton scheduler"):
        adapter.validate_serialized_replica_admission()
