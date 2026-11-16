# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
from ...model.video.minimax_h3 import SGLangMiniMaxH3Model
from ...model.video.native import SGLangVideoModel
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


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_rerank_engine_discovery_advertises_supported_adapters(
    enable_virtual_env, monkeypatch
):
    from ...model.rerank import register_builtin_model
    from ...model.rerank.llama_cpp.core import XllamaCppRerankModel
    from ...model.rerank.vllm.core import VLLMRerankModel
    from ...model.utils import (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    )

    for adapter in (XllamaCppRerankModel, VLLMRerankModel):
        monkeypatch.setattr(adapter, "check_lib", classmethod(lambda cls: True))
    register_builtin_model()
    discover = (
        get_engine_params_by_name_with_virtual_env
        if enable_virtual_env
        else get_engine_params_by_name
    )
    engines = discover("rerank", "bge-reranker-v2-m3", enable_virtual_env)
    assert engines is not None
    for engine, model_format in (
        ("sentence_transformers", "pytorch"),
        ("vllm", "pytorch"),
        ("llama.cpp", "ggufv2"),
    ):
        assert isinstance(engines[engine], list)
        assert any(
            spec.get("replica_autoscaling") is True
            and spec["model_format"] == model_format
            for spec in engines[engine]
        )


@pytest.mark.asyncio
async def test_rerank_transfers_waiting_request_with_its_arguments(monkeypatch):
    from functools import wraps
    from threading import Event

    from ...model.rerank import register_builtin_model
    from ...model.rerank.rerank_family import BUILTIN_RERANK_MODELS
    from ...model.rerank.sentence_transformers.core import (
        SentenceTransformerRerankModel,
    )

    async def local_actor_call(target, method, *args, **kwargs):
        kwargs.pop("_rpc_parent_metadata")
        return await getattr(target, method)(*args, **kwargs)

    monkeypatch.setattr("xinference.core.model.actor_call", local_actor_call)
    register_builtin_model()
    family = BUILTIN_RERANK_MODELS["bge-reranker-v2-m3"][0]
    donor_model, receiver_model = (
        SentenceTransformerRerankModel(uid, "/unused", family, "none", batch_size=1)
        for uid in ("donor", "receiver")
    )
    started, finish = asyncio.Event(), Event()
    loop = asyncio.get_running_loop()

    @wraps(donor_model._rerank)
    def score_running(documents, queries, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)
        assert documents == ["running"]
        return [0.5]

    @wraps(receiver_model._rerank)
    def score_waiting(documents, queries, **kwargs):
        assert documents == ["first", "second"]
        assert queries == ["waiting query", "waiting query"]
        return [0.2, 0.9]

    monkeypatch.setattr(donor_model, "_rerank", score_running)
    monkeypatch.setattr(receiver_model, "_rerank", score_waiting)
    donor = await new_replica("rerank", model=donor_model)
    receiver = await new_replica("rerank", model=receiver_model)
    running = asyncio.create_task(
        donor.rerank(["running"], "query", 1, None, False, False)
    )
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        waiting = asyncio.create_task(
            donor.rerank(["first", "second"], "waiting query", 1, None, True, False)
        )
        await wait_for_queue(donor)
        await asyncio.sleep(0.02)
        assert donor.redistribute_replica_requests(receiver, 1) == 1
        result = json.loads(await asyncio.wait_for(waiting, 1))
        assert result["results"] == [
            {"index": 1, "relevance_score": 0.9, "document": {"text": "second"}}
        ]
        assert not running.done()
        finish.set()
        await running
    finally:
        finish.set()
        await asyncio.gather(
            *(task for task in (running, waiting) if task is not None),
            return_exceptions=True,
        )
        for model in (donor_model, receiver_model):
            if model._process_batch_task is not None:
                model._process_batch_task.cancel()
                await asyncio.gather(model._process_batch_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_llama_cpp_rerank_reports_native_and_transferable_backlogs():
    from threading import Event

    from ...model.rerank.llama_cpp.core import XllamaCppRerankModel

    started, finish = asyncio.Event(), Event()
    loop = asyncio.get_running_loop()

    def handle_rerank(payload):
        loop.call_soon_threadsafe(started.set)
        assert finish.wait(timeout=5)
        return {
            "usage": {"total_tokens": 1},
            "results": [{"index": 0, "relevance_score": 0.5}],
        }

    def handle_metrics():
        waiting = 3 if started.is_set() and not finish.is_set() else 0
        running = 2 if waiting else 0
        return (
            f"llamacpp:requests_deferred {waiting}\n"
            f"llamacpp:requests_processing {running}\n"
        )

    model = XllamaCppRerankModel.__new__(XllamaCppRerankModel)
    model.model_family = SimpleNamespace(
        to_description=lambda: {"model_type": "rerank", "model_name": "reranker"}
    )
    server = model._llm = SimpleNamespace(
        handle_rerank=Mock(side_effect=handle_rerank), handle_metrics=handle_metrics
    )
    actor = await new_replica("rerank", model=model)
    running = asyncio.create_task(
        actor.rerank(["running"], "query", 1, None, False, False)
    )
    waiting = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        waiting = asyncio.create_task(
            actor.rerank(["waiting"], "query", 1, None, False, False)
        )
        await wait_for_queue(actor)
        load = await actor.get_replica_load()
        assert (load.queued, load.engine_waiting, load.engine_running) == (1, 3, 2)
        assert load.spare == 0
        assert server.handle_rerank.call_count == 1
        assert not await actor.begin_replica_drain(0)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert server.handle_rerank.call_count == 1
        finish.set()
        await running
        assert await actor.get_replica_load() == ReplicaLoad(0, 0, 1)
    finally:
        finish.set()
        await asyncio.gather(
            *(task for task in (running, waiting) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize(
    "model_name,backlog",
    [("bge-reranker-v2-m3", 3), ("Qwen3-VL-Reranker-2B", 4)],
)
def test_vllm_offline_rerank_reports_unfinished_pairs(model_name, backlog, monkeypatch):
    from ...model.rerank.vllm.core import VLLMRerankModel
    from ..replica_load import EngineLoad

    model = VLLMRerankModel.__new__(VLLMRerankModel)
    model.model_family = SimpleNamespace(model_name=model_name)
    model._qwen3_vl_reranker_template = "template"
    monkeypatch.setattr(model, "_clean_cache_if_needed", Mock())
    samples = []

    def score(query, documents, use_tqdm, **kwargs):
        samples.append(model.get_engine_load().waiting)
        count = len(documents) if isinstance(documents, list) else 1
        use_tqdm.update(count)
        return [object() for _ in range(count)]

    model._model = SimpleNamespace(
        score=score,
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=2)
            )
        ),
        get_metrics=lambda: [
            SimpleNamespace(name="vllm:num_requests_waiting", value=0),
            SimpleNamespace(name="vllm:num_requests_running", value=0),
        ],
    )
    model.enable_engine_load_reporting()
    model._pooling_load.configure_sync(model._model)
    assert model.get_engine_load() == EngineLoad()
    assert len(model._rerank(["passage"] * 5, "query")) == 5
    assert samples[0] == backlog
    assert model.get_engine_load() == EngineLoad()


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_class, model_type",
    [
        (SGLangImageModel, "image"),
        (SGLangVideoModel, "video"),
        (SGLangMiniMaxH3Model, "video"),
    ],
)
async def test_sglang_replica_admission_rejects_finite_rpc_timeout(
    adapter_class, model_type
):
    adapter = adapter_class.__new__(adapter_class)
    adapter.model_family = SimpleNamespace(
        to_description=lambda: {"model_type": model_type, "model_name": "pipeline"}
    )
    adapter._kwargs = {"scheduler_rpc_timeout": 1}
    ModelActor("supervisor", "worker", adapter, "fixed-0")
    with pytest.raises(ValueError, match="scheduler_rpc_timeout=None"):
        await new_replica(model_type, model=adapter)
    adapter._kwargs["scheduler_rpc_timeout"] = None
    actor = await new_replica(model_type, model=adapter)
    assert await actor.get_replica_load() == ReplicaLoad(0, 0, 1)
