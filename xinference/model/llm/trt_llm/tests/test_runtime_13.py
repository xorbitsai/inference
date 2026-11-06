# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..native import NativeSourceReleaser
from ..pd import TRTLLMPD
from ..xavier import CONFIG_ENV, configure_xavier
from . import test_core, test_priority, test_xavier

model = test_core.model
checkpoint = test_xavier.checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinator", [False, True])
@pytest.mark.parametrize("cancel_fails_first", [False, True])
async def test_modern_source_release_waits_for_native_transfer_ownership(
    coordinator, cancel_fails_first, monkeypatch
):
    backoff = Mock()
    monkeypatch.setattr("xinference.model.llm.trt_llm.native.time.sleep", backoff)
    source = SimpleNamespace(
        py_client_id=17,
        py_kv_transfer_timed_out=False,
        is_disagg_context_complete_state=False,
    )
    other = SimpleNamespace(py_client_id=18, py_kv_transfer_timed_out=False)
    transfers = {123: source, 124: other}
    polls = []
    cancel_acknowledged = False
    unpin = Mock()

    def reap(at_least=0):
        nonlocal cancel_acknowledged
        polls.append(at_least)
        if 123 not in transfers:
            return
        if not source.py_kv_transfer_timed_out:
            return
        if cancel_fails_first and len(polls) == 1:
            return
        # Native cancellation acknowledgement alone does not release KV.
        if not cancel_acknowledged:
            cancel_acknowledged = True
            return
        transfers.pop(123)
        unpin(123)

    executor = SimpleNamespace(
        async_transfer_manager=SimpleNamespace(requests_in_transfer=lambda: transfers),
        cancel_request=Mock(),
    )
    owner = SimpleNamespace(reap_context_sends=reap) if coordinator else executor
    name = (
        "reap_context_sends"
        if coordinator
        else "_check_disagg_ctx_cache_transfer_status"
    )
    setattr(owner, name, reap)
    if coordinator:
        executor.disagg = owner
    native = NativeSourceReleaser(executor)
    assert native.find_source(17) == 123
    assert native.find_source(99) is None
    tasks = [
        asyncio.create_task(asyncio.to_thread(native.release, 123)) for _ in range(2)
    ]
    for _ in range(100):
        if executor.cancel_request.called:
            break
        await asyncio.sleep(0.001)
    if cancel_fails_first:
        getattr(owner, name)(1)
        assert not cancel_acknowledged
    getattr(owner, name)(1)
    assert cancel_acknowledged and not any(t.done() for t in tasks)
    assert 123 in transfers
    unpin.assert_not_called()
    getattr(owner, name)(1)
    await asyncio.wait_for(asyncio.gather(*tasks), 2)
    assert transfers == {124: other}
    assert not other.py_kv_transfer_timed_out
    assert all(value == 0 for value in polls)
    unpin.assert_called_once_with(123)
    assert native._cancelled[123] is source
    native.release(123)
    backoff.reset_mock()
    getattr(owner, name)(1)
    assert polls[-1] == 0  # Retained native sender futures must not block the queue.
    backoff.assert_called_once_with(0.001)
    native.engine_stopped()
    assert getattr(owner, name) is reap and not native._cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinator", [False, True])
async def test_modern_source_release_reports_native_errors_without_unpin(coordinator):
    transfers = {123: SimpleNamespace(is_disagg_context_complete_state=False)}
    failure = Mock(side_effect=RuntimeError("NIXL cancel failed"))
    executor = SimpleNamespace(
        async_transfer_manager=SimpleNamespace(requests_in_transfer=lambda: transfers),
        cancel_request=Mock(),
    )
    owner = SimpleNamespace(reap_context_sends=failure) if coordinator else executor
    name = (
        "reap_context_sends"
        if coordinator
        else "_check_disagg_ctx_cache_transfer_status"
    )
    setattr(owner, name, failure)
    if coordinator:
        executor.disagg = owner
    native = NativeSourceReleaser(executor)
    task = asyncio.create_task(asyncio.to_thread(native.release, 123))
    for _ in range(100):
        if executor.cancel_request.called:
            break
        await asyncio.sleep(0.001)
    with pytest.raises(RuntimeError, match="NIXL cancel failed"):
        getattr(owner, name)()
    with pytest.raises(RuntimeError, match="NIXL cancel failed"):
        await task
    assert 123 in transfers
    native.engine_stopped()


@pytest.mark.parametrize("version", ["1.3.0rc24", "1.3.0rc29"])
def test_xavier_preserves_native_waiting_priority_and_owns_capacity(
    checkpoint, monkeypatch, version
):
    config = {
        "max_seq_len": 512,
        "scheduler_config": {"waiting_queue_policy": "priority"},
    }
    configure_xavier(checkpoint, config, {}, engine_version=version)
    assert config["scheduler_config"] == {
        "capacity_scheduler_policy": "GUARANTEED_NO_EVICT",
        "waiting_queue_policy": "priority",
    }
    import os

    assert json.loads(os.environ[CONFIG_ENV])["engine_version"] == version
    bad = {"scheduler_config": {"capacity_scheduler_policy": "MAX_UTILIZATION"}}
    with pytest.raises(ValueError, match="capacity scheduler"):
        configure_xavier(checkpoint, bad, {}, engine_version=version)


def test_xavier_uses_the_new_connector_module(monkeypatch):
    class Base:
        pass

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector",
        SimpleNamespace(KvCacheConnectorScheduler=Base, KvCacheConnectorWorker=Base),
    )
    spec = importlib.util.spec_from_file_location(
        "xinference.model.llm.trt_llm._test_new_connector",
        Path(__file__).parents[1] / "connector.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert issubclass(module.XavierConnectorWorker, Base)
    assert issubclass(module.XavierConnectorScheduler, Base)


class PDEngine(test_priority.NativePriorityEngine):
    def generate_async(self, tokens, params, streaming, priority=0.5, **kwargs):
        self.native_options = kwargs
        request = super().generate_async(tokens, params, streaming, priority)
        request.disaggregated_params = SimpleNamespace(
            first_gen_tokens=[9],
            ctx_request_id=321,
            opaque_state=b"native",
            draft_tokens=None,
            ctx_dp_rank=0,
            ctx_info_endpoint="tcp://source:1234",
        )
        if "cache_salt" in kwargs:
            from .. import pd

            state = pd._requests[42]
            if state.role == "decode":
                state.imported_tokens = len(tokens) - 1
        return request


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.3.0rc24", "1.3.0rc29"])
@pytest.mark.parametrize("stream", [False, True])
async def test_new_native_pd_loads_and_forwards_priority_on_both_roles(
    model, checkpoint, monkeypatch, version, stream
):
    monkeypatch.setattr("importlib.metadata.version", lambda name: version)
    monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", PDEngine)
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.disaggregated_params",
        SimpleNamespace(DisaggregatedParams=SimpleNamespace),
    )
    release = Mock()
    monkeypatch.setattr(
        TRTLLMPD,
        "bind_engine",
        lambda self, engine: setattr(
            self, "_native", SimpleNamespace(release=release, engine_stopped=Mock())
        ),
    )
    replicas = []
    try:
        for role in ("prefill", "decode"):
            replica = test_priority._new_model(
                model, monkeypatch, {"_nixl_config": {"role": role}}, version
            )
            replica.model_path = str(checkpoint)
            monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", PDEngine)
            replica.load()
            replicas.append(replica)
        p, d = replicas
        prefill = await p.async_generate(
            "hi",
            {"priority": -10000, "_pd_kv_transfer_params": {"do_remote_decode": True}},
            "pd-priority",
        )
        assert p._engine.priority == 1.0
        transfer = json.loads(json.dumps(prefill["_pd_kv_transfer_params"]))
        assert transfer["trt_llm"]["engine_version"] == version
        assert transfer["trt_llm"]["context"]["ctx_dp_rank"] == 0
        assert (
            transfer["trt_llm"]["context"]["ctx_info_endpoint"] == "tcp://source:1234"
        )
        result = await d.async_generate(
            "hi",
            {"priority": 10000, "stream": stream, "_pd_kv_transfer_params": transfer},
            "pd-priority",
        )
        if stream:
            assert [chunk async for chunk in result]
        assert d._engine.priority == 0.0
        assert d._engine.native_options["disaggregated_params"].ctx_dp_rank == 0
        assert not d._pd.active
        await p._pd.release("pd-priority")
        release.assert_called_once_with(321)
        assert not p._pd.active
    finally:
        for replica in replicas:
            replica.stop()


@pytest.mark.asyncio
async def test_native_pd_rejects_cross_version_handoff(model, monkeypatch):
    from ...xavier.pd_contract import prompt_digest

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.disaggregated_params",
        SimpleNamespace(DisaggregatedParams=SimpleNamespace),
    )
    pd = TRTLLMPD(
        "decode",
        "nixl",
        SimpleNamespace(fingerprint="checkpoint"),
        engine_version="1.3.0rc29",
    )
    transfer = {
        "do_remote_prefill": True,
        "trt_llm": {
            "backend": "nixl",
            "engine_version": "1.3.0rc24",
            "contract": "checkpoint",
            "prompt_digest": prompt_digest([0, 1]),
            "context": {},
        },
    }
    with pytest.raises(ValueError, match="matching runtime versions"):
        await pd.prepare("request", [0, 1], transfer)
    assert not pd.active


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.3.0rc24", "1.3.0rc29"])
@pytest.mark.parametrize("stream", [False, True])
async def test_new_xavier_pd_loads_and_forwards_priority_on_both_roles(
    model, checkpoint, monkeypatch, version, stream
):
    from .. import gpu

    monkeypatch.setattr(
        "xinference.model.llm.xavier.transport.gpu_pool_options", lambda *args: {}
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 42),
    )
    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=Mock()))
    client = SimpleNamespace(
        namespace="namespace",
        config={"rank": 1},
        rpc=Mock(return_value={}),
        close=Mock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.trt_llm.connector",
        SimpleNamespace(XavierClient=lambda: client, close_clients=Mock()),
    )
    replicas = []
    try:
        for role in ("prefill", "decode"):
            replica = test_priority._new_model(
                model,
                monkeypatch,
                {"_xavier_cache_config": {"role": role, "rank": 1}},
                version,
            )
            replica.model_path = checkpoint
            monkeypatch.setattr(sys.modules["tensorrt_llm"], "LLM", PDEngine)
            replica.load()
            replicas.append(replica)
            assert replica._engine.kwargs["scheduler_config"] == {
                "capacity_scheduler_policy": "GUARANTEED_NO_EVICT",
                "waiting_queue_policy": "priority",
            }
        p, d = replicas
        prefill = await p.async_generate(
            "hi",
            {"priority": -10000, "_pd_kv_transfer_params": {"do_remote_decode": True}},
            "pd-priority",
        )
        assert p._engine.priority == 1.0 and "cache_salt" in p._engine.native_options
        transfer = json.loads(json.dumps(prefill["_pd_kv_transfer_params"]))
        result = await d.async_generate(
            "hi",
            {"priority": 10000, "stream": stream, "_pd_kv_transfer_params": transfer},
            "pd-priority",
        )
        if stream:
            assert [chunk async for chunk in result]
        assert d._engine.priority == 0.0 and "cache_salt" in d._engine.native_options
        assert not d._pd.active
        await p._pd.release("pd-priority")
        assert not p._pd.active
        assert any(call.args[0] == "release" for call in client.rpc.call_args_list)
    finally:
        for replica in replicas:
            replica.stop()
