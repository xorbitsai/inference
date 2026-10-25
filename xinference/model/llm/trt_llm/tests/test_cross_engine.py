# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ...sglang.xavier.config import configure_xavier as configure_sglang
from ...xavier.backends.torch.pd import canonical_vllm_views
from ...xavier.contract import KVCacheContract
from ...xavier.pd_contract import build_pd_contract
from .. import gpu
from ..cross_engine import TRTCrossEngineGPUActor, canonical_trt_views
from ..pd import TRTLLMPD, XavierRequest
from ..xavier import CONFIG_ENV, configure_xavier
from .test_core import Request
from .test_core import model as model
from .test_xavier import checkpoint as checkpoint
from .test_xavier import connector as connector


def test_trt_and_sglang_build_the_same_bf16_contract(checkpoint, monkeypatch):
    monkeypatch.setenv(CONFIG_ENV, "")
    monkeypatch.setattr(
        "xinference.model.llm.xavier.transport.gpu_pool_options", lambda *a: {}
    )
    cache = dict(
        role="prefill",
        rank=1,
        host="127.0.0.1",
        heterogeneous=True,
        logical_dtype="bfloat16",
    )
    trt, sglang = dict(max_seq_len=512), dict(context_length=512)
    configure_xavier(checkpoint, trt, cache)
    configure_sglang(checkpoint, sglang, cache)
    import os

    trt_contract = KVCacheContract.from_dict(
        json.loads(os.environ[CONFIG_ENV])["contract"]
    )
    trt_contract.require_match(KVCacheContract.from_dict(cache["contract"]))
    assert trt["dtype"] == sglang["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="logical_dtype"):
        trt_contract.require_match(build_pd_contract(checkpoint, 512))
    with pytest.raises(ValueError, match="bfloat16"):
        configure_sglang(checkpoint, dict(dtype="float16"), cache)
    configure_xavier(checkpoint, {}, {**cache, "host_handoff": True})
    with pytest.raises(ValueError, match="BF16 replicas"):
        configure_xavier(checkpoint, {}, {**cache, "logical_dtype": "float16"})


def test_trt_views_share_storage_and_preserve_head_token_order(checkpoint):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    cache = (
        (torch.arange(3 * c.num_layers * c.layer_nbytes // 2, dtype=torch.int16) + 256)
        .view(torch.bfloat16)
        .reshape(3, c.num_layers, 2, -1)
    )
    views = canonical_trt_views(cache, c)
    for layer in range(c.num_layers):
        native = cache[:, layer].reshape(3, 2, c.num_kv_heads, 64, c.head_dim)
        vllm = canonical_vllm_views(
            {
                f"model.layers.{i}.attn": cache[:, i]
                .reshape(3, 2, c.num_kv_heads, 64, c.head_dim)
                .transpose(2, 3)
                for i in range(c.num_layers)
            },
            3,
            c,
        )
        for kind in (0, 1):
            key = str(layer + kind * c.num_layers)
            assert torch.equal(
                views[key].view(torch.int16),
                native[:, kind].transpose(1, 2).view(torch.int16),
            )
            assert torch.equal(
                views[key].view(torch.int16), vllm[key].view(torch.int16)
            )
            views[key][1, 5, 1, 2] = -123
            assert native[1, kind, 1, 5, 2] == -123
    with pytest.raises(ValueError, match="BF16"):
        canonical_trt_views(cache.float(), c)


@pytest.mark.asyncio
async def test_trt_source_is_pinned_before_first_token_and_waits_for_metadata(
    checkpoint,
):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=65)
    actor = TRTCrossEngineGPUActor({}, 0, c, directory, 1)
    actor.address = "localhost:1234"
    actor.transfer = SimpleNamespace(
        register_direct=Mock(),
        release_direct=Mock(),
        direct_requests={},
        send_lock=asyncio.Lock(),
    )
    await actor.publish(123, "request", [7, 2])
    actor.transfer.register_direct.assert_called_once()
    waiting = asyncio.create_task(actor.wait_chunk(123, 0))
    await asyncio.sleep(0)
    assert not waiting.done()
    actor.finalize(123, 42)
    assert (await waiting)["aux"] == dict(first_token=42, prompt_tokens=65)
    await actor.abort(123)
    assert not actor.rooms
    actor.transfer.release_direct.assert_called_once_with("123:0")
    # A cancelled room cannot be finalized or published after directory release.
    with pytest.raises(RuntimeError, match="cancelled"):
        actor.finalize(123, 42)
    with pytest.raises(RuntimeError, match="cancelled"):
        await actor.publish(123, "late", [7, 2])
    actor.aborted[123] = 0
    actor._prune_aborted()
    assert not actor.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("host_handoff", [False, True])
async def test_trt_cross_engine_uses_router_room_and_exact_prompt(
    checkpoint, role, host_handoff
):
    pd = TRTLLMPD(role, "nixl")
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    pd.client = SimpleNamespace(
        config=dict(heterogeneous=True, rank=1, host_handoff=host_handoff),
        namespace=c.fingerprint,
        rpc=Mock(),
    )
    tokens = list(range(65))
    handoff = dict(room=123, mode="host" if host_handoff else "gpu")
    flag = "do_remote_decode" if role == "prefill" else "do_remote_prefill"
    state = await pd.prepare("r", tokens, {flag: True, "sglang_xavier": handoff})
    assert state["context"].room == 123
    assert state["context"].num_pages == (2 if role == "prefill" else 1)
    args = pd.client.rpc.call_args.args
    assert args[0:3] == ("prepare", 123, c.fingerprint)
    with pytest.raises(ValueError, match="cache scope"):
        await pd.prepare(
            "salted",
            tokens,
            {flag: True, "sglang_xavier": {**handoff, "cache_scope": "x"}},
        )
    assert "salted" not in pd.active
    with pytest.raises(ValueError, match="handoff room"):
        await pd.prepare(
            "wrong-mode",
            tokens,
            {
                flag: True,
                "sglang_xavier": dict(handoff, mode="gpu" if host_handoff else "host"),
            },
        )
    assert "wrong-mode" not in pd.active
    assert XavierRequest(role, tuple(tokens), 123).num_pages == 1


@pytest.mark.asyncio
async def test_cancel_during_source_lookup_prevents_late_registration(checkpoint):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    directory = AsyncMock()
    entered, resume = asyncio.Event(), asyncio.Event()

    async def info(room):
        entered.set()
        await resume.wait()
        return dict(prompt_tokens=65)

    directory.request_info.side_effect = info
    actor = TRTCrossEngineGPUActor({}, 0, c, directory, 1)
    actor.transfer = SimpleNamespace(send_lock=asyncio.Lock())
    publishing = asyncio.create_task(actor.publish(123, "cancelled", [7, 2]))
    await entered.wait()
    await actor.abort(123)
    resume.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        await publishing
    assert not actor.rooms
    directory.publish_source.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "abort", "close"])
async def test_destroy_always_closes_cuda_import(checkpoint, monkeypatch, failure):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    actor = TRTCrossEngineGPUActor({}, 0, c, AsyncMock(), 1)
    actor._ipc_pointer = 123
    actor.rooms[1] = {}
    actor.abort = AsyncMock(
        side_effect=RuntimeError("abort") if failure == "abort" else None
    )
    actor.transfer = SimpleNamespace(
        close=AsyncMock(
            side_effect=RuntimeError("close") if failure == "close" else None
        )
    )
    close = Mock()
    monkeypatch.setattr("xinference.model.llm.trt_llm.cross_engine.close_import", close)
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            await actor.__pre_destroy__()
    else:
        await actor.__pre_destroy__()
    close.assert_called_once_with(123)
    assert actor._ipc_pointer is None


@pytest.mark.asyncio
@pytest.mark.parametrize("host", [False, True])
async def test_cross_prefill_finalizes_first_stop_token(model, monkeypatch, host):
    client = SimpleNamespace(
        config=dict(heterogeneous=True, host_handoff=host, rank=1),
        namespace="namespace",
        rpc=Mock(),
        close=Mock(),
    )
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.trt_llm.connector",
        SimpleNamespace(XavierClient=lambda: client),
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.inputs",
        SimpleNamespace(get_cache_salt_id=lambda salt: 42),
    )
    model._pd = TRTLLMPD("prefill", "xavier")
    rpc = Mock()
    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=rpc))

    def generate(tokens, params, **kwargs):
        class StoppedRequest(Request):
            async def __anext__(self):
                if self.finished:
                    raise StopAsyncIteration
                self.finished = True
                self.outputs = [
                    SimpleNamespace(
                        index=0,
                        text="",
                        token_ids=[999],
                        finish_reason="stop",
                    )
                ]
                return self

        assert params.max_tokens == 1 and params.stop_token_ids == [999]
        assert not getattr(params, "ignore_eos", False)
        assert not getattr(params, "include_stop_str_in_output", False)
        return StoppedRequest()

    monkeypatch.setattr(model._engine, "generate_async", generate)
    handoff = dict(room=123, mode="host" if host else "gpu")
    result = await model.async_generate(
        "hi",
        dict(_pd_kv_transfer_params=dict(do_remote_decode=True, sglang_xavier=handoff)),
        "stop",
    )
    assert result["choices"][0]["text"] == ""
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["_pd_kv_transfer_params"] == dict(
        do_remote_prefill=True, sglang_xavier=handoff
    )
    rpc.assert_called_once_with("finalize", 123, 999)
    await model.release_pd_request("stop")
    assert not model._pd.active


@pytest.mark.asyncio
@pytest.mark.parametrize("token_ids", [None, [], [0]])
async def test_cross_publish_requires_a_raw_sample(checkpoint, monkeypatch, token_ids):
    pd = TRTLLMPD("prefill", "nixl")
    pd.client = SimpleNamespace(config=dict(heterogeneous=True))
    pd.active["r"] = dict(
        context=SimpleNamespace(room=123), handoff=dict(room=123, mode="gpu")
    )
    rpc = Mock()
    monkeypatch.setattr(gpu, "service", lambda: SimpleNamespace(rpc=rpc))
    output = SimpleNamespace(
        outputs=(
            []
            if token_ids is None
            else [SimpleNamespace(token_ids=token_ids, finish_reason="stop")]
        )
    )
    if token_ids:
        result = await pd.publish("r", [1, 2], output)
        assert result["sglang_xavier"] == pd.active["r"]["handoff"]
        rpc.assert_called_once_with("finalize", 123, 0)
    else:
        with pytest.raises(RuntimeError, match="Incomplete"):
            await pd.publish("r", [1, 2], output)
        rpc.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("heterogeneous", [False, True])
async def test_gpu_service_selects_actor_and_uid(
    checkpoint, monkeypatch, heterogeneous
):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    service = gpu.GPUService.__new__(gpu.GPUService)
    directory = AsyncMock()
    service.client = SimpleNamespace(
        config=dict(
            host="localhost", rank=3, max_seq_len=128, heterogeneous=heterogeneous
        ),
        contract=c,
        directory=AsyncMock(return_value=directory),
    )
    service.cache = Mock(
        device=SimpleNamespace(index=2), element_size=Mock(return_value=2)
    )
    service.cache.__getitem__ = Mock(
        return_value=SimpleNamespace(numel=lambda: 1024, element_size=lambda: 2)
    )
    descriptor = {}
    monkeypatch.setattr(gpu, "export_pool", Mock(return_value=descriptor))
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(
        "xinference.model.llm.xavier.transport.gpu_pool_options",
        lambda *args: dict(external_address="localhost:1234"),
    )
    pool = SimpleNamespace(start=AsyncMock(), external_address="localhost:1234")
    create_pool = AsyncMock(return_value=pool)
    actor = SimpleNamespace(address="localhost:2345")
    create = AsyncMock(return_value=actor)
    monkeypatch.setattr(gpu.xo, "create_actor_pool", create_pool)
    monkeypatch.setattr(gpu.xo, "create_actor", create)
    await service._start()
    cls = TRTCrossEngineGPUActor if heterogeneous else gpu.TRTGPUActor
    assert create.call_args.args == (cls, descriptor, 2, c, directory, 3)
    assert create.call_args.kwargs["uid"] == f"{cls.default_uid()}-3"
    sync.assert_called_once_with(service.cache.device)
    pool.start.assert_awaited_once()
    directory.register_peer.assert_awaited_once_with(3, actor.address)


@pytest.mark.asyncio
@pytest.mark.parametrize("heterogeneous", [False, True])
async def test_connector_computes_namespace_and_configures_contract(
    connector, checkpoint, monkeypatch, heterogeneous
):
    from ...xavier.contract import fingerprint_metadata

    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    monkeypatch.setenv(
        CONFIG_ENV,
        json.dumps(
            dict(
                address="directory",
                uid="cache",
                role="prefill",
                heterogeneous=heterogeneous,
                contract=c.to_dict(),
            )
        ),
    )
    directory = AsyncMock()
    monkeypatch.setattr(connector.xo, "actor_ref", AsyncMock(return_value=directory))
    client = await asyncio.to_thread(connector.XavierClient)
    try:
        expected = (
            c.fingerprint
            if heterogeneous
            else fingerprint_metadata(
                dict(
                    contract=c.to_dict(),
                    engine="trt-llm",
                    version="1.2.1",
                    layout="trt-block-layer-kv-packed-gpu-v1",
                )
            )
        )
        assert client.namespace == expected
        directory.configure.assert_awaited_once_with(
            expected, *([c.to_dict()] if heterogeneous else [])
        )
        assert client.capacity_bytes == 0
    finally:
        await asyncio.to_thread(client.close)


@pytest.mark.parametrize("cross", [False, True])
def test_cancelled_room_expiry_is_shared(checkpoint, monkeypatch, cross):
    c = build_pd_contract(checkpoint, logical_dtype="bfloat16")
    cls = TRTCrossEngineGPUActor if cross else gpu.TRTGPUActor
    actor = cls({}, 0, c, AsyncMock(), 1)
    monkeypatch.setattr(gpu.time, "monotonic", lambda: 10)
    actor._mark_aborted(123)
    assert actor.aborted == {123: 310}
    monkeypatch.setattr(gpu.time, "monotonic", lambda: 310)
    actor._prune_aborted()
    assert not actor.aborted


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_sglang_validates_dtype_before_building_its_contract(checkpoint, heterogeneous):
    # The same-engine branch builds an FP16 contract independently of this
    # private option, so its explicit guard is not redundant with the contract.
    with pytest.raises(ValueError, match="requires float32 KV"):
        configure_sglang(
            checkpoint, {}, dict(logical_dtype="float32", heterogeneous=heterogeneous)
        )
