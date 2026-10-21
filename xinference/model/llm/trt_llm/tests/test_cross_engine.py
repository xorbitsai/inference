# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ...sglang.xavier.config import configure_xavier as configure_sglang
from ...xavier.backends.torch.pd import canonical_vllm_views
from ...xavier.contract import KVCacheContract
from ...xavier.pd_contract import build_pd_contract
from ..cross_engine import TRTCrossEngineGPUActor, canonical_trt_views
from ..pd import TRTLLMPD, XavierRequest
from ..xavier import CONFIG_ENV, configure_xavier
from .test_xavier import checkpoint as checkpoint


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
        torch.arange(3 * c.num_layers * c.layer_nbytes // 2, dtype=torch.float32)
        .to(torch.bfloat16)
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
            assert torch.equal(views[key], native[:, kind].transpose(1, 2))
            assert torch.equal(views[key], vllm[key])
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
