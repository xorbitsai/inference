# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import json
import struct
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from ...sglang.xavier.directory import XavierPDDirectory
from ..backends.torch.pd import (
    CrossEngineGPUActor,
    canonical_vllm_views,
    sglang_first_token_payload,
)
from ..pd_contract import build_pd_contract, prompt_digest


def cpu_actor(kv_contract, directory, rank):
    actor = CrossEngineGPUActor(
        SimpleNamespace(aux_item_lens=[]),
        replace(kv_contract, block_size=64),
        directory,
        "namespace",
        rank,
    )
    actor.transfer = SimpleNamespace(
        register_direct=Mock(),
        release_direct=Mock(),
        direct_requests={},
        poll_direct=Mock(return_value=set()),
        send_lock=asyncio.Lock(),
        metrics={},
    )
    actor.chunk_capacity = 1
    return actor


@pytest.mark.parametrize("layout", ["2NTHD", "N2THD", "BHN2D", "BHN2D_NHD"])
def test_vllm_canonical_views_preserve_engine_storage(kv_contract, layout):
    contract = replace(kv_contract, num_layers=2)
    tensors = {
        f"model.layers.{i}.self_attn.attn": torch.arange(
            7 * 2 * 4 * 2 * 4, dtype=torch.float16
        ).reshape(7, 2, 4, 2, 4)
        + i * 1000
        for i in range(2)
    }
    original = {name: tensor.clone() for name, tensor in tensors.items()}
    if layout == "2NTHD":
        tensors = {name: tensor.movedim(1, 0) for name, tensor in tensors.items()}
    if layout.startswith("BHN2D"):
        tensors = {
            name: tensor.permute(0, 3, 2, 1, 4).reshape(7, 2, 4, 8)
            for name, tensor in tensors.items()
        }
        if layout == "BHN2D_NHD":
            tensors = {
                name: tensor.transpose(1, 2).contiguous().transpose(1, 2)
                for name, tensor in tensors.items()
            }
    views = canonical_vllm_views(tensors, 7, contract)
    for kind in range(2):
        for index in range(2):
            name = f"model.layers.{index}.self_attn.attn"
            view = views[str(index + kind * 2)]
            assert torch.equal(view, original[name][:, kind])
            view[3, 2] = -1
            native = tensors[name] if layout == "N2THD" else tensors[name].movedim(1, 0)
            if layout.startswith("BHN2D"):
                assert (tensors[name][3, :, 2, kind * 4 : (kind + 1) * 4] == -1).all()
            else:
                assert (native[3, kind, 2] == -1).all()


def test_canonical_views_reject_missing_layer_and_dtype(kv_contract):
    contract = replace(kv_contract, num_layers=1)
    with pytest.raises(ValueError, match="Incomplete"):
        canonical_vllm_views({}, 7, contract)
    with pytest.raises(ValueError, match="layout"):
        canonical_vllm_views(
            {"model.layers.0.self_attn.attn": torch.zeros(7, 2, 4, 2, 4)}, 7, contract
        )


def test_first_token_metadata_and_room_marker():
    sizes = [64, 64, 64, 64, 512, 512, 64, 128, 1024, 64]
    payload = sglang_first_token_payload(sizes, 123, 42, 193)
    assert list(map(len, payload)) == sizes
    assert struct.unpack_from("<i", payload[0])[0] == 42
    assert struct.unpack_from("<i", payload[1])[0] == 193
    assert struct.unpack_from("<Q", payload[-1])[0] == 123
    with pytest.raises(ValueError, match="layout"):
        sglang_first_token_payload(sizes + [64], 123, 42, 193)


def test_directory_proves_imported_tokens_and_single_completion(kv_contract):
    directory = XavierPDDirectory()
    directory.configure(kv_contract.fingerprint)
    for rank, role in enumerate(("prefill", "decode")):
        directory.prepare(
            1,
            kv_contract.fingerprint,
            prompt_digest([1, 2]),
            role,
            rank=rank,
            timeout=45,
            prompt_tokens=2,
        )
    assert directory.rooms[1]["ranks"] == {"prefill": 0, "decode": 1}
    assert directory.rooms[1]["timeout"] == 45
    assert directory.request_info(1) == dict(prompt_tokens=2)
    directory.complete(1, 4096, 1)
    directory.complete(1, 4096, 1)
    stats = directory.get_stats()
    assert stats["imported_tokens"] == 1
    assert stats["gpu_bytes"] == 4096
    assert stats["completed_requests"] == 1
    directory.release(1, "decode")
    assert directory.check(1)
    directory.release(1, "prefill")
    assert directory.get_stats()["active_handoffs"] == 0


def test_contract_fingerprints_actual_assets_and_context(tmp_path):
    config = dict(
        model_type="qwen2",
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        hidden_size=8,
        max_position_embeddings=4096,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    initial = build_pd_contract(str(tmp_path))
    initial.require_match(build_pd_contract(str(tmp_path), 4096))
    with pytest.raises(ValueError, match="position_fingerprint"):
        initial.require_match(build_pd_contract(str(tmp_path), 8192))
    (tmp_path / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="weights_fingerprint"):
        initial.require_match(build_pd_contract(str(tmp_path)))


@pytest.mark.asyncio
async def test_late_source_publication_does_not_leak_a_room():
    directory = AsyncMock()
    directory.publish_source.side_effect = ValueError("cancelled room")
    actor = CrossEngineGPUActor(None, None, directory, "namespace", 1)
    with pytest.raises(ValueError, match="cancelled room"):
        await actor.open(123)
    assert not actor.rooms


@pytest.mark.asyncio
async def test_aborted_source_cannot_register_late_gpu_chunks(kv_contract):
    actor = cpu_actor(kv_contract, AsyncMock(), 1)
    await actor.open(123)
    actor.init(123, 3, 0)
    await actor.add_chunk(123, [1])
    await actor.abort(123)
    assert not actor.rooms
    with pytest.raises(RuntimeError, match="cancelled"):
        actor.init(123, 3, 0)
    with pytest.raises(RuntimeError, match="cancelled"):
        await actor.add_chunk(123, [1, 2, 3])
    with pytest.raises(RuntimeError, match="cancelled"):
        actor.done(123)
    with pytest.raises(RuntimeError, match="cancelled"):
        actor.chunk(123, 0)
    with pytest.raises(RuntimeError, match="cancelled"):
        await actor.wait_chunk(123, 0)
    with pytest.raises(RuntimeError, match="cancelled"):
        await actor.wait_done(123)
    assert not actor.rooms
    actor.transfer.register_direct.assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_abort_clears_room_and_wakes_existing_waiter():
    actor = CrossEngineGPUActor(None, None, None, "namespace", 1)
    actor.transfer = SimpleNamespace(send_lock=asyncio.Lock(), release_direct=Mock())
    state = actor.rooms[123] = {
        "completed": asyncio.Event(),
        "changed": asyncio.Event(),
        "deadline": asyncio.get_running_loop().time() + 1,
        "chunks": [],
    }
    waiter = asyncio.create_task(actor.wait_done(123))
    await asyncio.sleep(0)
    await actor.abort(123)
    with pytest.raises(RuntimeError, match="cancelled"):
        await waiter
    assert not actor.rooms and state["aborted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_tokens", [64, 65, 128, 129, 193])
@pytest.mark.parametrize("split_final", [False, True])
@pytest.mark.parametrize("full_allocation", [False, True])
async def test_vllm_imports_only_allocated_prefix_pages(
    kv_contract, monkeypatch, prompt_tokens, split_final, full_allocation
):
    directory = AsyncMock()
    directory.request_info.return_value = dict(prompt_tokens=prompt_tokens)
    directory.wait_source.return_value = dict(address="source", rank=1)
    source = cpu_actor(kv_contract, directory, 1)
    destination = cpu_actor(kv_contract, directory, 2)
    source_count = (prompt_tokens + 63) // 64
    target_count = (prompt_tokens - 1 + 63) // 64
    source.caches = {"K": torch.arange(7 * 64, dtype=torch.float16).reshape(7, 64)}
    destination.caches = {"K": torch.zeros(7, 64, dtype=torch.float16)}
    allocated_count = source_count if full_allocation else target_count
    pages, targets = [5, 1, 3, 0][:source_count], [2, 6, 4, 0][:allocated_count]
    await source.open(123)
    source.init(123, len(pages), 0)
    if split_final and len(pages) > 1:
        await source.add_chunk(123, pages[:-1])
        await source.add_chunk(123, pages[-1:], [b"metadata"])
    else:
        await source.add_chunk(123, pages, [b"metadata"])
    peer = SimpleNamespace(
        wait_chunk=source.wait_chunk,
        release_chunk=AsyncMock(side_effect=source.release_chunk),
    )
    monkeypatch.setattr("xoscar.actor_ref", AsyncMock(return_value=peer))

    async def load(requests, tickets):
        mapping = requests[0][1]["K"]
        for original, target in mapping.items():
            destination.caches["K"][target].copy_(source.caches["K"][original])
        source.release_chunk(tickets[0])
        return set()

    async def run(fn, *args):
        return await fn(*args)

    destination.transfer.load_direct, destination.transfer.run = load, run
    assert await destination.receive(123, targets, 0) == set()
    assert torch.equal(
        destination.caches["K"][targets[:target_count]],
        source.caches["K"][pages[:target_count]],
    )
    assert (destination.caches["K"][targets[target_count:]] == 0).all()
    assert source.rooms[123]["completed"].is_set()
    assert source.transfer.release_direct.call_count == len(source.rooms[123]["chunks"])
    directory.complete.assert_awaited_once_with(
        123, target_count * 64 * 2, prompt_tokens - 1
    )
    directory.source.assert_not_awaited()


@pytest.mark.asyncio
async def test_progress_and_configured_timeout_control_chunk_wait(
    kv_contract, monkeypatch
):
    monkeypatch.setenv("XINFERENCE_SGLANG_XAVIER_TRANSFER_TIMEOUT", "1000")
    actor = cpu_actor(kv_contract, AsyncMock(), 1)
    await actor.open(123)
    actor.init(123, 2, 0)
    waiting = asyncio.create_task(actor.wait_chunk(123, 0))
    await asyncio.sleep(0)
    await actor.add_chunk(123, [1])
    assert (await asyncio.wait_for(waiting, 1))["pages"] == [1]
    actor.transfer.register_direct.assert_called_once_with(
        "123:0", "123:0", [1], lease_timeout=1000
    )
    actor.rooms[123]["deadline"] = 0
    with pytest.raises(TimeoutError, match="transfer timed out"):
        await actor.wait_chunk(123, 1)
    with pytest.raises(TimeoutError, match="transfer timed out"):
        await actor.wait_done(123)
    assert not actor.rooms


@pytest.mark.asyncio
async def test_wait_done_honors_progress_refreshed_deadline(kv_contract, monkeypatch):
    actor = cpu_actor(kv_contract, AsyncMock(), 1)
    await actor.open(123)
    actor.init(123, 1, 0)
    state = actor.rooms[123]
    state["deadline"] = 0
    original = asyncio.wait_for
    calls = 0

    async def refresh(awaitable, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            awaitable.close()
            await actor.add_chunk(123, [1], [b"metadata"])
            raise asyncio.TimeoutError()
        actor.release_chunk("123:0")
        return await original(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", refresh)
    assert await actor.wait_done(123)
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_producer_poll_keeps_finished_ids_when_release_rpc_fails(
    kv_contract, expired
):
    directory = AsyncMock()
    directory.release.side_effect = OSError("supervisor unavailable")
    actor = cpu_actor(kv_contract, directory, 1)
    await actor.register_prefill(123, "request", [1], 42, 40)
    if not expired:
        actor.transfer.direct_requests["123:0"] = object()
        actor.release_chunk("123:0")
    actor.transfer.poll_direct.return_value = {"request"}
    assert await actor.poll_direct_gpu_v1() == {"request"}
    assert not actor.rooms
    directory.release.assert_awaited_once_with(123, "prefill")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens,error",
    [
        ([1], "two prompt tokens"),
        ([1, 2], "prompt mismatch"),
        ([1, 2, 3], "room limit"),
    ],
)
@pytest.mark.parametrize("host", [False, True])
async def test_request_preparation_errors_stay_outside_engine(
    kv_contract, monkeypatch, tokens, error, host
):
    import xoscar as xo

    from ..pd_contract import prepare_pd_request

    directory = AsyncMock()
    directory.prepare.side_effect = ValueError(error)
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=directory))
    config = dict(
        role="decode",
        rank=2,
        address="supervisor",
        uid="directory",
        contract=kv_contract.to_dict(),
        host_handoff=host,
    )
    params = dict(sglang_xavier=dict(room=123, mode="host" if host else "gpu"))
    with pytest.raises(ValueError, match=error):
        await prepare_pd_request(config, tokens, params)
    assert "xavier_prompt_digest" not in params
    if len(tokens) < 2:
        directory.prepare.assert_not_awaited()
    directory.prepare.side_effect = None
    await prepare_pd_request(config, [1, 2], params)
    assert params["xavier_prompt_digest"] == prompt_digest([1, 2])


@pytest.mark.asyncio
@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_request_preparation_rejects_wrong_transport_before_rpc(
    kv_contract, monkeypatch, host, role
):
    import xoscar as xo

    from ..pd_contract import prepare_pd_request

    lookup = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", lookup)
    config = dict(
        role=role,
        rank=1,
        address="supervisor",
        uid="directory",
        contract=kv_contract.to_dict(),
        host_handoff=host,
    )
    params = dict(sglang_xavier=dict(room=123, mode="gpu" if host else "host"))
    with pytest.raises(ValueError, match="handoff room"):
        await prepare_pd_request(config, [1, 2], params)
    lookup.assert_not_awaited()
    assert "xavier_prompt_digest" not in params


@pytest.mark.parametrize(
    "field", ["weights_fingerprint", "tokenizer_fingerprint", "position_fingerprint"]
)
def test_cross_engine_namespace_mismatch_names_contract_field(kv_contract, field):
    directory = XavierPDDirectory()
    directory.configure(kv_contract.fingerprint, kv_contract.to_dict())
    other = replace(kv_contract, **{field: "f" * 64})
    with pytest.raises(ValueError, match=field):
        directory.configure(other.fingerprint, other.to_dict())
