# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Hybrid attention needs position-correct state and independent cache groups."""
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from .test_direct_handoff import direct_runtime


@pytest.fixture
def recurrent(connector_module, connector_config):
    connector_config.kv_transfer_config.get_from_extra_config = lambda *a: {
        "rank": 0,
        "role": "prefill",
        "gpu_cache_bytes": 1024,
    }
    caches = SimpleNamespace(
        num_blocks=8,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["linear"],
                kv_cache_spec=SimpleNamespace(mamba_cache_mode="none"),
            ),
            SimpleNamespace(layer_names=["attention"], kv_cache_spec=SimpleNamespace()),
        ],
    )
    instance = connector_module.XavierConnector(connector_config, None, caches)
    yield instance
    instance.shutdown()


def test_prefill_state_matches_decoder_position(recurrent, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    actor = SimpleNamespace(register_direct_gpu_v1=AsyncMock())
    recurrent._get_transfer_ref = AsyncMock(return_value=actor)
    request = SimpleNamespace(
        request_id="p",
        prompt_token_ids=list(range(34)),
        _all_token_ids=list(range(34)),
        num_prompt_tokens=34,
        max_tokens=10,
        status="finished",
        kv_transfer_params={"do_remote_decode": True},
    )
    assert recurrent.get_num_new_matched_tokens(request, 0) == (0, False)
    assert request.num_prompt_tokens == 33 and request.max_tokens == 1
    assert request.prompt_token_ids == request._all_token_ids == list(range(33))
    recurrent.get_num_new_matched_tokens(request, 16)
    assert request.num_prompt_tokens == 33  # Retry must not truncate twice.
    retained, meta = recurrent.request_finished_all_groups(request, ([7], [2, 3, 4]))
    assert retained and not recurrent._history_enabled
    handoff = meta["xavier_direct"]
    assert handoff["tokens"] == 33
    assert handoff["blocks_by_group"] == [[7], [2, 3, 4]]
    actor.register_direct_gpu_v1.assert_awaited_once_with(
        handoff["ticket"], "p", {"linear": [7], "attention": [2, 3, 4]}
    )


def test_decoder_maps_recurrent_state_separately(recurrent):
    recurrent._is_producer = False
    recurrent._get_transfer_ref = AsyncMock(
        return_value=SimpleNamespace(
            claim_remote_direct_gpu_v1=AsyncMock(return_value=True)
        )
    )
    request = SimpleNamespace(
        request_id="d",
        num_computed_tokens=16,
        kv_transfer_params={
            "xavier_direct": {
                "rank": 0,
                "ticket": "t",
                "tokens": 33,
                "blocks": [2, 3, 4],
                "blocks_by_group": [[7], [2, 3, 4]],
            }
        },
    )
    assert recurrent.get_num_new_matched_tokens(request, 16) == (17, True)
    recurrent.update_state_after_alloc(
        request, SimpleNamespace(get_block_ids=lambda: ([6], [1, 4, 5])), 17
    )
    load = recurrent._requests_need_load["d"]
    assert load.local_transfers_by_group == {0: {0: {7: 6}}, 1: {0: {3: 4, 4: 5}}}
    assert request.kv_transfer_params["do_remote_prefill"] is False


@pytest.mark.parametrize("kv_first", [False, True])
def test_logical_blocks_are_zero_copy_views(recurrent, kv_first):
    cache = torch.arange(2 * 16 * 4 * 2 * 3).reshape(2, 16, 4, 2, 3)
    if not kv_first:
        cache = cache.movedim(1, 0).contiguous()
    view = recurrent._cache_block_view("attention", cache)
    assert view.shape[0] == 8 and view.shape[1] == 2
    view[3].fill_(-1)
    physical = cache.movedim(1, 0) if kv_first else cache
    assert (physical[6:8] == -1).all() and (physical[5] != -1).all()
    state = torch.zeros(8, 2, 4)
    assert recurrent._cache_block_view("linear", state) is state


def test_ticket_owns_blocks_per_layer(monkeypatch):
    runtime = direct_runtime(monkeypatch)
    runtime.caches = {
        "attention": torch.zeros(8, 2),
        "linear#0": torch.zeros(8, 3),
        "linear#1": torch.zeros(8, 4),
    }
    runtime.register_direct("t", "p", {"attention": [1, 2], "linear": [7]})
    assert runtime.direct_requests["t"].layer_blocks == {
        "attention": {1, 2},
        "linear#0": {7},
        "linear#1": {7},
    }


@pytest.mark.asyncio
async def test_grouped_transfer_preserves_state_and_rejects_other_group_blocks(
    monkeypatch,
):
    import xoscar as xo

    from ..request_transfer import LayerRead
    from .test_direct_handoff import peer_for

    source, dest = direct_runtime(monkeypatch), direct_runtime(monkeypatch)
    source.caches = {
        "attention": torch.arange(16, dtype=torch.bfloat16).reshape(8, 2),
        "linear#0": torch.arange(16, dtype=torch.float32).reshape(8, 2),
        "linear#1": -torch.arange(16, dtype=torch.float32).reshape(8, 2),
    }
    dest.caches = {
        name: torch.zeros_like(cache) for name, cache in source.caches.items()
    }
    source.register_direct("t", "p", {"attention": [1, 2], "linear": [7]})
    # A block held for attention is not permission to read another request's SSM.
    with pytest.raises(ValueError, match="source layout or block IDs"):
        await source.send_direct(
            "t",
            [LayerRead("linear#1", [1], [4], (2,), torch.float32)],
            dest.recv_ref,
            16,
        )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer_for(source)))

    async def copy(buffers, refs):
        refs[0].copy_(buffers[0])

    monkeypatch.setattr(xo, "copy_to", copy)
    maps = {"attention": {1: 3, 2: 4}, "linear#0": {7: 6}, "linear#1": {7: 6}}
    assert await dest.run(dest.load_direct, [{0: maps}], ["t"]) == set()
    for name, mapping in maps.items():
        assert torch.equal(
            dest.caches[name][list(mapping.values())],
            source.caches[name][list(mapping)],
        )
    assert source.poll_direct() == {"p"}


def test_single_token_prompt_does_not_transfer_advanced_state(recurrent, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    request = SimpleNamespace(
        request_id="p",
        prompt_token_ids=[10],
        _all_token_ids=[10],
        num_prompt_tokens=1,
        status="finished",
        kv_transfer_params={"do_remote_decode": True},
    )
    assert recurrent.get_num_new_matched_tokens(request, 0) == (0, False)
    retained, metadata = recurrent.request_finished_all_groups(request, ([7], [2]))
    assert not retained and metadata["xavier_direct"]["tokens"] == 0
    assert request.prompt_token_ids == [10]


def test_recurrent_abort_before_allocation(recurrent, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.request",
        SimpleNamespace(RequestStatus=SimpleNamespace(FINISHED_ABORTED="aborted")),
    )
    request = SimpleNamespace(
        request_id="p", status="aborted", kv_transfer_params={"do_remote_decode": True}
    )
    assert recurrent.request_finished_all_groups(request, ()) == (False, None)
