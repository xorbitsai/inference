# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import asyncio
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ....xavier.block_tracker import BlockTracker
from ....xavier.cache_lifecycle import CACHE_SESSION_ARG
from ....xavier.tests.test_cache_lifecycle import _pool


@pytest.fixture
def hybrid(monkeypatch, connector_module):
    """Run the real hybrid connector and tracker; stub the vLLM import surface."""
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="0.32.0"))
    base = sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    monkeypatch.setattr(base, "KVConnectorRole", SimpleNamespace(SCHEDULER="scheduler"))
    monkeypatch.setattr(base, "KVConnectorTransferResults", object, raising=False)
    for name, values in {
        "vllm.distributed.kv_transfer.kv_connector.v1.multi_connector": {
            "MultiConnector": object
        },
        "vllm.model_executor.layers.mamba.mamba_utils": {
            "get_conv_state_layout": lambda: "test"
        },
        "vllm.v1.kv_cache_interface": {
            "FullAttentionSpec": type("Attention", (), {}),
            "MambaSpec": type("Mamba", (), {}),
        },
        "vllm.v1.metrics.cache_hit_source": {
            "CacheHitSource": SimpleNamespace(P2P="p2p")
        },
        "vllm.v1.request": {
            "RequestStatus": SimpleNamespace(
                FINISHED_STOPPED="stopped", FINISHED_LENGTH_CAPPED="length"
            )
        },
    }.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**values))
    monkeypatch.setitem(
        sys.modules, "xinference.model.llm.vllm.xavier.v1_connector", connector_module
    )
    name = "xinference.model.llm.vllm.xavier._test_hybrid_lifecycle"
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "hybrid_gpu.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    connector = module.XavierHybridGPUConnector.__new__(module.XavierHybridGPUConnector)
    connector._offers = OrderedDict()
    connector._offers_by_block = {}
    connector._session_admissions = set()
    connector._session_phases = {}
    connector._attempted = set()
    connector._requests_need_load = {}
    connector._rank = 1
    connector._block_size = 16
    connector._identity = "model"
    connector._recurrent_groups = {1}
    connector._layer_group_ids = {"attention": 0, "recurrent": 1}
    connector._call = asyncio.run
    pool = _pool()
    pool.hash_block_size = 16
    pool.touch = Mock()
    pool.unpin_blocks = Mock()
    groups = [
        pool.blocks[:2],
        [SimpleNamespace(block_id=7, is_null=True), pool.blocks[2]],
    ]
    connector._kv_cache_manager = SimpleNamespace(
        block_pool=pool,
        coordinator=SimpleNamespace(
            find_longest_cache_hit=Mock(return_value=(groups, 32, None))
        ),
    )
    tracker = BlockTracker()
    tracker_ref = SimpleNamespace(
        query_gpu_prefix=AsyncMock(side_effect=tracker.query_gpu_prefix),
        publish_gpu_prefix=AsyncMock(side_effect=tracker.publish_gpu_prefix),
        remove_gpu_prefix=AsyncMock(side_effect=tracker.remove_gpu_prefix),
        get_cache_sessions=AsyncMock(side_effect=tracker.get_cache_sessions),
    )
    connector._get_tracker_ref = AsyncMock(return_value=tracker_ref)
    transfer = SimpleNamespace(
        register_shared_prefix_v1=AsyncMock(), revoke_shared_prefix_v1=AsyncMock()
    )
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    return module, connector, tracker, tracker_ref, transfer


def _request(session="agent", request_id="request", tokens=33):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=tokens,
        num_prompt_tokens=tokens,
        num_computed_tokens=32,
        block_hashes=[b"first", b"last"],
        skip_reading_prefix_cache=False,
        sampling_params=SimpleNamespace(
            extra_args={CACHE_SESSION_ARG: {"id": session}}
        ),
        status="stopped",
    )


def test_publish_tags_ordered_blocks_and_replaces_same_sessions_old_offer(hybrid):
    _, connector, tracker, _, transfer = hybrid
    request = _request()
    connector._publish(request)
    ticket = next(iter(connector._offers))
    assert connector._offers[ticket].session_id == "agent"
    assert connector._offers[ticket].block_ids == [0, 1, 2]
    assert tracker.query_gpu_prefix(2, [("model", b"last")])[1] == ticket
    request.block_hashes[-1] = b"new"
    connector._publish(request)
    transfer.revoke_shared_prefix_v1.assert_awaited_once_with(ticket)
    assert len(connector._offers) == 1
    assert tracker.query_gpu_prefix(2, [("model", b"last")]) is None
    assert tracker.query_gpu_prefix(2, [("model", b"new")]) is not None
    assert all(ticket not in tickets for tickets in connector._offers_by_block.values())


@pytest.mark.parametrize("tokens", [2, 33])
def test_admission_orders_once_and_combines_phase_lookup_with_prefix_rpc(
    hybrid, tokens
):
    _, connector, tracker, rpc, _ = hybrid
    connector._publish(_request())
    tracker.update_cache_session({"id": "agent", "revision": 1, "phase": "waiting"})
    request = _request("other", tokens=tokens)
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert [
        b.block_id for b in connector._kv_cache_manager.block_pool.free_block_queue
    ] == [3, 4, 5, 6, 7, 2, 1, 0]
    assert rpc.query_gpu_prefix.await_count == 1
    assert rpc.query_gpu_prefix.await_args.kwargs == {"session_ids": ["agent"]}
    rpc.get_cache_sessions.assert_not_awaited()
    connector.get_num_new_matched_tokens(request, 0)
    assert rpc.query_gpu_prefix.await_count == (2 if tokens == 33 else 1)
    connector._publish = Mock()
    connector.request_finished_all_groups(request, [])
    assert request.request_id not in connector._session_admissions


def test_waiting_offers_do_not_displace_other_sessions_slots(hybrid, monkeypatch):
    module, connector, tracker, _, transfer = hybrid
    monkeypatch.setattr(module, "_MAX_OFFERS", 3)
    tickets = {}
    for session, phase in [("a", "waiting"), ("b", "active"), ("c", "waiting")]:
        connector._publish(_request(session))
        tickets[session] = next(reversed(connector._offers))
        tracker.update_cache_session({"id": session, "revision": 1, "phase": phase})
    connector.get_num_new_matched_tokens(_request("d"), 0)
    connector._publish(_request("d"))
    transfer.revoke_shared_prefix_v1.assert_awaited_once_with(tickets["b"])
    assert {offer.session_id for offer in connector._offers.values()} == {"a", "c", "d"}


def test_ordering_failure_cannot_abort_model_admission(hybrid):
    _, connector, tracker, rpc, _ = hybrid
    connector._publish(_request())
    tracker.update_cache_session({"id": "agent", "revision": 1, "phase": "waiting"})
    pool = connector._kv_cache_manager.block_pool
    before = list(pool.free_block_queue)
    pool.free_block_queue.remove = Mock(side_effect=RuntimeError("invalid block"))
    assert connector.get_num_new_matched_tokens(_request("other"), 0) == (0, False)
    assert pool.free_block_queue == before
    assert rpc.query_gpu_prefix.await_count == 1


def test_tracker_combined_lookup_keeps_legacy_return_shape():
    tracker = BlockTracker()
    tracker.publish_gpu_prefix(1, "hash", "ticket", 32, [[0, 1]])
    tracker.update_cache_session({"id": "agent", "revision": 1, "phase": "waiting"})
    offer = (1, "ticket", 32, [[0, 1]])
    assert tracker.query_gpu_prefix(2, ["hash"]) == offer
    assert tracker.query_gpu_prefix(2, ["hash"], ["agent"]) == (
        offer,
        {"agent": "waiting"},
    )
    assert tracker.query_gpu_prefix(1, ["hash"], ["agent"]) == (
        None,
        {"agent": "waiting"},
    )
