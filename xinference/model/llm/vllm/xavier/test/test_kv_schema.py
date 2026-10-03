# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from ..snapshot import KVSnapshotStore
from ..transfer import TransferActor


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "mismatch", [None, "dtype", "shape", "layers", "block_size", "missing"]
)
def test_schema_mismatch_is_a_scheduler_miss(connector, monkeypatch, dtype, mismatch):
    import xoscar as xo

    schema = (16, {"layer": ((2, 4), dtype)})
    source_schema = (16, {"layer": ((2, 4), dtype)})
    if mismatch == "dtype":
        other = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
        source_schema = (16, {"layer": ((2, 4), other)})
    elif mismatch == "shape":
        source_schema = (16, {"layer": ((4, 2), dtype)})
    elif mismatch == "layers":
        source_schema = (16, {"other": ((2, 4), dtype)})
    elif mismatch == "block_size":
        source_schema = (32, schema[1])
    elif mismatch == "missing":
        source_schema = None
    store = KVSnapshotStore(1)
    store.stage("layer", [4], torch.zeros(1, 2, 4, dtype=dtype))
    store.publish([4], {"layer"})
    producer = SimpleNamespace(_snapshot_store=store, _kv_schema_v1=source_schema)
    remote = SimpleNamespace(
        reserve_blocks_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.reserve_blocks_v1(producer, *args)
        ),
        release_blocks_v1=AsyncMock(side_effect=store.release),
    )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=remote))
    consumer = SimpleNamespace(_kv_schema_v1=schema, _world_addresses=["a", "b", "c"])

    async def reserve(*args):
        return await TransferActor.reserve_remote_blocks_v1(consumer, *args)

    transfer = SimpleNamespace(reserve_remote_blocks_v1=AsyncMock(side_effect=reserve))
    connector._get_transfer_ref = AsyncMock(return_value=transfer)
    connector._query_remote_blocks = AsyncMock(return_value={2: {(1, 4, 0)}})
    request = SimpleNamespace(request_id="r", prompt_token_ids=list(range(17)))
    assert connector.get_num_new_matched_tokens(request, 0) == (
        16 if mismatch is None else 0,
        False,
    )
    assert bool(store.leases) == (mismatch is None)
    assert bool(connector._requests_need_load) == (mismatch is None)
    assert bool(connector._leased_requests) == (mismatch is None)


def test_worker_registers_normalized_resolved_schema(connector):
    actor = SimpleNamespace()
    ref = SimpleNamespace(
        configure_kv_schema_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.configure_kv_schema_v1(actor, *args)
        )
    )
    connector._get_transfer_ref = AsyncMock(return_value=ref)
    connector.register_kv_caches(
        {"layer": torch.zeros(2, 8, 16, 2, 4, dtype=torch.bfloat16)}
    )
    assert actor._kv_schema_v1 == (16, {"layer": ((2, 16, 2, 4), torch.bfloat16)})
    TransferActor.configure_kv_schema_v1(actor, *actor._kv_schema_v1)
    with pytest.raises(ValueError, match="schema changed"):
        TransferActor.configure_kv_schema_v1(
            actor, 16, {"layer": ((2,), torch.float16)}
        )


@pytest.mark.asyncio
async def test_unregistered_consumer_returns_miss_without_rpc(monkeypatch):
    import xoscar as xo

    actor_ref = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", actor_ref)
    assert not await TransferActor.reserve_remote_blocks_v1(
        SimpleNamespace(), "r", {1: {1: 0}}
    )
    actor_ref.assert_not_awaited()
