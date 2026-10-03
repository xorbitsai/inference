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

    schema = (16, {"layer": ((2, 4), dtype)}, "auto")
    source_schema = (16, {"layer": ((2, 4), dtype)}, "auto")
    if mismatch == "dtype":
        other = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
        source_schema = (16, {"layer": ((2, 4), other)}, "auto")
    elif mismatch == "shape":
        source_schema = (16, {"layer": ((4, 2), dtype)}, "auto")
    elif mismatch == "layers":
        source_schema = (16, {"other": ((2, 4), dtype)}, "auto")
    elif mismatch == "block_size":
        source_schema = (32, schema[1], "auto")
    elif mismatch == "missing":
        source_schema = None
    store = KVSnapshotStore(1)
    store.stage("layer", [4], torch.zeros(1, 2, 4, dtype=dtype))
    store.publish([4], {"layer"})
    producer = SimpleNamespace(
        _snapshot_store=store,
        _kv_schema_v1=source_schema,
        _schema_mismatch_warnings=set(),
    )
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


def test_worker_schema_handshake_defers_actor_rpc(connector, monkeypatch):
    import xoscar as xo

    actor = SimpleNamespace(_kv_schema_v1=None)
    ref = SimpleNamespace(
        configure_snapshots_v1=AsyncMock(),
        configure_kv_schema_v1=AsyncMock(
            side_effect=lambda *args: TransferActor.configure_kv_schema_v1(actor, *args)
        ),
    )
    actor_ref = AsyncMock(return_value=ref)
    monkeypatch.setattr(xo, "actor_ref", actor_ref)
    connector.register_kv_caches(
        {"layer": torch.zeros(2, 8, 16, 2, 4, dtype=torch.bfloat16)}
    )
    actor_ref.assert_not_awaited()
    metadata = connector.get_handshake_metadata()
    connector._kv_schema = None
    connector.set_xfer_handshake_metadata({0: metadata})
    connector._call(connector._get_transfer_ref())
    assert actor._kv_schema_v1 == (
        16,
        {"layer": ((2, 16, 2, 4), torch.bfloat16)},
        "auto",
    )
    TransferActor.configure_kv_schema_v1(actor, *actor._kv_schema_v1)
    with pytest.raises(ValueError, match="schema changed"):
        TransferActor.configure_kv_schema_v1(
            actor, 16, {"layer": ((2,), torch.float16)}, "auto"
        )


@pytest.mark.asyncio
async def test_unregistered_consumer_returns_miss_without_rpc(monkeypatch):
    import xoscar as xo

    actor_ref = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", actor_ref)
    assert not await TransferActor.reserve_remote_blocks_v1(
        SimpleNamespace(_kv_schema_v1=None), "r", {1: {1: 0}}
    )
    actor_ref.assert_not_awaited()


def test_fp8_encoding_is_part_of_worker_schema(connector_module, connector_config):
    schemas = []
    for dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        connector_config.cache_config.cache_dtype = dtype
        instance = connector_module.XavierConnector(
            connector_config,
            None,
            SimpleNamespace(num_blocks=8, kv_cache_groups=[]),
        )
        try:
            instance.register_kv_caches({"layer": torch.zeros(8, 2, dtype=torch.uint8)})
            schemas.append(instance.get_handshake_metadata())
        finally:
            instance.shutdown()
    assert len({schema.cache_dtype for schema in schemas}) == 3
    assert all(schema.layers == schemas[0].layers for schema in schemas)
    assert schemas[0] != schemas[1] != schemas[2]


@pytest.mark.parametrize("metadata", [{}, {0: None, 1: None}])
def test_handshake_rejects_non_single_worker(connector, metadata):
    with pytest.raises(ValueError, match="one worker"):
        connector.set_xfer_handshake_metadata(metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["configure_snapshots_v1", "configure_kv_schema_v1"]
)
async def test_failed_configuration_is_retried(connector, monkeypatch, failure):
    import xoscar as xo

    connector.register_kv_caches({"layer": torch.zeros(8, 2)})
    ref = SimpleNamespace(
        configure_snapshots_v1=AsyncMock(), configure_kv_schema_v1=AsyncMock()
    )
    getattr(ref, failure).side_effect = [RuntimeError("transient"), None]
    lookup = AsyncMock(return_value=ref)
    monkeypatch.setattr(xo, "actor_ref", lookup)
    with pytest.raises(RuntimeError, match="transient"):
        await connector._get_transfer_ref()
    assert connector._transfer_ref is None
    assert await connector._get_transfer_ref() is ref
    assert await connector._get_transfer_ref() is ref
    assert lookup.await_count == 2
    ref.configure_kv_schema_v1.assert_awaited_with(
        16, {"layer": ((2,), torch.float32)}, "auto"
    )


@pytest.mark.parametrize(
    "field", ["block_size", "layer set", "shape", "dtype", "cache_dtype"]
)
def test_schema_warning_once_per_peer_and_field(caplog, field):
    schema = (16, {"layer": ((2,), torch.uint8)}, "fp8_e4m3")
    changes = {
        "block_size": (32, schema[1], schema[2]),
        "layer set": (16, {"other": ((2,), torch.uint8)}, schema[2]),
        "shape": (16, {"layer": ((4,), torch.uint8)}, schema[2]),
        "dtype": (16, {"layer": ((2,), torch.float32)}, schema[2]),
        "cache_dtype": (16, schema[1], "fp8_e5m2"),
    }
    actor = SimpleNamespace(
        _kv_schema_v1=schema,
        _schema_mismatch_warnings=set(),
        _snapshot_store=KVSnapshotStore(1),
    )
    for lease in ["1:first", "1:second", "2:third"]:
        assert not TransferActor.reserve_blocks_v1(actor, lease, [1], changes[field])
    assert len(caplog.records) == 2
    assert all(field in record.message for record in caplog.records)
    assert (
        "peer 1" in caplog.records[0].message and "peer 2" in caplog.records[1].message
    )
    assert not actor._snapshot_store.leases


@pytest.mark.asyncio
async def test_real_second_peer_mismatch_releases_first_lease(monkeypatch):
    import xoscar as xo

    schema = (16, {"layer": ((2,), torch.uint8)}, "fp8_e4m3")
    actors, refs = [], []
    for encoding in ["fp8_e4m3", "fp8_e5m2"]:
        store = KVSnapshotStore(1)
        store.stage("layer", [1], torch.zeros(1, 2, dtype=torch.uint8))
        store.publish([1], {"layer"})
        actor = SimpleNamespace(
            _snapshot_store=store,
            _kv_schema_v1=(16, schema[1], encoding),
            _schema_mismatch_warnings=set(),
        )
        actors.append(actor)
        refs.append(
            SimpleNamespace(
                reserve_blocks_v1=AsyncMock(
                    side_effect=lambda *args, actor=actor: TransferActor.reserve_blocks_v1(
                        actor, *args
                    )
                ),
                release_blocks_v1=AsyncMock(side_effect=store.release),
            )
        )
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(side_effect=refs))
    consumer = SimpleNamespace(_kv_schema_v1=schema, _world_addresses=["a", "b"])
    assert not await TransferActor.reserve_remote_blocks_v1(
        consumer, "2:r", {0: {1: 0}, 1: {1: 1}}
    )
    for ref, actor in zip(refs, actors):
        ref.reserve_blocks_v1.assert_awaited_once_with("2:r", [1], schema)
        ref.release_blocks_v1.assert_awaited_once_with("2:r")
        assert not actor._snapshot_store.leases
