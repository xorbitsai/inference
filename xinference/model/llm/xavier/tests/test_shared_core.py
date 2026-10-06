# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import xoscar as xo
from xoscar.backends.allocate_strategy import ProcessIndex

from ..block_tracker import BlockTracker
from ..collective_manager import CollectiveManager
from ..constants import DEFAULT_TRANSFER_ACTOR_UID


class AdapterPeerActor(xo.StatelessActor):
    def __init__(self):
        super().__init__()
        self.world_addresses = []

    def connect_full_mesh(self, prefix, world_addresses):
        self.world_addresses = world_addresses

    def get_world_addresses(self):
        return self.world_addresses


@pytest.mark.parametrize(
    "name,backend",
    [
        ("actor_loop", False),
        ("block_tracker", False),
        ("collective", False),
        ("profiling", False),
        ("utils", False),
        ("snapshot", True),
        ("tiered_snapshot", True),
        ("request_transfer", True),
        ("gpu_transfer", True),
        ("direct_handoff", True),
        ("direct_history", True),
    ],
)
def test_legacy_modules_share_the_implementation_and_state(name, backend):
    if backend:
        pytest.importorskip("torch")
    prefix = "xinference.model.llm.xavier"
    if backend:
        prefix += ".backends.torch"
    shared = importlib.import_module(f"{prefix}.{name}")
    legacy = importlib.import_module(f"xinference.model.llm.vllm.xavier.{name}")
    assert legacy is shared


def test_legacy_actor_and_transport_exports_remain_compatible():
    from ...vllm.xavier.block_tracker import VLLMBlockTracker
    from ...vllm.xavier.collective_manager import CollectiveManager as LegacyManager
    from ...vllm.xavier.transport import (
        XAVIER_CONNECTOR_MODULE,
        get_transport_host,
        validate_gpu_cache_budget,
    )
    from .. import transport

    assert VLLMBlockTracker is BlockTracker
    assert BlockTracker.default_uid() == "vllm-block-tracker-actor"
    assert LegacyManager is CollectiveManager
    assert get_transport_host is transport.get_transport_host
    assert validate_gpu_cache_budget is transport.validate_gpu_cache_budget
    assert XAVIER_CONNECTOR_MODULE == "xinference.model.llm.vllm.xavier.v1_connector"
    # Existing serialized actor class references use the old module and name.
    legacy_reference = (
        b"cxinference.model.llm.vllm.xavier.block_tracker\nVLLMBlockTracker\n."
    )
    assert pickle.loads(legacy_reference) is BlockTracker


def test_shared_coordination_imports_without_engine_or_device_dependencies():
    # A fresh interpreter exercises imports rather than reusing modules loaded
    # by other tests. Xinference's normal parent-package registration runs first.
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import importlib.util
        import sys
        import xinference.model.llm

        prefix = "xinference.model.llm.xavier"
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]

        original_import = builtins.__import__
        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            resolved = name
            if level:
                resolved = importlib.util.resolve_name(
                    "." * level + name, globals["__package__"]
                )
            if resolved.split(".")[0] in {"torch", "vllm", "sglang", "mlx"}:
                raise AssertionError("Unexpected dependency: " + resolved)
            if resolved.startswith("xinference.model.llm.vllm"):
                raise AssertionError("Unexpected engine adapter: " + resolved)
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        for name in (
            "actor_loop", "block_tracker", "collective", "collective_manager", "constants",
            "contract",
            "profiling", "transport", "utils",
        ):
            importlib.import_module(prefix + "." + name)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("rank", [0, 1])
async def test_default_coordinator_uid_matches_vllm_transfer_actors(monkeypatch, rank):
    pytest.importorskip("torch")
    from ...vllm.xavier.transfer import Rank0TransferActor, TransferActor

    manager = CollectiveManager("model")
    assert DEFAULT_TRANSFER_ACTOR_UID == "vllm-transfer-actor"
    assert (
        manager._transfer_actor_uid
        == TransferActor.default_uid()
        == Rank0TransferActor.default_uid()
        == DEFAULT_TRANSFER_ACTOR_UID
    )
    peer = SimpleNamespace(address="peer")
    actor_ref = AsyncMock(return_value=peer)
    monkeypatch.setattr(xo, "actor_ref", actor_ref)

    await manager.register_rank(rank, "peer")
    actor_class = Rank0TransferActor if rank == 0 else TransferActor
    actor_ref.assert_awaited_once_with(
        address="peer", uid=f"{actor_class.default_uid()}-{rank}"
    )
    assert manager._rank_to_ref[rank] is peer


@pytest.mark.asyncio
@pytest.mark.parametrize("uid", ["vllm-transfer-actor", "sglang-transfer-actor"])
async def test_coordinator_registers_and_recovers_adapter_peers(monkeypatch, uid):
    manager = CollectiveManager("model", transfer_actor_uid=uid)
    peer = SimpleNamespace(
        address="peer",
        connect_full_mesh=AsyncMock(),
        release_consumer_leases_v1=AsyncMock(),
    )
    tracker = SimpleNamespace(register_rank=AsyncMock(), unregister_rank=AsyncMock())
    manager._tracker_ref = tracker
    actor_ref = AsyncMock(return_value=peer)
    monkeypatch.setattr(xo, "actor_ref", actor_ref)

    await manager.register_rank(1, "peer", update=True)
    actor_ref.assert_awaited_once_with(address="peer", uid=f"{uid}-1")
    peer.connect_full_mesh.assert_awaited_once()
    assert peer.connect_full_mesh.await_args.args[1] == ["peer"]
    tracker.register_rank.assert_awaited_once_with(1)

    await manager.unregister_rank(2)
    tracker.unregister_rank.assert_awaited_once_with(2)
    peer.release_consumer_leases_v1.assert_awaited_once_with(2)


@pytest.mark.asyncio
async def test_shared_actors_register_and_recover_in_a_subprocess():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=1)
    async with pool:
        placement = {
            "address": pool.external_address,
            "allocate_strategy": ProcessIndex(1),
        }
        tracker = await xo.create_actor(
            BlockTracker, uid=f"{BlockTracker.default_uid()}-model", **placement
        )
        manager = await xo.create_actor(
            CollectiveManager,
            model_uid="model",
            transfer_actor_uid="adapter-transfer",
            **placement,
        )
        peer = await xo.create_actor(
            AdapterPeerActor, uid="adapter-transfer-1", **placement
        )
        await manager.register_rank(1, peer.address, update=True)
        assert await peer.get_world_addresses() == [peer.address]
        await tracker.register_blocks(0, [(123, 7)], 1)
        assert await tracker.query_blocks(0, [(123, 9)]) == {1: {(123, 7, 9)}}
        await manager.unregister_rank(1)
        assert await tracker.query_blocks(0, [(123, 9)]) == {}
        await manager.register_rank(1, peer.address, update=True)
        assert await tracker.query_blocks(0, [(123, 9)]) == {}
        await tracker.register_blocks(0, [(456, 7)], 1)
        assert await tracker.query_blocks(0, [(456, 9)]) == {1: {(456, 7, 9)}}
