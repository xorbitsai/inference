# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import pytest
import torch

from ..tiered_snapshot import TieredKVSnapshotStore


def store(gpu=1, cpu=2):
    # CPU device exercises placement policy without a CUDA dependency.
    return TieredKVSnapshotStore(cpu, gpu * 16, 16, torch.device("cpu"))


def stage(s, key):
    for layer in ["K", "V"]:
        s.stage(layer, [key], torch.tensor([[key, key + 1.0]]))
    assert s.publish([key], {"K", "V"}) == [key]


def test_gpu_overflow_preserves_cpu_capacity_and_content():
    s = store()
    for key in [1, 2, 3]:
        stage(s, key)
    assert s.counts == {"gpu": 1, "cpu": 2}
    assert s.tiers == {1: "cpu", 2: "cpu", 3: "gpu"}
    assert s.ready == {1, 2, 3}
    assert not s.evicted
    assert s.read("K", [1]).tolist() == [[1, 2]]
    stage(s, 4)
    assert s.evicted == {1}
    assert s.ready == {2, 3, 4}
    assert s.stats()["gpu_reserved_bytes"] == 16


def test_leases_prevent_migration_and_cpu_eviction():
    s = store(cpu=1)
    stage(s, 1)
    assert s.reserve("2:first", [1])
    stage(s, 2)
    assert s.tiers == {1: "gpu", 2: "cpu"}
    assert s.reserve("2:second", [2])
    s.stage("K", [3], torch.tensor([[3.0, 4.0]]))
    assert 3 not in s.blocks
    assert s.publish([3], {"K", "V"}) == []
    s.release_consumer(2)
    stage(s, 3)
    assert s.tiers == {1: "cpu", 3: "gpu"}
    assert s.evicted == {2}


def test_zero_gpu_budget_and_atomic_publication():
    s = store(gpu=0)
    value = torch.tensor([[1.0, 2.0]])
    s.stage("K", [1], value)
    value.fill_(42)
    assert s.publish([1], {"K", "V"}) == []
    assert not s.reserve("2:r", [1])
    s.stage("V", [1], torch.tensor([[3.0, 4.0]]))
    assert s.publish([1], {"K", "V"}) == [1]
    assert s.reserve("2:r", [1])
    assert s.metrics["cpu_hits"] == 1
    assert s.counts["gpu"] == 0
    assert s.read("K", [1]).tolist() == [[1, 2]]


def test_lru_lease_release_and_immutable_content():
    s = store(gpu=2)
    stage(s, 1)
    stage(s, 2)
    assert s.reserve("2:r", [1])
    s.release("2:r")
    stage(s, 3)
    assert s.tiers[2] == "cpu"
    s.stage("K", [1], torch.tensor([[99.0, 99.0]]))
    assert s.read("K", [1]).tolist() == [[1, 2]]


def test_full_block_size_is_enforced():
    s = store()
    with pytest.raises(ValueError, match="full-block"):
        s.stage("K", [1], torch.ones(1, 5))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_physical_placement_demotion_and_mixed_read(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required for physical GPU placement")
    target = torch.device(device)
    s = TieredKVSnapshotStore(2, 16, 16, target)
    stage(s, 1)
    assert s.blocks[1]["K"].device == target
    stage(s, 2)
    assert s.blocks[1]["K"].device.type == "cpu"
    assert s.blocks[2]["K"].device == target
    assert s.read("K", [2, 1]).tolist() == [[2, 3], [1, 2]]
    result = s.read("K", [1, 2], device=target)
    assert result.device == target
    assert result.cpu().tolist() == [[1, 2], [2, 3]]
    assert s.stats()["demotions"] == 1


def test_invalid_stage_does_not_evict_existing_content():
    s = store(cpu=1)
    stage(s, 1)
    stage(s, 2)
    before = dict(s.tiers)
    with pytest.raises(ValueError, match="full-block"):
        s.stage("K", [3], torch.ones(1, 5))
    assert s.tiers == before
    assert s.ready == {1, 2}
    assert not s.evicted
    assert s.stats()["gpu_reserved_bytes"] == 16


def test_partial_block_reserves_full_budget_and_keeps_existing_layer():
    s = store()
    s.stage("K", [1], torch.ones(1, 2))
    assert s.stats()["gpu_reserved_bytes"] == 16
    with pytest.raises(ValueError, match="full-block"):
        s.stage("V", [1], torch.ones(1, 3))
    assert s.publish([1], {"K", "V"}) == []
    assert s.read("K", [1]).tolist() == [[1, 1]]
    s.stage("V", [1], torch.ones(1, 2))
    assert s.publish([1], {"K", "V"}) == [1]


def test_budget_rounds_down_to_whole_blocks():
    s = TieredKVSnapshotStore(2, 31, 16, torch.device("cpu"))
    stage(s, 1)
    stage(s, 2)
    assert s.gpu_capacity == 1
    assert s.stats()["gpu_reserved_bytes"] == 16


def test_failed_first_copy_releases_empty_slot(monkeypatch):
    s = store()

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch.Tensor, "to", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        s.stage("K", [1], torch.ones(1, 2))
    assert not s.blocks
    assert not s.tiers
    assert s.counts == {"gpu": 0, "cpu": 0}
