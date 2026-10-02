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
