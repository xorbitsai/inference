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
    assert s.reserve("2:r", [2])
    stage(s, 3)
    assert s.tiers[2] == "gpu"
    s.release("2:r")
    stage(s, 4)
    assert s.tiers[2] == "cpu"
    s.stage("K", [2], torch.tensor([[99.0, 99.0]]))
    assert s.read("K", [2]).tolist() == [[2, 3]]


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


def test_leased_cpu_does_not_block_unleased_gpu_replacement(assert_gpu_lru_consistent):
    s = store(cpu=1)
    stage(s, 1)
    stage(s, 2)
    assert s.reserve("2:r", [1])
    stage(s, 3)
    assert s.tiers == {1: "cpu", 3: "gpu"}
    assert s.ready == {1, 3}
    assert s.evicted == {2}
    assert s.counts == {"gpu": 1, "cpu": 1}
    assert s.read("K", [1, 3]).tolist() == [[1, 2], [3, 4]]
    assert_gpu_lru_consistent(s)


def test_failed_demotion_preserves_both_tiers(monkeypatch):
    s = store(cpu=1)
    stage(s, 1)
    stage(s, 2)
    before = s.stats()
    order = list(s.blocks)
    original_to = torch.Tensor.to
    failed_value = s.blocks[2]["V"]

    def fail_second_layer(value, *args, **kwargs):
        if value is failed_value:
            raise RuntimeError("host copy failed")
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fail_second_layer)
    with pytest.raises(RuntimeError, match="host copy failed"):
        s.stage("K", [3], torch.ones(1, 2))
    assert list(s.blocks) == order
    assert s.tiers == {1: "cpu", 2: "gpu"}
    assert s.ready == {1, 2}
    assert not s.evicted
    assert s.stats() == before
    assert s.read("K", [1, 2]).tolist() == [[1, 2], [2, 3]]


def test_invalid_later_key_rejects_entire_batch():
    s = store(cpu=1)
    stage(s, 1)
    stage(s, 2)
    before = s.stats()
    order = list(s.blocks)
    # Key 3 fits, but key 2 already occupies its full block budget.
    with pytest.raises(ValueError, match="full-block"):
        s.stage("extra", [3, 2], torch.ones(2, 2))
    assert list(s.blocks) == order
    assert s.tiers == {1: "cpu", 2: "gpu"}
    assert s.ready == {1, 2}
    assert not s.evicted
    assert s.stats() == before
    assert s.read("K", [1, 2]).tolist() == [[1, 2], [2, 3]]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_demotion_packs_compatible_layers_and_preserves_shapes(monkeypatch, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required for physical copies")
    layers = {
        "K": torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3),
        "V": torch.arange(4, device=device, dtype=torch.float32).reshape(4, 1),
        "index": torch.arange(3, device=device, dtype=torch.int64),
    }
    expected = {name: value.cpu().clone() for name, value in layers.items()}
    copies = []
    original_to = torch.Tensor.to

    def track_copy(value, *args, **kwargs):
        if value.device.type == "cuda" and args and args[0] == "cpu":
            copies.append(value.numel())
        return original_to(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        if device == "cpu":
            # Exercise the real grouping/cat/split/reshape path on CPU storage.
            # Only the device metadata used for dispatch is substituted.
            patch.setattr(
                torch.Tensor, "device", property(lambda _: torch.device("cuda:0"))
            )
        patch.setattr(torch.Tensor, "to", track_copy)
        host = TieredKVSnapshotStore._copy_to_cpu(layers)
    assert copies == [10, 3]
    for name, value in host.items():
        assert value.device.type == "cpu"
        assert value.dtype == expected[name].dtype
        assert value.shape == expected[name].shape
        assert torch.equal(value, expected[name])
    for value in layers.values():
        value.zero_()
    assert all(torch.equal(host[name], value) for name, value in expected.items())
    host["K"].fill_(42)
    assert torch.equal(host["V"], expected["V"])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_failed_packed_demotion_preserves_content(
    monkeypatch, device, assert_gpu_lru_consistent
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required for physical copies")
    s = TieredKVSnapshotStore(1, 16, 16, torch.device(device))
    stage(s, 1)
    stage(s, 2)
    before = s.stats()
    order = list(s.blocks)
    original_to = torch.Tensor.to

    def fail_host_copy(value, *args, **kwargs):
        if value.device.type == "cuda" and args and args[0] == "cpu":
            raise RuntimeError("packed copy failed")
        return original_to(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        if device == "cpu":
            patch.setattr(
                torch.Tensor, "device", property(lambda _: torch.device("cuda:0"))
            )
        patch.setattr(torch.Tensor, "to", fail_host_copy)
        with pytest.raises(RuntimeError, match="packed copy failed"):
            stage(s, 3)
    assert list(s.blocks) == order
    assert s.tiers == {1: "cpu", 2: "gpu"}
    assert s.ready == {1, 2}
    assert not s.evicted
    assert s.stats() == before
    assert s.read("K", [1, 2]).tolist() == [[1, 2], [2, 3]]
    assert_gpu_lru_consistent(s)


def test_gpu_victims_follow_global_lru_after_hits_and_leases(assert_gpu_lru_consistent):
    s = store(gpu=2, cpu=3)
    for key in [1, 2, 3]:
        stage(s, key)
    # A repeated staging hit refreshes GPU LRU without copying the snapshot.
    s.stage("K", [2], torch.zeros(1, 2))
    stage(s, 4)
    assert s.tiers == {1: "cpu", 2: "gpu", 3: "cpu", 4: "gpu"}
    assert s.reserve("2:hot", [2])
    stage(s, 5)
    assert s.tiers[4] == "cpu"
    s.release("2:hot")
    s.touch(2)
    stage(s, 6)
    assert s.tiers[5] == "cpu"
    assert 1 not in s.blocks
    assert list(s._gpu_lru) == [2, 6]
    assert_gpu_lru_consistent(s)
    assert s.read("K", [2]).tolist() == [[2, 3]]


def test_gpu_lru_tracks_drops_and_failed_first_copy(monkeypatch):
    s = store(gpu=2)
    stage(s, 1)
    s._drop(1)
    assert not s._gpu_lru

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch.Tensor, "to", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        stage(s, 2)
    assert not s._gpu_lru
    assert not s.blocks


def test_incremental_size_uses_physical_dtype_and_cleans_failed_slots(monkeypatch):
    s = store(gpu=2)
    s.stage("K", [1], torch.ones(1, 2, dtype=torch.float32), torch.bfloat16)
    assert s._block_sizes == {1: 8}
    # Immutable hits must not add the same layer twice.
    s.stage("K", [1], torch.ones(1, 2, dtype=torch.float32))
    assert s._block_sizes == {1: 8}
    with pytest.raises(ValueError, match="full-block"):
        s.stage("V", [1], torch.ones(1, 3))
    assert s._block_sizes == {1: 8}
    s._drop(1)
    assert not s._block_sizes

    def fail(*args, **kwargs):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(torch.Tensor, "to", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        stage(s, 2)
    assert not s._block_sizes


def test_incremental_size_survives_demotion_eviction_and_key_reuse():
    s = store(gpu=1, cpu=1)
    for key in [1, 2]:
        stage(s, key)
    assert s._block_sizes == {1: 16, 2: 16}
    stage(s, 3)
    assert s._block_sizes == {2: 16, 3: 16}
    # Reusing an evicted hash starts accounting from its new content.
    s.stage("K", [1], torch.ones(1, 1))
    assert s._block_sizes == {3: 16, 1: 4}
    for key, layers in s.blocks.items():
        assert s._block_sizes[key] == sum(
            value.numel() * value.element_size() for value in layers.values()
        )
