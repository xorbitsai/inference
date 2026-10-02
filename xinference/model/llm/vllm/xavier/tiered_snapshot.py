# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Budgeted hot snapshots with CPU overflow and shared publication/read leases."""

from typing import Dict, List

import torch

from .snapshot import KVSnapshotStore


class TieredKVSnapshotStore(KVSnapshotStore):
    def __init__(
        self,
        cpu_capacity: int,
        gpu_budget_bytes: int,
        block_bytes: int,
        gpu_device: torch.device,
    ):
        if cpu_capacity <= 0 or gpu_budget_bytes < 0 or block_bytes <= 0:
            raise ValueError("Invalid tiered snapshot capacity or budget")
        super().__init__(cpu_capacity + gpu_budget_bytes // block_bytes)
        self.cpu_capacity = cpu_capacity
        self.gpu_capacity = gpu_budget_bytes // block_bytes
        self.gpu_device = gpu_device
        self.tiers: Dict[int, str] = {}
        self.counts = {"gpu": 0, "cpu": 0}
        self.metrics = dict(demotions=0, gpu_hits=0, cpu_hits=0, skipped=0)
        self.block_bytes = block_bytes

    def _drop(self, key: int) -> None:
        self.counts[self.tiers.pop(key)] -= 1
        del self.blocks[key]
        self.ready.discard(key)
        self.evicted.add(key)

    def _cpu_room(self, pinned: set) -> bool:
        if self.counts["cpu"] < self.cpu_capacity:
            return True
        victim = next(
            (k for k in self.blocks if self.tiers[k] == "cpu" and k not in pinned),
            None,
        )
        if victim is None:
            return False
        self._drop(victim)
        return True

    def _admit(self, key: int, pinned: set) -> bool:
        if self.gpu_capacity and self.counts["gpu"] >= self.gpu_capacity:
            victim = next(
                (k for k in self.blocks if self.tiers[k] == "gpu" and k not in pinned),
                None,
            )
            if victim is not None and self._cpu_room(pinned):
                # Complete all host copies before publishing the new location.
                host = {
                    layer: value.to("cpu", copy=True)
                    for layer, value in self.blocks[victim].items()
                }
                self.blocks[victim] = host
                self.tiers[victim] = "cpu"
                self.counts["gpu"] -= 1
                self.counts["cpu"] += 1
                self.metrics["demotions"] += 1
        tier = "gpu" if self.counts["gpu"] < self.gpu_capacity else "cpu"
        if tier == "cpu" and not self._cpu_room(pinned):
            self.metrics["skipped"] += 1
            return False
        self.blocks[key] = {}
        self.tiers[key] = tier
        self.counts[tier] += 1
        return True

    def stage(self, layer: str, keys: List[int], tensors: torch.Tensor):
        pinned = set().union(*self.leases.values()) if self.leases else set()
        for key, tensor in zip(keys, tensors):
            if key not in self.blocks and not self._admit(key, pinned):
                continue
            self.blocks.move_to_end(key)
            if layer not in self.blocks[key]:
                used = sum(
                    t.numel() * t.element_size() for t in self.blocks[key].values()
                )
                if used + tensor.numel() * tensor.element_size() > self.block_bytes:
                    raise ValueError("Snapshot exceeds configured full-block size")
                device = (
                    self.gpu_device if self.tiers[key] == "gpu" else torch.device("cpu")
                )
                self.blocks[key][layer] = (
                    tensor.detach().to(device, copy=True).contiguous()
                )

    def reserve(self, lease: str, keys: List[int]) -> bool:
        if not super().reserve(lease, keys):
            return False
        for key in keys:
            self.metrics[self.tiers[key] + "_hits"] += 1
        return True

    def stats(self) -> dict:
        return dict(
            self.metrics,
            gpu_blocks=self.counts["gpu"],
            cpu_blocks=self.counts["cpu"],
            gpu_capacity=self.gpu_capacity,
            cpu_capacity=self.cpu_capacity,
            gpu_reserved_bytes=self.counts["gpu"] * self.block_bytes,
        )
