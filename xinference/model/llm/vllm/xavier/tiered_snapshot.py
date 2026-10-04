# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Budgeted hot snapshots with CPU overflow and shared publication/read leases."""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch

from .snapshot import KVSnapshotStore


class TieredKVSnapshotStore(KVSnapshotStore):
    """GPU-first snapshot policy, independent of the actor transport.

    ``block_bytes`` reserves a complete block across all layers, including while
    it is only partially staged. The GPU budget covers retained snapshots, not
    temporary copy/read tensors or the CUDA allocator's reserved memory.
    Leased blocks cannot migrate or be evicted. Reads default to CPU to retain
    the base store's interface; GPU consumers must select their output device.
    CUDA operations follow the caller's current stream.
    """

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
        # GPU victim lookup must not walk an arbitrarily large CPU history.
        # Keep the GPU subsequence in the same order as the global block LRU.
        self._gpu_lru: OrderedDict[int, None] = OrderedDict()
        self.counts = {"gpu": 0, "cpu": 0}
        self.metrics = dict(demotions=0, gpu_hits=0, cpu_hits=0, skipped=0)
        self.block_bytes = block_bytes

    def touch(self, key: int) -> None:
        self.blocks.move_to_end(key)
        if key in self._gpu_lru:
            self._gpu_lru.move_to_end(key)

    def _drop(self, key: int) -> None:
        self._gpu_lru.pop(key, None)
        self.counts[self.tiers.pop(key)] -= 1
        del self.blocks[key]
        del self.logical_dtypes[key]
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

    @staticmethod
    def _copy_to_cpu(layers: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # A synchronous D2H copy per layer stalls the producer for every layer
        # of every evicted block. Pack compatible layers and synchronize once
        # per group. Each host allocation belongs to one block, so evicting a
        # block releases its storage without retaining unrelated snapshots.
        groups: Dict[
            Tuple[torch.device, torch.dtype], List[Tuple[str, torch.Tensor]]
        ] = {}
        host = {}
        for layer, value in layers.items():
            if value.device.type != "cuda":
                host[layer] = value.to("cpu", copy=True)
            else:
                groups.setdefault((value.device, value.dtype), []).append(
                    (layer, value)
                )
        for group in groups.values():
            packed = torch.cat([value.reshape(-1) for _, value in group]).to("cpu")
            values = packed.split([value.numel() for _, value in group])
            for (layer, original), value in zip(group, values):
                host[layer] = value.reshape(original.shape)
        return host

    def _admit(self, key: int, pinned: set) -> bool:
        if self.gpu_capacity and self.counts["gpu"] >= self.gpu_capacity:
            victim = next(
                (k for k in self._gpu_lru if k not in pinned),
                None,
            )
            if victim is not None:
                cpu_available = self.counts["cpu"] < self.cpu_capacity or any(
                    self.tiers[k] == "cpu" and k not in pinned for k in self.blocks
                )
                if cpu_available:
                    # Copy before evicting CPU content or publishing the new tier.
                    host = self._copy_to_cpu(self.blocks[victim])
                    self._cpu_room(pinned)
                    self.blocks[victim] = host
                    self.tiers[victim] = "cpu"
                    del self._gpu_lru[victim]
                    self.counts["gpu"] -= 1
                    self.counts["cpu"] += 1
                    self.metrics["demotions"] += 1
                else:
                    # CPU leases must not prevent reuse of an unleased GPU slot.
                    self._drop(victim)
        tier = "gpu" if self.counts["gpu"] < self.gpu_capacity else "cpu"
        if tier == "cpu" and not self._cpu_room(pinned):
            self.metrics["skipped"] += 1
            return False
        self.blocks[key] = {}
        self.logical_dtypes[key] = {}
        self.tiers[key] = tier
        if tier == "gpu":
            self._gpu_lru[key] = None
        self.counts[tier] += 1
        return True

    def stage(
        self,
        layer: str,
        keys: List[int],
        tensors: torch.Tensor,
        logical_dtype: torch.dtype | None = None,
    ):
        # Reject the whole batch before changing content, placement, or LRU order.
        for key, tensor in zip(keys, tensors):
            existing = self.blocks.get(key, {})
            if layer in existing:
                continue
            used = sum(t.numel() * t.element_size() for t in existing.values())
            if used + tensor.numel() * tensor.element_size() > self.block_bytes:
                raise ValueError("Snapshot exceeds configured full-block size")
        pinned = set().union(*self.leases.values()) if self.leases else set()
        for key, tensor in zip(keys, tensors):
            if key not in self.blocks and not self._admit(key, pinned):
                continue
            self.touch(key)
            if layer not in self.blocks[key]:
                device = (
                    self.gpu_device if self.tiers[key] == "gpu" else torch.device("cpu")
                )
                try:
                    self.blocks[key][layer] = (
                        tensor.detach().to(device, copy=True).contiguous()
                    )
                    self.logical_dtypes[key][layer] = logical_dtype or tensors.dtype
                except Exception:
                    # A failed first layer must not consume an empty slot.
                    if not self.blocks[key]:
                        self._drop(key)
                    raise

    def reserve(self, lease: str, keys: List[int]) -> bool:
        if not super().reserve(lease, keys):
            return False
        for key in keys:
            if key in self._gpu_lru:
                self._gpu_lru.move_to_end(key)
            self.metrics[self.tiers[key] + "_hits"] += 1
        return True

    def read(
        self, layer: str, keys: List[int], device: torch.device | None = None
    ) -> torch.Tensor:
        target = torch.device("cpu") if device is None else device
        return torch.stack(
            [self.blocks[key][layer].to(target) for key in keys]
        ).contiguous()

    def stats(self) -> dict:
        return dict(
            self.metrics,
            gpu_blocks=self.counts["gpu"],
            cpu_blocks=self.counts["cpu"],
            gpu_capacity=self.gpu_capacity,
            cpu_capacity=self.cpu_capacity,
            gpu_reserved_bytes=self.counts["gpu"] * self.block_bytes,
        )
