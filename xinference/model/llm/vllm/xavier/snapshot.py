# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from collections import OrderedDict
from typing import Dict, List, Set

import torch


class KVSnapshotStore:
    """Bounded, content-addressed CPU blocks with atomic publication and read leases.

    All mutation runs on the TransferActor's event loop. Tensor storage is owned
    by each block, never by a GPU slot or a view retaining an entire request.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.blocks: OrderedDict[int, Dict[str, torch.Tensor]] = OrderedDict()
        self.ready: Set[int] = set()
        self.leases: Dict[str, Set[int]] = {}
        self.evicted: Set[int] = set()

    def stage(self, layer: str, keys: List[int], tensors: torch.Tensor):
        pinned = set().union(*self.leases.values()) if self.leases else set()
        for key, tensor in zip(keys, tensors):
            if key not in self.blocks:
                if len(self.blocks) >= self.capacity:
                    victim = next((k for k in self.blocks if k not in pinned), None)
                    if victim is None:
                        continue  # No room: leave this block to local recomputation.
                    del self.blocks[victim]
                    self.ready.discard(victim)
                    self.evicted.add(victim)
                self.blocks[key] = {}
            self.blocks.move_to_end(key)
            # Identical content is immutable, including while a reader holds it.
            if layer not in self.blocks[key]:
                self.blocks[key][layer] = tensor.clone().contiguous()

    def publish(self, keys: List[int], layers: Set[str]) -> List[int]:
        available = []
        for key in keys:
            if layers and layers.issubset(self.blocks.get(key, {})):
                self.ready.add(key)
                available.append(key)
        return available

    def reserve(self, lease: str, keys: List[int]) -> bool:
        if not set(keys).issubset(self.ready):
            return False
        self.leases.setdefault(lease, set()).update(keys)
        for key in keys:
            self.blocks.move_to_end(key)
        return True

    def release(self, lease: str):
        self.leases.pop(lease, None)

    def release_consumer(self, rank: int):
        for lease in list(self.leases):
            if lease.startswith(f"{rank}:"):
                self.release(lease)

    def read(self, layer: str, keys: List[int]) -> torch.Tensor:
        return torch.stack([self.blocks[key][layer] for key in keys]).contiguous()


def block_major_view(tensor: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Normalize supported attention layouts without guessing ambiguous axes."""
    if tensor.ndim == 5 and tensor.shape[0] == 2:
        if tensor.shape[1] == num_blocks:
            if num_blocks == 2:
                raise ValueError("Ambiguous Xavier KV block axis with only two blocks")
            return tensor.movedim(1, 0)
    if tensor.shape[0] != num_blocks:
        raise ValueError(f"Unsupported Xavier KV layout {tuple(tensor.shape)}")
    return tensor
