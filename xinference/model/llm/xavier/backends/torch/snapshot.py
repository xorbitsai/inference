# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch


class _PackedKVBlock(MutableMapping[str, torch.Tensor]):
    """Index an owned slab; create tensor views only when a reader needs them."""

    __slots__ = (
        "_packed",
        "_schema",
        "_row",
        "_layers",
        "_overrides",
        "_removed",
        "_numpy",
    )

    def __init__(
        self,
        packed: torch.Tensor,
        schema: Dict[str, Tuple[Tuple[int, ...], torch.dtype, int, int]],
        row: Optional[int] = None,
    ):
        self._packed, self._schema, self._row = packed, schema, row
        self._layers: Dict[str, Optional[torch.Tensor]] = {}
        self._overrides: Optional[Set[str]] = None
        self._removed: Optional[Set[str]] = None
        self._numpy: Optional[np.ndarray] = None

    def packed_layer(
        self, name: str
    ) -> Optional[Tuple[Tuple[int, ...], torch.dtype, np.ndarray]]:
        if name not in self or (
            self._overrides is not None and name in self._overrides
        ):
            return None
        if self._numpy is None:
            value = self._packed.numpy()
            self._numpy = value if self._row is None else value[self._row]
        shape, dtype, offset, nbytes = self._schema[name]
        return shape, dtype, self._numpy[offset : offset + nbytes]

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self:
            raise KeyError(name)
        value = self._layers.get(name)
        if value is None:
            shape, dtype, offset, nbytes = self._schema[name]
            carrier = torch.float16 if dtype == torch.bfloat16 else dtype
            row = self._packed if self._row is None else self._packed[self._row]
            value = row[offset : offset + nbytes].view(carrier).view(shape)
            self._layers[name] = value
        return value

    def __setitem__(self, name: str, value: torch.Tensor) -> None:
        self._layers[name] = value
        if self._overrides is None:
            self._overrides = set()
        self._overrides.add(name)
        if self._removed is not None:
            self._removed.discard(name)

    def __delitem__(self, name: str) -> None:
        if name not in self:
            raise KeyError(name)
        self._layers.pop(name, None)
        if self._removed is None:
            self._removed = set()
        self._removed.add(name)

    def __contains__(self, name: object) -> bool:
        return (name in self._schema or name in self._layers) and (
            self._removed is None or name not in self._removed
        )

    def __iter__(self) -> Iterator[str]:
        for name in self._schema:
            if self._removed is None or name not in self._removed:
                yield name
        for name in self._layers:
            if name not in self._schema:
                yield name

    def __len__(self) -> int:
        return sum(1 for _ in self)


class KVSnapshotStore:
    """Bounded, content-addressed CPU blocks with atomic publication and read leases.

    All mutation runs on the TransferActor's event loop. CPU payloads never
    alias recyclable GPU slots. Shared packed allocations are charged in full
    and evicted together, so sparse views cannot retain unbounded batches.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.blocks: OrderedDict[int, MutableMapping[str, torch.Tensor]] = OrderedDict()
        self.logical_dtypes: Dict[int, Dict[str, torch.dtype]] = {}
        self.ready: Set[int] = set()
        self.leases: Dict[str, Set[int]] = {}
        self.evicted: Set[int] = set()
        self._packed_slabs: Dict[int, Set[int]] = {}
        self._packed_slab_slots: Dict[int, int] = {}
        self._block_slabs: Dict[int, int] = {}
        self._reserved_slots = 0

    def _used_slots(self) -> int:
        return len(self.blocks) - len(self._block_slabs) + self._reserved_slots

    def _evict_one(self, pinned: Set[int]) -> bool:
        for key in self.blocks:
            slab = self._block_slabs.get(key)
            victims = self._packed_slabs[slab] if slab is not None else {key}
            if victims & pinned:
                continue
            for victim in victims:
                del self.blocks[victim]
                del self.logical_dtypes[victim]
                self.ready.discard(victim)
                self.evicted.add(victim)
                self._block_slabs.pop(victim, None)
            if slab is not None:
                del self._packed_slabs[slab]
                self._reserved_slots -= self._packed_slab_slots.pop(slab)
            return True
        return False

    def stage(
        self,
        layer: str,
        keys: List[int],
        tensors: torch.Tensor,
        logical_dtype: torch.dtype | None = None,
    ):
        pinned = set().union(*self.leases.values()) if self.leases else set()
        for key, tensor in zip(keys, tensors):
            if key not in self.blocks:
                if self._used_slots() >= self.capacity and not self._evict_one(pinned):
                    continue  # No room: leave this block to local recomputation.
                self.blocks[key] = {}
                self.logical_dtypes[key] = {}
            self.blocks.move_to_end(key)
            # Identical content is immutable, including while a reader holds it.
            if layer not in self.blocks[key]:
                self.blocks[key][layer] = tensor.clone().contiguous()
                self.logical_dtypes[key][layer] = logical_dtype or tensors.dtype

    def stage_packed_blocks(
        self,
        keys: List[int],
        packed: torch.Tensor,
        layers: List[Tuple[str, Tuple[int, ...], torch.dtype, int, int]],
        *,
        retain_storage: bool = False,
    ) -> None:
        """Own complete blocks, with bounded whole-slab ownership if requested."""
        if packed.dtype != torch.uint8 or packed.ndim != 2 or len(packed) != len(keys):
            raise ValueError("Invalid packed snapshot payload")
        pinned = set().union(*self.leases.values()) if self.leases else set()
        schema = {
            name: (shape, dtype, offset, nbytes)
            for name, shape, dtype, offset, nbytes in layers
        }
        dtypes = {name: dtype for name, _, dtype, _, _ in layers}
        if retain_storage and keys and all(key not in self.blocks for key in keys):
            storage = packed.untyped_storage()
            slab = storage.data_ptr()
            row_bytes = packed.shape[1]
            slots = (storage.nbytes() + row_bytes - 1) // row_bytes
            if slots <= self.capacity and slab not in self._packed_slabs:
                while self._used_slots() + slots > self.capacity:
                    if not self._evict_one(pinned):
                        break
                else:
                    self._packed_slabs[slab] = set(keys)
                    self._packed_slab_slots[slab] = slots
                    self._reserved_slots += slots
                    for index, key in enumerate(keys):
                        self.blocks[key] = _PackedKVBlock(packed, schema, index)
                        self.logical_dtypes[key] = dtypes.copy()
                        self._block_slabs[key] = slab
                    return
        rows = packed.numpy()
        for index, key in enumerate(keys):
            fresh = key not in self.blocks
            if fresh:
                if self._used_slots() >= self.capacity and not self._evict_one(pinned):
                    continue
                self.blocks[key] = {}
                self.logical_dtypes[key] = {}
            self.blocks.move_to_end(key)
            # NumPy memcpy avoids a CPU thread-pool launch for every block.
            # This allocation never retains any other block in the batch.
            block = torch.from_numpy(rows[index].copy())
            if fresh:
                self.blocks[key] = _PackedKVBlock(block, schema)
                self.logical_dtypes[key] = dtypes.copy()
                continue
            for name, shape, dtype, offset, nbytes in layers:
                if name not in self.blocks[key]:
                    carrier = torch.float16 if dtype == torch.bfloat16 else dtype
                    self.blocks[key][name] = (
                        block[offset : offset + nbytes].view(carrier).view(shape)
                    )
                    self.logical_dtypes[key][name] = dtype

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
        blocks = [self.blocks[key] for key in keys]
        rows = []
        for block in blocks:
            if not isinstance(block, _PackedKVBlock):
                break
            row = block.packed_layer(layer)
            if row is None:
                break
            rows.append(row)
        if rows and len(rows) == len(blocks):
            shape, dtype, _ = rows[0]
            if all(row[:2] == (shape, dtype) for row in rows):
                # Copy a read in NumPy without materializing and stacking a
                # tensor view for every layer of every block.
                carrier = torch.float16 if dtype == torch.bfloat16 else dtype
                packed = torch.from_numpy(np.stack([row[2] for row in rows]))
                return packed.view(carrier).view(len(keys), *shape)
        return torch.stack([self.blocks[key][layer] for key in keys]).contiguous()


def block_major_view(
    tensor: torch.Tensor, num_blocks: int, *, allow_multiple: bool = False
) -> torch.Tensor:
    """Normalize supported attention layouts without guessing ambiguous axes."""

    def matches(size: int) -> bool:
        return size == num_blocks or (
            allow_multiple and num_blocks > 0 and size > 0 and size % num_blocks == 0
        )

    first_is_blocks = matches(tensor.shape[0])
    if tensor.ndim == 5 and tensor.shape[0] == 2 and matches(tensor.shape[1]):
        if first_is_blocks:
            raise ValueError("Ambiguous Xavier KV block axis with only two blocks")
        return tensor.movedim(1, 0)
    if not first_is_blocks:
        raise ValueError(f"Unsupported Xavier KV layout {tuple(tensor.shape)}")
    return tensor
