# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Gather and pack aligned CUDA KV blocks without per-layer intermediates."""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_words(
    meta,
    indices,
    output,
    count,
    ROW_WORDS: tl.constexpr,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row = tl.program_id(2), tl.program_id(1)
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    address = tl.load(meta + layer * 3).to(tl.pointer_type(tl.uint64))
    offset = tl.load(meta + layer * 3 + 1)
    group = tl.load(meta + layer * 3 + 2)
    source_block = tl.load(indices + group * count + row)
    source_position = source_block * BLOCK_STRIDE
    remaining = position
    for dimension in tl.static_range(len(SHAPE) - 1, -1, -1):
        source_position += (remaining % SHAPE[dimension]) * STRIDES[dimension]
        remaining = remaining // SHAPE[dimension]
    value = tl.load(address + source_position, mask=position < SIZE, other=0)
    target = output.to(tl.pointer_type(tl.uint64))
    tl.store(target + row * ROW_WORDS + offset + position, value, mask=position < SIZE)


class PackedGather:
    @classmethod
    def try_create(
        cls, caches: Dict[str, torch.Tensor], source_groups: Dict[str, int]
    ) -> Optional["PackedGather"]:
        if not caches or len({tensor.device for tensor in caches.values()}) != 1:
            return None
        for tensor in caches.values():
            size = tensor.element_size()
            if (
                not tensor.is_cuda
                or tensor.ndim < 2
                or tensor.stride(-1) != 1
                or tensor.shape[-1] * size % 8
                or tensor.data_ptr() % 8
                or any(stride * size % 8 for stride in tensor.stride()[:-1])
            ):
                return None
        return cls(caches, source_groups)

    def __init__(self, caches: Dict[str, torch.Tensor], source_groups: Dict[str, int]):
        self.caches = caches
        self.layers: List[Tuple[str, Tuple[int, ...], torch.dtype, int, int]] = []
        self.source_names = list(dict.fromkeys(source_groups.values()))
        group_indices = {group: i for i, group in enumerate(self.source_names)}
        self.representatives = {
            group: next(name for name in caches if source_groups[name] == group)
            for group in self.source_names
        }
        self.source_groups = source_groups
        self.source_limits = {
            group: min(
                tensor.shape[0]
                for name, tensor in caches.items()
                if source_groups[name] == group
            )
            for group in self.source_names
        }
        geometries = defaultdict(list)
        offset = 0
        for name, tensor in caches.items():
            size = tensor.element_size()
            shape = (*tensor.shape[1:-1], tensor.shape[-1] * size // 8)
            strides = tuple(step * size // 8 for step in tensor.stride()[1:-1]) + (1,)
            block_stride = tensor.stride(0) * size // 8
            nbytes = math.prod(tensor.shape[1:]) * size
            geometries[(shape, strides, block_stride)].append(
                [tensor.data_ptr(), offset // 8, group_indices[source_groups[name]]]
            )
            self.layers.append(
                (name, tuple(tensor.shape[1:]), tensor.dtype, offset, nbytes)
            )
            offset += nbytes
        self.row_bytes = offset
        self.device = next(iter(caches.values())).device
        self.metadata = []
        for geometry, rows in geometries.items():
            host = torch.tensor(rows, dtype=torch.int64, pin_memory=True)
            self.metadata.append(
                (host, host.to(self.device, non_blocking=True), geometry)
            )

    def __call__(
        self, sources: Dict[str, List[int]], output: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        count = len(next(iter(sources.values())))
        for name in self.caches:
            representative = self.representatives[self.source_groups[name]]
            if sources[name] != sources[representative]:
                raise ValueError("Inconsistent source indices in one KV group")
        for group, name in self.representatives.items():
            if len(sources[name]) != count:
                raise ValueError("Inconsistent KV group batch lengths")
            limit = self.source_limits[group]
            if any(index < 0 or index >= limit for index in sources[name]):
                raise IndexError("Invalid Xavier source block index")
        host = torch.tensor(
            [sources[self.representatives[group]] for group in self.source_names],
            dtype=torch.int64,
            pin_memory=True,
        )
        indices = host.to(self.device, non_blocking=True)
        if output is None:
            output = torch.empty(
                (count, self.row_bytes), dtype=torch.uint8, device=self.device
            )
        elif (
            output.shape != (count, self.row_bytes)
            or output.dtype != torch.uint8
            or output.device != self.device
            or not output.is_contiguous()
        ):
            raise ValueError("Invalid packed export destination")
        for _, meta, (shape, strides, block_stride) in self.metadata:
            size = math.prod(shape)
            _gather_words[(triton.cdiv(size, 1024), count, meta.shape[0])](
                meta,
                indices,
                output,
                count,
                self.row_bytes // 8,
                size,
                shape,
                strides,
                block_stride,
                1024,
                num_warps=4,
            )
        return output, [host, indices]

    def warmup(self) -> None:
        # Compile before CUDA graph capture and request processing. Count is a
        # runtime argument, so subsequent batch sizes reuse the same kernel.
        self({name: [0] for name in self.caches})
        torch.cuda.current_stream(self.device).synchronize()
