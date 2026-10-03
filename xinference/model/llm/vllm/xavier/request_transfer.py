# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded cross-layer Gloo messages for Xavier V1 snapshot reads."""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_REQUEST_BLOCKS = 64


@dataclass
class LayerRead:
    layer: str
    keys: List[int]
    destinations: List[int]
    block_shape: Tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return len(self.keys) * math.prod(self.block_shape) * self.dtype.itemsize


def batch_reads(reads):
    batch = []
    size = 0
    for read in reads:
        if batch and read.dtype != batch[0].dtype:
            yield batch
            batch, size = [], 0
        block_bytes = math.prod(read.block_shape) * read.dtype.itemsize
        if block_bytes <= 0:
            raise ValueError("Invalid KV block size")
        offset = 0
        while offset < len(read.keys):
            limit = min(MAX_REQUEST_BLOCKS, len(read.keys) - offset)
            count = max(1, min(limit, (MAX_REQUEST_BYTES - size) // block_bytes))
            if batch and size + count * block_bytes > MAX_REQUEST_BYTES:
                yield batch
                batch, size = [], 0
                continue
            part = LayerRead(
                read.layer,
                read.keys[offset : offset + count],
                read.destinations[offset : offset + count],
                read.block_shape,
                read.dtype,
            )
            batch.append(part)
            size += part.nbytes
            offset += count
            if size >= MAX_REQUEST_BYTES:
                yield batch
                batch, size = [], 0
    if batch:
        yield batch


def pack_reads(store, reads):
    buffers = []
    for read in reads:
        if not set(read.keys).issubset(store.ready):
            raise KeyError("Requested KV snapshots are not published")
        tensor = store.read(read.layer, read.keys)
        if tensor.dtype != read.dtype or tuple(tensor.shape) != (
            len(read.keys),
            *read.block_shape,
        ):
            raise ValueError("KV snapshot layout does not match the receiver")
        buffers.append(tensor.view(torch.uint8).reshape(-1))
    return torch.cat(buffers)


def unpack_reads(payload, reads):
    expected = sum(read.nbytes for read in reads)
    if payload.dtype != torch.uint8 or payload.numel() != expected:
        raise ValueError("Invalid Xavier cross-layer payload size or dtype")
    offset = 0
    for read in reads:
        end = offset + read.nbytes
        yield read, payload[offset:end].view(read.dtype).reshape(
            len(read.keys), *read.block_shape
        )
        offset = end
