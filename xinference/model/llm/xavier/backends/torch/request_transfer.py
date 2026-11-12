# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded cross-layer Gloo messages for Xavier V1 snapshot reads."""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Same-host buffers may carry several messages' worth: no socket bounds them.
MAX_SHARED_BUFFER_BYTES = 4 * MAX_REQUEST_BYTES
MAX_REQUEST_BLOCKS = 64


@dataclass
class LayerRead:
    layer: str
    keys: List[int]
    destinations: List[int]
    block_shape: Tuple[int, ...]
    dtype: torch.dtype
    logical_dtype: torch.dtype | None = None

    @property
    def nbytes(self) -> int:
        return len(self.keys) * math.prod(self.block_shape) * self.dtype.itemsize


def batch_reads(reads, max_bytes=None):
    max_bytes = MAX_REQUEST_BYTES if max_bytes is None else max_bytes
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
            count = max(1, min(limit, (max_bytes - size) // block_bytes))
            if batch and size + count * block_bytes > max_bytes:
                yield batch
                batch, size = [], 0
                continue
            part = LayerRead(
                read.layer,
                read.keys[offset : offset + count],
                read.destinations[offset : offset + count],
                read.block_shape,
                read.dtype,
                read.logical_dtype,
            )
            batch.append(part)
            size += part.nbytes
            offset += count
            if size >= max_bytes:
                yield batch
                batch, size = [], 0
    if batch:
        yield batch


def pack_reads(store, reads):
    buffers = []
    for read in reads:
        if not store.has_layer(read.layer, read.keys):
            raise KeyError("Requested KV snapshots are not published")
        if not store.has_layer(read.layer, read.keys, read.logical_dtype or read.dtype):
            raise ValueError("Xavier logical KV dtype does not match the receiver")
        tensor = store.read(read.layer, read.keys)
        if tensor.dtype != read.dtype or tuple(tensor.shape) != (
            len(read.keys),
            *read.block_shape,
        ):
            raise ValueError("KV snapshot layout does not match the receiver")
        buffers.append(tensor.view(torch.uint8).reshape(-1))
    return torch.cat(buffers)


def collect_reads(store, reads):
    """Validate reads on the store's loop and return their immutable blocks.

    Returns None when a block is not a host tensor of the layout the receiver
    expects block by block; ``pack_reads`` handles those stores.
    """
    sources = []
    for read in reads:
        if not store.has_layer(read.layer, read.keys):
            raise KeyError("Requested KV snapshots are not published")
        if not store.has_layer(read.layer, read.keys, read.logical_dtype or read.dtype):
            raise ValueError("Xavier logical KV dtype does not match the receiver")
        blocks = [store.blocks[key][read.layer] for key in read.keys]
        for block in blocks:
            if block.device.type != "cpu":
                return None
            if block.dtype != read.dtype or tuple(block.shape) != read.block_shape:
                raise ValueError("KV snapshot layout does not match the receiver")
        sources.append(blocks)
    return sources


def write_reads(sources, reads, out):
    """Copy collected blocks into ``out`` in read order.

    Published blocks are immutable and the caller holds them, so this may run
    off the store's loop while the store keeps serving other requests.
    """
    if out.dtype != torch.uint8 or out.numel() != sum(read.nbytes for read in reads):
        raise ValueError("Invalid Xavier cross-layer payload size or dtype")
    offset = 0
    for read, blocks in zip(reads, sources):
        end = offset + read.nbytes
        torch.stack(
            blocks,
            out=out[offset:end].view(read.dtype).view(len(blocks), *read.block_shape),
        )
        offset = end


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
