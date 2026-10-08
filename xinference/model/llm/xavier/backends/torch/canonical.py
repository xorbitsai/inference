# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Owned, bit-preserving FP16 payloads for the shared KV contract.

These synchronous conversion helpers establish a correctness boundary for new
adapters. Existing same-engine GPU handoff keeps its current layout and path.
"""

import sys
from typing import Tuple, Union

import torch

from ...contract import KVCacheContract, KVLayerMetadata


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise ValueError("KV protocol v1 requires little-endian payloads")


def _layer_shape(
    contract: KVCacheContract, num_blocks: int, layout: str
) -> Tuple[int, ...]:
    if type(num_blocks) is not int or num_blocks <= 0:
        raise ValueError("A positive number of complete KV blocks is required")
    tail = (contract.block_size, contract.num_kv_heads, contract.head_dim)
    if layout == "N2HTD":
        tail = (contract.num_kv_heads, contract.block_size, contract.head_dim)
    elif layout != "N2THD":
        raise ValueError("Unsupported physical KV layout")
    return (num_blocks, 2, *tail)


def encode_layer(
    tensor: torch.Tensor, metadata: KVLayerMetadata, layout: str
) -> torch.Tensor:
    """Export block-major split K/V as an owned contiguous CPU byte tensor."""
    _require_little_endian()
    if tensor.ndim != 5 or tuple(tensor.shape) != _layer_shape(
        metadata.contract, len(metadata.keys), layout
    ):
        raise ValueError("KV layer geometry differs from the contract")
    if tensor.dtype != torch.float16:
        raise ValueError("KV logical dtype differs from the contract")
    if layout == "N2HTD":
        tensor = tensor.permute(0, 1, 3, 2, 4)
    tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
    return tensor.view(torch.uint8).reshape(-1)


def decode_layer(
    payload: torch.Tensor,
    metadata: KVLayerMetadata,
    expected: KVLayerMetadata,
    layout: str,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Import canonical bytes into an owned block-major split K/V tensor."""
    _require_little_endian()
    metadata.require_match(expected)
    contract = expected.contract
    num_blocks = len(expected.keys)
    shape = _layer_shape(contract, num_blocks, layout)
    if (
        payload.dtype != torch.uint8
        or payload.device.type != "cpu"
        or payload.ndim != 1
        or not payload.is_contiguous()
        or payload.numel() != num_blocks * contract.layer_nbytes
    ):
        raise ValueError("Invalid canonical KV payload")
    # A byte slice can be contiguous while starting at an odd storage offset.
    # Own and align the bytes before reinterpreting their 16-bit representation.
    tensor = (
        payload.clone()
        .view(torch.float16)
        .reshape(_layer_shape(contract, num_blocks, "N2THD"))
    )
    if layout == "N2HTD":
        tensor = tensor.permute(0, 1, 3, 2, 4)
    return tensor.to(device=device).contiguous().reshape(shape)


def export_token_slots(
    keys: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    metadata: KVLayerMetadata,
) -> torch.Tensor:
    """Gather explicit [slot, KV head, dimension] pools into canonical bytes."""
    contract = metadata.contract
    expected = (contract.num_kv_heads, contract.head_dim)
    if (
        keys.ndim != 3
        or tuple(keys.shape[1:]) != expected
        or values.shape != keys.shape
        or keys.dtype != torch.float16
        or values.dtype != keys.dtype
        or keys.device != values.device
    ):
        raise ValueError("KV token pool differs from the contract")
    if (
        slots.ndim != 1
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.numel()
        or slots.numel() % contract.block_size
        or slots.numel() != len(metadata.keys) * contract.block_size
    ):
        raise ValueError("KV slots must identify complete blocks")
    slots = slots.to(device=keys.device, dtype=torch.long)
    shape = (-1, contract.block_size, *expected)
    tensor = torch.stack(
        [
            keys.index_select(0, slots).reshape(shape),
            values.index_select(0, slots).reshape(shape),
        ],
        dim=1,
    )
    return encode_layer(tensor, metadata, "N2THD")
