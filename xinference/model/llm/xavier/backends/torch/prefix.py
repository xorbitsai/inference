# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Owned CPU pages in [layer, K/V, token, head, dimension] order."""

import torch


def pack_pages(views, slots, contract):
    """Snapshot all layers before the caller releases engine-owned GPU slots."""
    data = (
        torch.stack(
            [
                torch.stack(
                    [
                        views[str(layer + kv * contract.num_layers)][slots]
                        for kv in (0, 1)
                    ],
                    dim=1,
                )
                for layer in range(contract.num_layers)
            ],
            dim=1,
        )
        .cpu()
        .contiguous()
    )
    return [page.view(torch.uint8).numpy().tobytes() for page in data]


def unpack_pages(views, slots, pages, contract):
    expected = contract.num_layers * contract.layer_nbytes
    if len(slots) != len(pages) or any(
        type(page) is not bytes or len(page) != expected for page in pages
    ):
        raise ValueError("Incomplete shared-prefix KV pages")
    if not pages:
        return
    shape = (
        len(pages),
        contract.num_layers,
        2,
        contract.block_size,
        contract.num_kv_heads,
        contract.head_dim,
    )
    data = (
        torch.frombuffer(bytearray(b"".join(pages)), dtype=torch.uint8)
        .view(getattr(torch, contract.logical_dtype))
        .reshape(shape)
    )
    for layer in range(contract.num_layers):
        for kv in (0, 1):
            target = views[str(layer + kv * contract.num_layers)]
            target[slots] = data[:, layer, kv].to(target.device)
    # Engine slots may be reused as soon as this callback returns.
    devices = {view.device for view in views.values() if view.is_cuda}
    for device in devices:
        torch.cuda.synchronize(device)
