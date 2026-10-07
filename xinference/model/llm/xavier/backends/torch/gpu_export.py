# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Bounded reusable GPU exports with explicit producer/consumer completion."""

import uuid
from typing import List, Tuple

import torch


class GPUExportArena:
    def __init__(self, slot_bytes: int, slots: int, device: torch.device):
        self.source = uuid.uuid4().hex
        self.slot_bytes = slot_bytes
        self.device = device
        self.buffers = [
            torch.empty(slot_bytes, dtype=torch.uint8, device=device)
            for _ in range(slots)
        ]
        self.available = list(range(slots))
        self.events = [torch.cuda.Event(interprocess=True) for _ in range(slots)]
        for event in self.events:
            event.record(torch.cuda.current_stream(device))
        self.handles = [event.ipc_handle() for event in self.events]

    def allocate(self, shape: Tuple[int, int]) -> Tuple[int, torch.Tensor]:
        if shape[0] * shape[1] > self.slot_bytes:
            raise ValueError("GPU export exceeds one arena slot")
        slot = self.available.pop()
        return slot, self.buffers[slot][: shape[0] * shape[1]].view(shape)

    def record(self, slot: int, shape: Tuple[int, int]) -> dict:
        self.events[slot].record(torch.cuda.current_stream(self.device))
        return {"gpu_export": self.source, "slot": slot, "shape": shape}

    def metadata(self) -> list:
        from torch.multiprocessing.reductions import reduce_tensor

        # Issue one IPC refcount per actor lifetime. Reusing an old descriptor
        # after actor recovery would reuse a released refcount.
        return [
            (reduce_tensor(buffer)[1], handle)
            for buffer, handle in zip(self.buffers, self.handles)
        ]

    def release(self, slots: List[int]) -> None:
        for slot in slots:
            self.events[slot].synchronize()
            if slot not in self.available:
                self.available.append(slot)

    def close(self) -> None:
        for event in self.events:
            event.synchronize()
        self.buffers.clear()
        self.available.clear()
