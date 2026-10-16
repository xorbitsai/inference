# Copyright 2022-2026 Xinference Holdings Pte. Ltd

from __future__ import annotations

import warnings

import torch

from ..core import _resolve_sentence_transformer_device


class _PublicDeviceModel:
    def __init__(self, device):
        self.device = device

    @property
    def _target_device(self):
        raise AssertionError("Deprecated _target_device must not be accessed")


class _ExplicitDeviceModel:
    @property
    def device(self):
        raise AssertionError("Model device must not be read when device is explicit")

    @property
    def _target_device(self):
        raise AssertionError("Deprecated _target_device must not be accessed")


def test_resolve_sentence_transformer_device_uses_public_property() -> None:
    model = _PublicDeviceModel(torch.device("cpu"))

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        device = _resolve_sentence_transformer_device(model, None)

    assert device == torch.device("cpu")
    assert not any("_target_device" in str(item.message) for item in records)


def test_resolve_sentence_transformer_device_preserves_explicit_device() -> None:
    model = _ExplicitDeviceModel()

    assert _resolve_sentence_transformer_device(model, "cuda:1") == "cuda:1"


def test_resolve_sentence_transformer_device_preserves_supported_values() -> None:
    for device in ("cpu", "cuda", "cuda:1", "mps"):
        model = _PublicDeviceModel(device)
        assert _resolve_sentence_transformer_device(model, None) == device
