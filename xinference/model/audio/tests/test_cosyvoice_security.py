# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from .... import constants
from ...cache_manager import CacheManager
from .. import BUILTIN_AUDIO_MODELS
from .. import core as audio_core
from ..cosyvoice import CosyVoiceModel


@pytest.fixture
def cosy_spec(monkeypatch):
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", False)
    spec = BUILTIN_AUDIO_MODELS["CosyVoice-300M"][0].copy(deep=True)
    monkeypatch.setattr(audio_core, "match_audio", Mock(return_value=spec))
    monkeypatch.setattr(
        audio_core,
        "resolve_audio_model_name_and_engine",
        lambda *args, **kwargs: (spec.model_name, None),
    )
    return spec


@pytest.mark.parametrize("builtin", [False, True])
def test_explicit_cosyvoice_path_requires_operator_trust(
    cosy_spec, monkeypatch, builtin
):
    cosy_spec.is_builtin = builtin
    cache = Mock()
    monkeypatch.setattr(CacheManager, "cache", cache)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        audio_core.create_audio_model_instance(
            "test", cosy_spec.model_name, model_path="/mnt/models/operator-model"
        )
    cache.assert_not_called()
    assert cosy_spec.is_builtin is builtin


def test_custom_cosyvoice_hub_model_rejected_before_download(cosy_spec, monkeypatch):
    cosy_spec.is_builtin = False
    cache = Mock()
    monkeypatch.setattr(CacheManager, "cache", cache)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        audio_core.create_audio_model_instance(
            "test", cosy_spec.model_name, trust_remote_code=True, is_builtin=True
        )
    cache.assert_not_called()


def test_untrusted_cosyvoice_load_rejected_before_vendor_import(cosy_spec, monkeypatch):
    cosy_spec.is_builtin = False
    vendor = ModuleType("cosyvoice.cli.cosyvoice")
    constructor = Mock(side_effect=AssertionError("vendor loader must not run"))
    vendor.CosyVoice = constructor
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", vendor)
    model = CosyVoiceModel("test", "/mnt/untrusted", cosy_spec)
    with pytest.raises(ValueError, match="XINFERENCE_TRUST_REMOTE_CODE=1"):
        model.load()
    constructor.assert_not_called()


@pytest.mark.parametrize("builtin", [False, True])
def test_operator_can_load_trusted_cosyvoice_local_path(
    cosy_spec, monkeypatch, builtin
):
    cosy_spec.is_builtin = builtin
    monkeypatch.setattr(constants, "XINFERENCE_TRUST_REMOTE_CODE", True)
    vendor = ModuleType("cosyvoice.cli.cosyvoice")
    constructor = Mock()
    vendor.CosyVoice = constructor
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", vendor)
    model = audio_core.create_audio_model_instance(
        "test", cosy_spec.model_name, model_path="/mnt/models/operator-model"
    )
    model.load()
    constructor.assert_called_once_with("/mnt/models/operator-model", load_jit=False)
    assert cosy_spec.is_builtin is builtin


def test_builtin_cosyvoice_uses_registered_cache_without_opt_in(cosy_spec, monkeypatch):
    cosy_spec.is_builtin = True
    monkeypatch.setattr(
        CacheManager, "cache", Mock(return_value="/mnt/cache/cosyvoice")
    )
    vendor = ModuleType("cosyvoice.cli.cosyvoice")
    constructor = Mock()
    vendor.CosyVoice = constructor
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", vendor)
    model = audio_core.create_audio_model_instance("test", cosy_spec.model_name)
    model.load()
    constructor.assert_called_once_with("/mnt/cache/cosyvoice", load_jit=False)
