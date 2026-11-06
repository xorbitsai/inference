# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from .. import core


def _model(monkeypatch, *, policy="priority", vllm_version="0.21.0"):
    class SamplingParams:
        def __init__(self, n=1, max_tokens=2):
            self.max_tokens = max_tokens

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_VERSION", Version(vllm_version))
    model = object.__new__(core.VLLMModel)
    model._engine = SimpleNamespace(
        generate=Mock(side_effect=RuntimeError("engine reached"))
    )
    model._model_config = {"scheduling_policy": policy}
    model._nixl_config = model._xavier_config = None
    model.reasoning_parser = None
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("priority", [-10, 0, 10])
async def test_priority_is_engine_metadata_not_sampling_params(
    monkeypatch, stream, priority
):
    model = _model(monkeypatch)
    with pytest.raises(RuntimeError, match="engine reached"):
        await core.VLLMModel.async_generate.__wrapped__(
            model, "hello", {"priority": priority, "stream": stream, "max_tokens": 2}
        )
    call = model._engine.generate.call_args
    assert call.kwargs.get("priority", 0) == priority
    assert not hasattr(call.args[1], "priority")


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}, {"priority": 0}])
async def test_default_priority_preserves_legacy_engines(monkeypatch, config):
    model = _model(monkeypatch, policy="fcfs", vllm_version="0.6.2")
    with pytest.raises(RuntimeError, match="engine reached"):
        await core.VLLMModel.async_generate.__wrapped__(
            model, "hello", dict(config, max_tokens=2)
        )
    assert "priority" not in model._engine.generate.call_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,vllm_version,message",
    [
        ("fcfs", "0.21.0", "scheduling_policy=priority"),
        ("priority", "0.6.2", "vLLM >= 0.6.3"),
    ],
)
async def test_nonzero_priority_requires_engine_support(
    monkeypatch, policy, vllm_version, message
):
    model = _model(monkeypatch, policy=policy, vllm_version=vllm_version)
    with pytest.raises(ValueError, match=message):
        await core.VLLMModel.async_generate.__wrapped__(
            model, "hello", {"priority": -1}
        )
    model._engine.generate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [True, 1.5, "1", None])
async def test_direct_generation_validates_priority(monkeypatch, priority):
    model = _model(monkeypatch)
    with pytest.raises(ValueError, match="priority must be an integer"):
        await core.VLLMModel.async_generate.__wrapped__(
            model, "hello", {"priority": priority}
        )
    model._engine.generate.assert_not_called()
