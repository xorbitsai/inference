# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from ..core import SGLANGModel


def _model(enabled=True, lower_first=False):
    model = object.__new__(SGLANGModel)
    model._model_config = {
        "enable_priority_scheduling": enabled,
        "schedule_low_priority_values_first": lower_first,
        "default_priority_value": 17,
    }
    model.model_uid = "priority-model-0"
    model._active_request_ids = set()
    model.abort_request = AsyncMock()
    model._engine = SimpleNamespace(generate_url="http://engine/generate")
    model.model_family = SimpleNamespace(has_architecture=lambda name: False)
    model.model_spec = SimpleNamespace(model_format="pytorch")
    model._n_worker = 1
    model._get_cuda_count = lambda: 1
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("priority", [-10, 0, 10, None])
@pytest.mark.parametrize("lower_first", [False, True])
async def test_priority_is_top_level_http_metadata(
    monkeypatch, stream, priority, lower_first
):
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.post.side_effect = RuntimeError("engine reached")
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kwargs: session)
    model = _model(lower_first=lower_first)
    config = {"stream": stream}
    if priority is not None:
        config["priority"] = priority
    with pytest.raises(RuntimeError, match="engine reached"):
        result = await model.async_generate("hello", config)
        if stream:
            await anext(result)
    payload = session.post.call_args.kwargs["json"]
    # The unified request default is 0, overriding SGLang’s native default.
    expected = priority or 0
    assert payload["priority"] == (expected if lower_first else -expected)
    assert "priority" not in payload["sampling_params"]


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [-1, True, 1.5, "1", None])
async def test_priority_rejected_before_http_when_invalid_or_disabled(priority):
    model = _model(enabled=False)
    with pytest.raises(ValueError, match="priority"):
        await model.async_generate("hello", {"priority": priority})


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}, {"priority": 0}])
async def test_default_priority_is_omitted_when_scheduling_disabled(config):
    model = _model(enabled=False)
    model._active_request_ids = set()
    model._non_stream_generate = AsyncMock(side_effect=RuntimeError("engine reached"))
    with pytest.raises(RuntimeError, match="engine reached"):
        await model.async_generate("hello", config)
    assert "priority" not in model._non_stream_generate.call_args.kwargs
