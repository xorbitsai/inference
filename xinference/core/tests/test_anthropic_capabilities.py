# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ...api.protocols.anthropic_capabilities import (
    AnthropicBackendCapabilities,
    SupportLevel,
)
from ..model import ModelActor
from ..pd_model import PDModelActor


def _model_actor():
    actor = ModelActor.__new__(ModelActor)
    actor._replica_model_uid = "model-1-0"
    actor._model_state = "ready"
    actor._lock = None
    actor._model = SimpleNamespace(
        get_anthropic_capabilities=lambda: AnthropicBackendCapabilities(
            engine="vllm",
            vision=SupportLevel.NATIVE,
        )
    )
    return actor


@pytest.mark.asyncio
async def test_model_actor_capabilities_are_json_serializable():
    payload = await _model_actor().get_anthropic_capabilities()

    description = json.loads(payload)
    assert description["engine"] == "vllm"
    assert description["vision"] == "native"
    assert description["supported_context_edits"] == ["clear_thinking_20251015"]


@pytest.mark.asyncio
async def test_pd_model_forwards_serialized_decode_capabilities():
    payload = await _model_actor().get_anthropic_capabilities()
    decode = SimpleNamespace(get_anthropic_capabilities=AsyncMock(return_value=payload))
    actor = PDModelActor("pd")
    actor._decode_replicas = {"decode": decode}

    forwarded = await actor.get_anthropic_capabilities()

    assert json.loads(forwarded)["supported_context_edits"] == [
        "clear_thinking_20251015"
    ]
    decode.get_anthropic_capabilities.assert_awaited_once_with()
