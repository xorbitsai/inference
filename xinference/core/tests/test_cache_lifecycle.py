# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from packaging.version import parse

from ...model.llm.vllm import core as vllm_core
from .. import supervisor as supervisor_module
from ..supervisor import SupervisorActor


@pytest.mark.asyncio
@pytest.mark.parametrize("peer", [True, False, "legacy", "unavailable"])
async def test_cache_capability_requires_all_replicas(monkeypatch, peer):
    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    owner = SupervisorActor.__new__(SupervisorActor)
    owner._pd_model_mapping = {}
    owner._block_tracker_mapping = {
        "model": SimpleNamespace(get_cache_sessions=AsyncMock(return_value={}))
    }
    owner._replica_model_uid_to_worker = {"rep0": object(), "rep1": [object()]}
    owner._iter_active_replica_model_uids = lambda _: ["rep0", "rep1"]
    owner._supports_thinking_budget = AsyncMock(return_value=False)
    owner.describe_model = AsyncMock(return_value={"model_type": "LLM"})
    models = [
        SimpleNamespace(supports_agent_cache_lifecycle=AsyncMock(return_value=True)),
        SimpleNamespace(supports_agent_cache_lifecycle=AsyncMock(return_value=peer)),
    ]
    if peer == "legacy":
        models[1] = SimpleNamespace()
    elif peer == "unavailable":
        models[1].supports_agent_cache_lifecycle.side_effect = RuntimeError("loading")

    async def get_model(owner, uid, worker, operation, **kwargs):
        return models[int(uid[-1])]

    monkeypatch.setattr(supervisor_module, "call_worker_route", get_model)
    result = await owner.describe_running_model("model")
    assert (result.get("supports_agent_cache_lifecycle") is True) is (peer is True)


@pytest.mark.parametrize(
    "hybrid,budget,role,engine_version,expected",
    [
        (True, 0, "hybrid", "0.32.0", True),
        (True, None, "hybrid", "0.32.0", False),
        (False, 0, "hybrid", "0.32.0", False),
        (True, 0, "prefill", "0.32.0", False),
        (True, 0, "hybrid", "0.31.0", False),
    ],
)
def test_only_supported_gpu_prefix_models_advertise_lifecycle(
    monkeypatch, hybrid, budget, role, engine_version, expected
):
    model = vllm_core.VLLMModel.__new__(vllm_core.VLLMModel)
    model._xavier_config = {
        "gpu_cache_bytes": budget,
        "role": role,
        "block_tracker_uid": "tracker",
    }
    model._engine = SimpleNamespace(model_config=SimpleNamespace(is_hybrid=hybrid))
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", parse(engine_version))
    assert model.supports_agent_cache_lifecycle() is expected
