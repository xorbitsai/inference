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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["supported", "non-hybrid", "pd", "no-tracker"])
async def test_cache_updates_require_actual_model_support(monkeypatch, mode):
    owner = SupervisorActor.__new__(SupervisorActor)
    owner._pd_model_mapping = {"model": object()} if mode == "pd" else {}
    tracker = SimpleNamespace(
        get_cache_sessions=AsyncMock(return_value={}),
        update_cache_session=AsyncMock(return_value=True),
    )
    owner._block_tracker_mapping = {} if mode == "no-tracker" else {"model": tracker}
    owner._iter_active_replica_model_uids = lambda _: ["rep0"]
    owner._replica_model_uid_to_worker = {"rep0": object()}
    model = SimpleNamespace(
        supports_agent_cache_lifecycle=AsyncMock(return_value=mode == "supported")
    )
    monkeypatch.setattr(
        supervisor_module, "call_worker_route", AsyncMock(return_value=model)
    )
    session = {"id": "agent", "revision": 0, "phase": "active"}
    if mode == "supported":
        assert await owner.update_cache_session("model", session)
        tracker.update_cache_session.assert_awaited_once_with(session)
    else:
        with pytest.raises(ValueError, match="does not support"):
            await owner.update_cache_session("model", session)
        tracker.update_cache_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_probes_independent_capabilities_concurrently(monkeypatch):
    import asyncio

    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    owner = SupervisorActor.__new__(SupervisorActor)
    owner.describe_model = AsyncMock(return_value={"model_type": "LLM"})
    thinking_started, lifecycle_started = asyncio.Event(), asyncio.Event()

    async def thinking(uid):
        thinking_started.set()
        await lifecycle_started.wait()
        return True

    async def lifecycle(uid):
        lifecycle_started.set()
        await thinking_started.wait()
        return True

    owner._supports_thinking_budget = thinking
    owner._supports_agent_cache_lifecycle = lifecycle
    description = await asyncio.wait_for(owner.describe_running_model("model"), 1)
    assert description["supports_thinking_budget"]
    assert description["supports_agent_cache_lifecycle"]
