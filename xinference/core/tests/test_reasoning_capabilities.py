# CI: linux
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from .. import supervisor as supervisor_module
from ..supervisor import SupervisorActor


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [True, False, "legacy", "unavailable"])
async def test_running_model_budget_capability_requires_every_replica(
    monkeypatch, second
):
    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    owner = SupervisorActor.__new__(SupervisorActor)
    owner._pd_model_mapping = {}
    owner._iter_active_replica_model_uids = lambda _: ["rep0", "rep1"]
    owner._replica_model_uid_to_worker = {"rep0": object(), "rep1": [object()]}
    owner.describe_model = AsyncMock(
        return_value={
            "model_type": "LLM",
            "model_ability": ["chat"],
            "reasoning_effort": None,
        }
    )
    models = [
        SimpleNamespace(
            get_reasoning_capabilities=AsyncMock(
                return_value=json.dumps({"supports_thinking_budget": True}).encode()
            )
        ),
        SimpleNamespace(
            get_reasoning_capabilities=AsyncMock(
                return_value={"supports_thinking_budget": second}
            )
        ),
    ]
    if second == "legacy":
        models[1] = SimpleNamespace()
    elif second == "unavailable":
        models[1].get_reasoning_capabilities.side_effect = RuntimeError("loading")

    async def get_model(owner, uid, worker, operation, **kwargs):
        assert operation == "get_model"
        return models[int(uid[-1])]

    monkeypatch.setattr(supervisor_module, "call_worker_route", get_model)
    description = await owner.describe_running_model("model")
    assert description["supports_thinking_budget"] is (second is True)
    assert description["supports_chat_session_affinity"] is True
    assert description["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_other_model_types_do_not_probe_reasoning(monkeypatch):
    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    owner = SupervisorActor.__new__(SupervisorActor)
    owner.describe_model = AsyncMock(return_value={"model_type": "image"})
    owner._supports_thinking_budget = AsyncMock()
    assert await owner.describe_running_model("image") == {"model_type": "image"}
    owner._supports_thinking_budget.assert_not_called()


@pytest.mark.asyncio
async def test_pd_description_probes_router_without_selecting_a_replica(monkeypatch):
    monkeypatch.setattr(supervisor_module, "XINFERENCE_TOKEN_ROUTER_ENABLED", False)
    owner = SupervisorActor.__new__(SupervisorActor)
    owner.describe_model = AsyncMock(
        return_value={"model_type": "LLM", "model_ability": ["chat"]}
    )
    owner._pd_model_mapping = {
        "pd": SimpleNamespace(
            get_reasoning_capabilities=AsyncMock(
                return_value={"supports_thinking_budget": True}
            )
        )
    }
    description = await owner.describe_running_model("pd")
    assert description["supports_thinking_budget"] is True
    assert "supports_chat_session_affinity" not in description


@pytest.mark.asyncio
async def test_actor_capability_query_serializes_without_reserving_generation():
    from ..model import ModelActor

    model = SimpleNamespace(
        model_family=SimpleNamespace(to_description=lambda: {"model_type": "LLM"}),
        get_reasoning_capabilities=lambda: {"supports_thinking_budget": True},
    )
    actor = ModelActor("test:123", "test:345", model, "reasoning-test")
    actor._model_state = "ready"
    assert json.loads(await actor.get_reasoning_capabilities()) == {
        "supports_thinking_budget": True
    }
    assert actor._serve_count == 0
