# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ...model.llm.weight_cache import RELOAD_FIELDS
from ..restful_api import RESTfulAPI


def make_api():
    api = RESTfulAPI.__new__(RESTfulAPI)
    supervisor = MagicMock()
    supervisor.get_model_reload_config = AsyncMock(
        return_value={"engine": "vllm", "parameters": RELOAD_FIELDS["vllm"]}
    )
    supervisor.reload_model = AsyncMock(
        return_value={"operation_id": "one", "status": "reloading"}
    )
    api._get_supervisor_ref = AsyncMock(return_value=supervisor)
    return api, supervisor


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"model_config": []},
        {"model_config": {"max_num_seqs": True}},
        {"model_config": {"dtype": "float16"}},
        {"model_config": {"max_num_seqs": 8}, "drain_timeout": True},
        {"model_config": {"max_num_seqs": 8}, "drain_timeout": float("inf")},
        {"model_config": {"max_num_seqs": 8}, "ignored": 1},
    ],
)
async def test_invalid_reload_is_not_scheduled(body):
    api, supervisor = make_api()
    request = MagicMock(json=AsyncMock(return_value=body))
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 400
    supervisor.reload_model.assert_not_called()


@pytest.mark.asyncio
async def test_reload_accepted_and_duplicate_conflict():
    api, supervisor = make_api()
    request = MagicMock(
        json=AsyncMock(return_value={"model_config": {"max_num_seqs": 8}})
    )
    response = await api.reload_model("test", request)
    assert response.status_code == 202
    assert json.loads(response.body)["operation_id"] == "one"
    supervisor.reload_model.assert_awaited_once_with("test", {"max_num_seqs": 8}, 300)
    supervisor.reload_model.side_effect = RuntimeError("already reloading")
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_loading_model_returns_retryable_status():
    from ...core.exceptions import ModelNotReadyError

    api, supervisor = make_api()
    supervisor.reload_model.side_effect = ModelNotReadyError("Model is loading")
    request = MagicMock(
        json=AsyncMock(return_value={"model_config": {"max_num_seqs": 8}})
    )
    with pytest.raises(HTTPException) as exc:
        await api.reload_model("test", request)
    assert exc.value.status_code == 503
