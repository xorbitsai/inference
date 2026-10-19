# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from xinference.cache_isolation import cache_scope

from ..pd_contract import prepare_pd_request, prompt_digest


@pytest.mark.asyncio
async def test_cross_engine_preflight_binds_scope_before_actor_call(
    monkeypatch, kv_contract
):
    ref = SimpleNamespace(prepare=AsyncMock())
    actor_ref = AsyncMock(return_value=ref)
    monkeypatch.setattr("xoscar.actor_ref", actor_ref)
    config = dict(
        role="prefill", contract=kv_contract.to_dict(), address="a", uid="cache", rank=1
    )
    params = {
        "sglang_xavier": {"mode": "gpu", "room": 1, "cache_scope": cache_scope("a")}
    }
    tokens = [1, 2, 3]
    await prepare_pd_request(config, tokens, params, "a")
    assert params["xavier_prompt_digest"] == prompt_digest(tokens, "a")
    assert params["xavier_prompt_digest"] != prompt_digest(tokens, "b")
    assert params["xavier_prompt_digest"] != prompt_digest(tokens)
    assert ref.prepare.call_args.args[2] == prompt_digest(tokens, "a")
    actor_ref.reset_mock()
    for salt in (None, "b"):
        with pytest.raises(ValueError, match="scope mismatch"):
            await prepare_pd_request(config, tokens, params, salt)
    actor_ref.assert_not_called()
