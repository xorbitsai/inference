# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from unittest.mock import AsyncMock

import pytest
import torch
import xoscar as xo

from ...xavier.backends.torch.storage import XavierCacheActor
from ...xavier.contract import KVCacheContract
from ..core import SGLANGModel
from ..xavier.pd import SGLangXavierHandoff


@pytest.fixture
async def deployment(monkeypatch):
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierCacheActor, 768, address=pool.external_address, uid="cache"
        )
        contract = KVCacheContract(
            "a" * 64, "b" * 64, "c" * 64, "d" * 64, 3, 2, 4, 4, "float16"
        )
        namespace = await actor.configure(contract.to_dict(), {"layout": "layer_first"})
        await actor.put(namespace, ["prompt"], [torch.zeros(384, dtype=torch.uint8)])
        # Only the engine's tokenizer/hash boundary is mocked. Real actor RPCs,
        # leases, model generation wrappers and response metadata are exercised.
        monkeypatch.setattr(
            SGLangXavierHandoff, "_keys", lambda self, prompt, salt: [prompt]
        )
        models = []
        for role in ("prefill", "decode"):
            model = object.__new__(SGLANGModel)
            model.model_uid = "pd-" + role
            model._active_request_ids = set()
            model._xavier_handoff = SGLangXavierHandoff(
                {"address": actor.address, "uid": "cache", "role": role}, 4, None
            )
            model._non_stream_generate = AsyncMock(
                return_value={
                    "text": "output",
                    "meta_info": {
                        "prompt_tokens": 7,
                        "completion_tokens": 1,
                        "cached_tokens": 4,
                        "finish_reason": {"type": "length"},
                    },
                }
            )
            model.abort_request = AsyncMock(return_value="DONE")
            models.append(model)
        yield actor, models[0], models[1]


async def _prefill(model):
    return await model.async_generate(
        "prompt",
        generate_config={
            "max_new_tokens": 20,
            "n": 1,
            "stream": True,
            "_pd_kv_transfer_params": {"do_remote_decode": True},
        },
        request_id="r",
    )


@pytest.mark.asyncio
async def test_prefill_acknowledges_published_pages_and_decode_releases(deployment):
    actor, prefill, decode = deployment
    result = await _prefill(prefill)
    args = prefill._non_stream_generate.call_args.kwargs
    assert args["max_new_tokens"] == 1 and "n" not in args
    transfer = result["_pd_kv_transfer_params"]
    assert transfer["sglang_xavier"]["cached_tokens"] == 4
    assert (await actor.get_stats())["active_handoffs"] == 1
    response = await decode.async_generate(
        "prompt",
        generate_config={
            "_pd_kv_transfer_params": transfer,
        },
    )
    assert response["choices"][0]["text"] == "output"
    assert (await actor.get_stats())["active_handoffs"] == 0
    assert not prefill._active_request_ids and not decode._active_request_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["miss", "exception", "cancel"])
async def test_decode_failure_releases_pd_lease(deployment, failure):
    actor, prefill, decode = deployment
    transfer = (await _prefill(prefill))["_pd_kv_transfer_params"]
    if failure == "miss":
        decode._non_stream_generate.return_value["meta_info"]["cached_tokens"] = 0
    else:
        decode._non_stream_generate.side_effect = (
            asyncio.CancelledError() if failure == "cancel" else RuntimeError("engine")
        )
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        await decode.async_generate(
            "prompt", generate_config={"_pd_kv_transfer_params": transfer}
        )
    assert (await actor.get_stats())["active_handoffs"] == 0
    assert not decode._active_request_ids
    if failure != "miss":
        decode.abort_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["engine", "cancel", "publish"])
async def test_prefill_failure_releases_pending_capacity(deployment, failure):
    actor, prefill, _ = deployment
    error = asyncio.CancelledError() if failure == "cancel" else RuntimeError("failed")
    if failure == "publish":
        prefill._xavier_handoff.publish = AsyncMock(side_effect=error)
    else:
        prefill._non_stream_generate.side_effect = error
    with pytest.raises(type(error)):
        await _prefill(prefill)
    assert (await actor.get_stats())["active_handoffs"] == 0
    assert not prefill._active_request_ids


@pytest.mark.asyncio
async def test_invalid_options_do_not_reserve_handoff(deployment, monkeypatch):
    actor, prefill, _ = deployment
    monkeypatch.setattr(
        prefill,
        "_sanitize_generate_config",
        lambda config: (_ for _ in ()).throw(ValueError("options")),
    )
    with pytest.raises(ValueError, match="options"):
        await _prefill(prefill)
    assert (await actor.get_stats())["active_handoffs"] == 0
    prefill._non_stream_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_wrong_prompt_and_role_rejected_before_decode(deployment):
    actor, prefill, decode = deployment
    transfer = (await _prefill(prefill))["_pd_kv_transfer_params"]
    with pytest.raises(ValueError, match="prompt"):
        await decode.async_generate(
            "different", generate_config={"_pd_kv_transfer_params": transfer}
        )
    decode._non_stream_generate.assert_not_awaited()
    with pytest.raises(ValueError, match="decode replica"):
        await prefill.async_generate(
            "prompt", generate_config={"_pd_kv_transfer_params": transfer}
        )
    await actor.release_handoff(transfer["sglang_xavier"]["ticket"])
    with pytest.raises(RuntimeError, match="expired"):
        await decode.async_generate(
            "prompt", generate_config={"_pd_kv_transfer_params": transfer}
        )


@pytest.mark.asyncio
async def test_stream_disconnect_aborts_engine_and_releases_lease(deployment):
    actor, prefill, decode = deployment
    transfer = (await _prefill(prefill))["_pd_kv_transfer_params"]

    async def chunks(*args, **kwargs):
        yield {
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "cached_tokens": 4,
            "finish_reason": None,
        }, "first"
        await asyncio.Event().wait()

    decode._stream_generate = chunks
    stream = await decode.async_generate(
        "prompt",
        generate_config={
            "stream": True,
            "_pd_kv_transfer_params": transfer,
        },
        request_id="r",
    )
    assert (await anext(stream))["choices"][0]["text"] == "first"
    await stream.aclose()
    decode.abort_request.assert_awaited_once_with("r")
    assert (await actor.get_stats())["active_handoffs"] == 0
    assert not decode._active_request_ids
