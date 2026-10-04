# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import xoscar as xo

from ..core import SGLANGModel
from ..xavier.directory import XavierPDDirectory
from ..xavier.pd import SGLangXavierHandoff


@pytest.fixture
async def deployment():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        actor = await xo.create_actor(
            XavierPDDirectory, address=pool.external_address, uid="cache"
        )
        await actor.configure("gpu-namespace")
        models = []
        for role in ("prefill", "decode"):
            model = object.__new__(SGLANGModel)
            model.model_uid = "pd-" + role
            model._active_request_ids = set()
            model._xavier_handoff = SGLangXavierHandoff(
                dict(address=actor.address, uid="cache", role=role),
                4,
                SimpleNamespace(encode=lambda text: list(text.encode())),
            )
            model._non_stream_generate = AsyncMock(
                return_value=dict(
                    text="output",
                    meta_info=dict(
                        prompt_tokens=7,
                        completion_tokens=1,
                        cached_tokens=0,
                        finish_reason={"type": "length"},
                    ),
                )
            )
            model.abort_request = AsyncMock(return_value="DONE")
            models.append(model)
        yield actor, models[0], models[1]


def config(role, **kwargs):
    return dict(
        max_tokens=16,
        _pd_kv_transfer_params=dict(
            **(
                {"do_remote_decode": True}
                if role == "prefill"
                else {"do_remote_prefill": True}
            ),
            sglang_xavier=dict(mode="gpu", room=123),
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_paired_generation_uses_native_bootstrap_and_gpu_completion(deployment):
    actor, prefill, decode = deployment

    async def generated(*args, **kwargs):
        await actor.complete(123, 4096)
        return prefill._non_stream_generate.return_value

    decode._non_stream_generate.side_effect = generated
    p, d = await asyncio.gather(
        prefill.async_generate("prompt", config("prefill"), request_id="r"),
        decode.async_generate("prompt", config("decode"), request_id="r"),
    )
    assert p["_pd_kv_transfer_params"]["sglang_xavier"]["mode"] == "gpu"
    assert d["choices"][0]["text"] == "output"
    for model in (prefill, decode):
        sent = model._non_stream_generate.call_args.kwargs
        assert sent["bootstrap_room"] == 123 and sent["bootstrap_host"] == "xavier"
        assert "cache_salt" not in sent
        assert not model._active_request_ids
    stats = await actor.get_stats()
    assert stats["gpu_bytes"] == 4096 and stats["completed_requests"] == 1
    assert stats["active_handoffs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["miss", "exception", "cancel"])
async def test_decode_failure_aborts_engine_and_clears_metadata(deployment, failure):
    actor, _, decode = deployment
    await actor.prepare(123, "gpu-namespace", "unused", "prefill")
    # Match the wrapper's exact prompt fingerprint without starting P inference.
    import hashlib
    import json

    await actor.release(123)
    await actor.prepare(
        123,
        "gpu-namespace",
        hashlib.sha256(json.dumps(list(b"prompt")).encode()).hexdigest(),
        "prefill",
    )
    if failure != "miss":
        decode._non_stream_generate.side_effect = (
            asyncio.CancelledError() if failure == "cancel" else RuntimeError("engine")
        )
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        await decode.async_generate("prompt", config("decode"), request_id="r")
    decode.abort_request.assert_awaited_once_with("r")
    assert not decode._active_request_ids
    assert (await actor.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["engine", "cancel"])
async def test_prefill_failure_clears_metadata(deployment, failure):
    actor, prefill, _ = deployment
    prefill._non_stream_generate.side_effect = (
        asyncio.CancelledError() if failure == "cancel" else RuntimeError("engine")
    )
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        await prefill.async_generate("prompt", config("prefill"), request_id="r")
    assert (await actor.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
async def test_invalid_options_do_not_reserve_handoff(deployment, monkeypatch):
    actor, prefill, _ = deployment
    monkeypatch.setattr(
        prefill,
        "_sanitize_generate_config",
        lambda config: (_ for _ in ()).throw(ValueError("options")),
    )
    with pytest.raises(ValueError, match="options"):
        await prefill.async_generate("prompt", config("prefill"))
    assert (await actor.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
async def test_wrong_prompt_and_role_rejected_before_decode(deployment):
    actor, prefill, decode = deployment
    await prefill._xavier_handoff.prepare(
        "prompt", config("prefill")["_pd_kv_transfer_params"]
    )
    with pytest.raises(ValueError, match="prompt mismatch"):
        await decode.async_generate("different", config("decode"))
    decode._non_stream_generate.assert_not_awaited()
    with pytest.raises(ValueError, match="decode replica"):
        await prefill.async_generate("prompt", config("decode"))


@pytest.mark.asyncio
async def test_stream_disconnect_aborts_and_releases_metadata(deployment):
    actor, prefill, decode = deployment
    await prefill._xavier_handoff.prepare(
        "prompt", config("prefill")["_pd_kv_transfer_params"]
    )

    async def chunks(*args, **kwargs):
        await actor.complete(123, 4096)
        yield dict(prompt_tokens=7, completion_tokens=1, finish_reason=None), "first"
        await asyncio.Event().wait()

    decode._stream_generate = chunks
    stream = await decode.async_generate(
        "prompt", config("decode", stream=True), request_id="r"
    )
    assert (await anext(stream))["choices"][0]["text"] == "first"
    await stream.aclose()
    decode.abort_request.assert_awaited_once_with("r")
    assert (await actor.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
async def test_directory_rejects_duplicates_mismatches_and_expiry(deployment):
    actor, _, _ = deployment
    await actor.prepare(123, "gpu-namespace", "prompt", "prefill")
    with pytest.raises(ValueError, match="duplicate"):
        await actor.prepare(123, "gpu-namespace", "prompt", "prefill")
    with pytest.raises(ValueError, match="prompt mismatch"):
        await actor.prepare(123, "gpu-namespace", "other", "decode")
    await actor.release(123)
    with pytest.raises(RuntimeError, match="expired"):
        await actor.source(123)
