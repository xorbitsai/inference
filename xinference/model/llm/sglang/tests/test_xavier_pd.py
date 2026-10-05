# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
import hashlib
import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

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
@pytest.mark.parametrize("completed", [False, True])
async def test_stream_checks_gpu_completion_before_first_output(deployment, completed):
    actor, prefill, decode = deployment
    await prefill._xavier_handoff.prepare(
        "prompt", config("prefill")["_pd_kv_transfer_params"]
    )
    check = AsyncMock(wraps=decode._xavier_handoff.check_hit)
    decode._xavier_handoff.check_hit = check

    async def chunks(*args, **kwargs):
        if completed:
            await actor.complete(123, 4096)
        for i in range(3):
            yield dict(
                prompt_tokens=7,
                completion_tokens=i + 1,
                finish_reason={"type": "length"} if i == 2 else None,
            ), str(i)

    decode._stream_generate = chunks
    stream = await decode.async_generate(
        "prompt", config("decode", stream=True), request_id="r"
    )
    if completed:
        outputs = [chunk async for chunk in stream]
        assert "".join(chunk["choices"][0]["text"] for chunk in outputs) == "012"
        decode.abort_request.assert_not_awaited()
        await actor.release(123, "prefill")
    else:
        with pytest.raises(RuntimeError, match="GPU KV transfer"):
            await anext(stream)
        decode.abort_request.assert_awaited_once_with("r")
    check.assert_awaited_once()
    assert not decode._active_request_ids
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


@pytest.mark.asyncio
async def test_cached_directory_still_validates_each_request(deployment, monkeypatch):
    actor, prefill, decode = deployment
    directory = directory_spy(actor)
    lookup = AsyncMock(return_value=directory)
    monkeypatch.setattr(xo, "actor_ref", lookup)
    p, d = prefill._xavier_handoff, decode._xavier_handoff
    transfer = config("prefill")["_pd_kv_transfer_params"]
    for room in (123, 124):
        transfer["sglang_xavier"]["room"] = room
        await p.prepare("prompt", transfer)
        await d.accept("prompt", transfer)
        with pytest.raises(ValueError, match="duplicate"):
            await p.prepare("prompt", transfer)
        with pytest.raises(RuntimeError, match="GPU KV transfer"):
            await d.check_hit({}, transfer["sglang_xavier"])
        await actor.complete(room, 4096)
        await d.check_hit({}, transfer["sglang_xavier"])
        await p.publish(transfer["sglang_xavier"])
        await d.release(transfer["sglang_xavier"])
    assert lookup.await_count == directory.get_stats.await_count == 2
    assert directory.prepare.await_count == 6
    stats = await actor.get_stats()
    assert stats["completed_requests"] == 2 and stats["active_handoffs"] == 0

    # Both replicas now have caches. A different paired prompt must still fail.
    await p.prepare("next prompt", transfer)
    with pytest.raises(ValueError, match="prompt mismatch"):
        await d.accept("wrong prompt", transfer)
    with pytest.raises(ValueError, match="namespaces differ"):
        await actor.configure("different-namespace")
    await p.release(transfer["sglang_xavier"], failed=True)
    assert (await actor.get_stats())["active_handoffs"] == 0
    assert lookup.await_count == directory.get_stats.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["lookup", "stats", "unregistered", "cancel"])
async def test_directory_initialization_failure_can_retry(
    deployment, monkeypatch, failure
):
    actor, prefill, _ = deployment
    error = asyncio.CancelledError() if failure == "cancel" else RuntimeError("lookup")
    directory = directory_spy(actor)
    lookup = AsyncMock(return_value=directory)
    monkeypatch.setattr(xo, "actor_ref", lookup)
    if failure == "lookup":
        lookup.side_effect = [error, directory]
    elif failure == "unregistered":
        directory.get_stats.side_effect = [
            {"namespace": None},
            {"namespace": "gpu-namespace"},
        ]
    else:
        directory.get_stats.side_effect = [error, {"namespace": "gpu-namespace"}]
    expected = (
        asyncio.CancelledError
        if failure == "cancel"
        else ValueError if failure == "unregistered" else RuntimeError
    )
    transfer = config("prefill")["_pd_kv_transfer_params"]
    with pytest.raises(expected):
        await prefill._xavier_handoff.prepare("prompt", transfer)
    assert (await actor.get_stats())["active_handoffs"] == 0
    await prefill._xavier_handoff.prepare("prompt", transfer)
    await prefill._xavier_handoff.release(transfer["sglang_xavier"], failed=True)
    assert (await actor.get_stats())["active_handoffs"] == 0


def directory_spy(actor):
    async def call(method, *args):
        return await getattr(actor, method)(*args)

    return SimpleNamespace(
        **{
            method: AsyncMock(side_effect=partial(call, method))
            for method in ("get_stats", "prepare", "check", "release")
        }
    )


def fingerprint_handoff():
    tokenizer = SimpleNamespace(
        encode=Mock(side_effect=lambda text: list(text.encode()))
    )
    h = SGLangXavierHandoff(dict(role="decode"), 64, tokenizer)
    h._namespace = "namespace"
    h._directory_actor = SimpleNamespace(prepare=AsyncMock())
    return h


async def prepare(h, prompt, room=1):
    return await h.prepare(prompt, dict(sglang_xavier=dict(mode="gpu", room=room)))


@pytest.mark.asyncio
async def test_repeated_prompt_reuses_fingerprint_and_validates_every_room():
    h = fingerprint_handoff()
    for room in range(1, 4):
        await prepare(h, "prompt", room)
    h.tokenizer.encode.assert_called_once_with("prompt")
    assert h._directory_actor.prepare.await_count == 3
    expected = hashlib.sha256(json.dumps(list(b"prompt")).encode()).hexdigest()
    assert h._directory_actor.prepare.call_args.args == (
        3,
        "namespace",
        expected,
        "decode",
    )
    assert list(h._prompt_hashes) == [hashlib.sha256(b"prompt").digest()]
    assert list(h._prompt_hashes.values()) == [expected]


@pytest.mark.asyncio
async def test_fingerprints_are_bounded_and_recently_used_entry_survives():
    h = fingerprint_handoff()
    for n in range(256):
        await prepare(h, str(n))
    await prepare(h, "0")
    await prepare(h, "new")
    assert len(h._prompt_hashes) == 256
    assert hashlib.sha256(b"0").digest() in h._prompt_hashes
    assert hashlib.sha256(b"1").digest() not in h._prompt_hashes
    await prepare(h, "1")
    assert h.tokenizer.encode.call_count == 258


@pytest.mark.asyncio
async def test_changed_prompt_and_new_deployment_recompute_tokens():
    first, second = fingerprint_handoff(), fingerprint_handoff()
    await prepare(first, "one")
    await prepare(first, "two")
    await prepare(second, "one")
    assert (
        first.tokenizer.encode.call_count == 2
        and second.tokenizer.encode.call_count == 1
    )
    assert len(first._prompt_hashes) == 2 and len(second._prompt_hashes) == 1


@pytest.mark.asyncio
async def test_cached_prompt_does_not_bypass_pairing_failure():
    h = fingerprint_handoff()
    await prepare(h, "one")
    h._directory_actor.prepare.side_effect = ValueError(
        "duplicate room or mismatched prompt"
    )
    with pytest.raises(ValueError, match="duplicate room"):
        await prepare(h, "one")
    assert (
        h.tokenizer.encode.call_count == 1
        and h._directory_actor.prepare.await_count == 2
    )


@pytest.mark.asyncio
async def test_encoding_failure_is_never_cached():
    h = fingerprint_handoff()
    h.tokenizer.encode.side_effect = ValueError("tokenizer failure")
    for _ in range(2):
        with pytest.raises(ValueError, match="tokenizer failure"):
            await prepare(h, "one")
    assert not h._prompt_hashes
    h._directory_actor.prepare.assert_not_awaited()
