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
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failed", [False, True])
async def test_release_failure_preserves_generation_result_or_original_error(
    deployment, stream, failed, caplog
):
    _, _, decode = deployment
    decode._xavier_handoff = SimpleNamespace(
        role="decode",
        accept=AsyncMock(return_value={"room": 1}),
        check_hit=AsyncMock(),
        release=AsyncMock(side_effect=RuntimeError("directory unavailable")),
    )
    if failed:
        decode._non_stream_generate.side_effect = RuntimeError(
            "original engine failure"
        )

    async def chunks(*args, **kwargs):
        if failed:
            raise RuntimeError("original engine failure")
        yield decode._non_stream_generate.return_value["meta_info"], "output"

    decode._stream_generate = chunks

    async def consume():
        result = await decode.async_generate(
            "prompt", config("decode", stream=stream), request_id="r"
        )
        return [chunk async for chunk in result] if stream else result

    if failed:
        with pytest.raises(RuntimeError, match="original engine failure"):
            await consume()
    else:
        assert await consume()
    assert "Failed to release SGLang handoff" in caplog.text
    decode._xavier_handoff.release.assert_awaited_once()
    assert not decode._active_request_ids


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
        None,
        600,
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
            for method in (
                "get_stats",
                "prepare",
                "check",
                "wait_complete",
                "renew_completed",
                "release",
            )
        }
    )


def fingerprint_handoff():
    tokenizer = SimpleNamespace(
        encode=Mock(side_effect=lambda text: list(text.encode()))
    )
    h = SGLangXavierHandoff(dict(role="decode"), tokenizer)
    h._namespace = "namespace"
    h._directory_actor = SimpleNamespace(prepare=AsyncMock())
    return h


async def prepare(h, prompt, room=1):
    return await h.prepare(prompt, dict(sglang_xavier=dict(mode="gpu", room=room)))


@pytest.mark.asyncio
@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("host", [False, True])
async def test_repeated_prompt_reuses_fingerprint_and_validates_every_room(
    heterogeneous,
    host,
):
    h = fingerprint_handoff()
    h.config["host_handoff"] = host
    if heterogeneous:
        h.config.update(heterogeneous=True, rank=1)
    for room in range(1, 4):
        await h.prepare(
            "prompt",
            dict(sglang_xavier=dict(mode="host" if host else "gpu", room=room)),
        )
    h.tokenizer.encode.assert_called_once_with("prompt")
    assert h._directory_actor.prepare.await_count == 3
    expected = hashlib.sha256(json.dumps(list(b"prompt")).encode()).hexdigest()
    arguments = (
        3,
        "namespace",
        expected,
        "decode",
        1 if heterogeneous else None,
        600,
    )
    if heterogeneous:
        arguments += (len(b"prompt"),)
    assert h._directory_actor.prepare.call_args.args == arguments
    assert list(h._prompt_hashes) == [hashlib.sha256(b"prompt").digest()]
    assert list(h._prompt_hashes.values()) == [(expected, len(b"prompt"))]


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


@pytest.mark.asyncio
async def test_completed_room_survives_long_decode_and_requires_both_releases(
    monkeypatch,
):
    directory = XavierPDDirectory()
    directory.configure("ns")
    clock = [0]
    monkeypatch.setattr(
        "xinference.model.llm.sglang.xavier.directory.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    directory.prepare(1, "ns", "prompt", "prefill", 0)
    directory.prepare(1, "ns", "prompt", "decode", 1)
    directory.complete(1, 1024)
    directory.prepare(2, "ns", "prompt", "prefill", 0)
    for now in range(300, 3601, 300):
        clock[0] = now
        assert directory.renew_completed(1)
    assert directory.check(1)
    assert not directory.check(2)
    directory.release(1, "prefill")
    assert directory.check(1)
    directory.release(1, "decode")
    assert directory.get_stats()["active_handoffs"] == 0


def test_completed_room_expires_after_lost_decode_release(monkeypatch):
    clock = [0]
    monkeypatch.setattr(
        "xinference.model.llm.sglang.xavier.directory.time",
        SimpleNamespace(monotonic=lambda: clock[0]),
    )
    directory = XavierPDDirectory()
    directory.configure("ns")
    directory.prepare(1, "ns", "prompt", "prefill", timeout=10)
    directory.prepare(1, "ns", "prompt", "decode", timeout=10)
    clock[0] = 9
    directory.complete(1, 1024)
    directory.release(1, "prefill")
    clock[0] = 11
    assert directory.check(1)  # Completion starts a fresh retention interval.
    clock[0] = 20
    assert directory.get_stats()["active_handoffs"] == 0
    assert not directory.renew_completed(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wait_source", "wait_complete"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_directory_wait_wakes_on_progress_or_cleanup(
    deployment, method, cancelled
):
    directory, _, _ = deployment
    await directory.prepare(1, "gpu-namespace", "prompt", "prefill")
    await directory.prepare(1, "gpu-namespace", "prompt", "decode")
    waiting = asyncio.create_task(getattr(directory, method)(1))
    await asyncio.sleep(0.02)
    assert not waiting.done()
    if cancelled:
        await directory.release(1)
        with pytest.raises(RuntimeError, match="cancelled"):
            await asyncio.wait_for(waiting, 1)
    else:
        if method == "wait_source":
            await directory.publish_source(1, {"rank": 0, "address": "gpu"})
        else:
            await directory.complete(1, 1024)
        assert await asyncio.wait_for(waiting, 1)
        await directory.release(1)


@pytest.mark.asyncio
async def test_directory_destroy_wakes_pending_waiters(deployment):
    directory, _, _ = deployment
    await directory.prepare(1, "gpu-namespace", "prompt", "prefill")
    waiting = asyncio.create_task(directory.wait_source(1))
    await asyncio.sleep(0.02)
    assert not waiting.done()
    await xo.destroy_actor(directory)
    with pytest.raises(RuntimeError, match="cancelled"):
        await asyncio.wait_for(waiting, 1)


@pytest.mark.asyncio
async def test_decode_renews_completed_lease_and_stops_on_release(
    deployment, monkeypatch
):
    from ..xavier.settings import TRANSFER_TIMEOUT_ENV

    monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, "0.2")
    directory, prefill, decode = deployment
    transfer = config("decode")["_pd_kv_transfer_params"]
    await prefill._xavier_handoff.prepare("prompt", transfer)
    handoff = await decode._xavier_handoff.accept("prompt", transfer)
    await directory.complete(123, 1024)
    await directory.release(123, "prefill")
    task = decode._xavier_handoff._lease_tasks[123]
    await asyncio.sleep(0.4)
    await decode._xavier_handoff.check_hit({}, handoff)
    await decode._xavier_handoff.release(handoff)
    assert task.done() and not decode._xavier_handoff._lease_tasks
    assert (await directory.get_stats())["active_handoffs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_release_finishes_when_renewal_rpc_swallows_cancellation(
    monkeypatch, failed
):
    from ..xavier.settings import TRANSFER_TIMEOUT_ENV

    monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, "0.03")
    handoff = fingerprint_handoff()
    entered = asyncio.Event()

    async def renew(room):
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            # Model Python 3.10 wait_for returning the completed RPC's result
            # or exception instead of propagating the caller's cancellation.
            if failed:
                raise RuntimeError("RPC completed while cancelling")
            return True

    handoff._directory_actor.renew_completed = AsyncMock(side_effect=renew)
    handoff._directory_actor.release = AsyncMock()
    original_call = handoff._call

    async def call(method, *args, **kwargs):
        if method == "renew_completed":
            return await handoff._directory_actor.renew_completed(*args)
        return await original_call(method, *args, **kwargs)

    monkeypatch.setattr(handoff, "_call", call)
    accepted = await handoff.accept(
        "prompt", config("decode")["_pd_kv_transfer_params"]
    )
    task = handoff._lease_tasks[123]
    releasing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        releasing = asyncio.create_task(handoff.release(accepted))
        done, _ = await asyncio.wait([releasing], timeout=1)
        assert releasing in done
        await releasing
        assert task.done() and not handoff._lease_tasks
        handoff._directory_actor.renew_completed.assert_awaited_once_with(123)
        handoff._directory_actor.release.assert_awaited_once_with(123, "decode")
    finally:
        handoff._directory_actor.renew_completed.side_effect = None
        handoff._directory_actor.renew_completed.return_value = False
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if releasing is not None:
            await asyncio.gather(releasing, return_exceptions=True)


@pytest.mark.asyncio
async def test_unconsumed_stream_reserves_no_request_or_directory_room(deployment):
    directory, _, decode = deployment
    stream = await decode.async_generate("prompt", config("decode", stream=True))
    assert not decode._active_request_ids
    assert (await directory.get_stats())["active_handoffs"] == 0
    await stream.aclose()


def test_transfer_timeout_configuration(monkeypatch):
    from ..xavier.settings import TRANSFER_TIMEOUT_ENV, transfer_timeout

    monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, "900")
    assert transfer_timeout() == 900
    directory = XavierPDDirectory()
    directory.configure("ns")
    monkeypatch.setattr("time.monotonic", lambda: 0)
    directory.prepare(1, "ns", "p", "prefill")
    monkeypatch.setattr("time.monotonic", lambda: 601)
    assert directory.source(1) is None
    monkeypatch.setattr("time.monotonic", lambda: 901)
    with pytest.raises(RuntimeError, match="expired"):
        directory.source(1)
    for value in ("0", "-1", "nan", "inf"):
        monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, value)
        with pytest.raises(ValueError):
            transfer_timeout()


@pytest.mark.asyncio
async def test_launch_timeout_reaches_directory_with_a_different_environment(
    monkeypatch,
):
    from ..xavier.settings import TRANSFER_TIMEOUT_ENV

    directory = XavierPDDirectory()
    directory.configure("ns")
    handoff = SGLangXavierHandoff(
        dict(role="prefill", rank=0),
        SimpleNamespace(encode=lambda text: [1]),
    )

    async def prepare(*args):
        with monkeypatch.context() as patch:
            patch.delenv(TRANSFER_TIMEOUT_ENV, raising=False)
            directory.prepare(*args)

    handoff._directory_actor = SimpleNamespace(
        get_stats=AsyncMock(return_value={"namespace": "ns"}),
        prepare=prepare,
    )
    monkeypatch.setenv(TRANSFER_TIMEOUT_ENV, "900")
    monkeypatch.setattr("time.monotonic", lambda: 0)
    await handoff.prepare("prompt", {"sglang_xavier": {"mode": "gpu", "room": 1}})
    monkeypatch.setattr("time.monotonic", lambda: 601)
    assert directory.source(1) is None
