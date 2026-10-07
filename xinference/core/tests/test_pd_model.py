# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ..pd_model import PDModelActor, RoundRobinSchedulingPolicy
from ..replica_config import ReplicaConfig, validate_pd_replica_configs
from ..rpc_context import rpc_context


def test_round_robin_updates():
    policy = RoundRobinSchedulingPolicy([1, 2])
    assert [policy.schedule() for _ in range(5)] == [1, 2, 1, 2, 1]
    policy.update_replicas([3])
    assert policy.schedule() == 3
    policy.update_replicas([])
    with pytest.raises(RuntimeError):
        policy.schedule()


@pytest.mark.parametrize(
    "roles,engine,valid",
    [
        (["prefill", "decode"], "vLLM", True),
        (["prefill", "prefill", "decode"], "vLLM", True),
        (["prefill", "decode"], "SGLang", True),
        (["prefill", "decode"], "MLX", True),
        (["hybrid"], "transformers", False),
        (["prefill"], "vLLM", None),
        (["decode"], "vLLM", None),
        (["prefill", "decode", "hybrid"], "vLLM", None),
        (["prefill", "decode"], "transformers", None),
    ],
)
def test_validate_topology(roles, engine, valid):
    configs = [ReplicaConfig(role=role) for role in roles]
    if valid is None:
        with pytest.raises(ValueError):
            validate_pd_replica_configs(configs, engine, "LLM")
    else:
        assert validate_pd_replica_configs(configs, engine, "LLM") is valid


@pytest.fixture
async def router():
    actor = PDModelActor("pd")
    actor.address = "localhost:1234"
    prefill, decode = MagicMock(), MagicMock()
    for model in (prefill, decode):
        model.chat = AsyncMock(return_value=b'{"choices":[]}')
        model.generate = AsyncMock(return_value=b'{"choices":[]}')
        model.free_model_cache = AsyncMock()
        model.set_unpin_handler = AsyncMock()
        model.decrease_serve_count = AsyncMock()
        model.abort_request = AsyncMock(return_value="DONE")
    await actor.add_prefill_actor("p", prefill)
    await actor.add_decode_actor("d", decode)
    prefill.chat.return_value = prefill.generate.return_value = {
        "_pd_kv_transfer_params": {"do_remote_prefill": True}
    }
    return actor, prefill, decode


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "generate"])
async def test_infer_preserves_decode_config(router, method):
    actor, prefill, decode = router
    config = {"max_tokens": 32, "stream": False}
    raw = {"max_tokens": 32, "stream": False}
    result = await actor._infer(
        method, "prompt", config, raw_params=raw, request_id="r"
    )
    assert result == b'{"choices":[]}'
    assert config["max_tokens"] == raw["max_tokens"] == 32
    p_call = getattr(prefill, method).call_args
    d_call = getattr(decode, method).call_args
    assert p_call.args[1]["max_tokens"] == 1
    assert d_call.args[1]["max_tokens"] == 32
    assert p_call.kwargs["request_id"] == d_call.kwargs["request_id"] == "r"
    assert not actor._request_set
    prefill.free_model_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_cross_engine_allows_plain_text_response_format(router):
    actor, prefill, decode = router
    actor._model_engine = "heterogeneous"
    await actor._infer(
        "generate", "prompt", {"response_format": {"type": "text"}}, request_id="r"
    )
    assert prefill.generate.await_count == decode.generate.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,backend",
    [("sglang", "xavier"), ("sglang", "nixl"), ("heterogeneous", "xavier")],
)
async def test_sglang_starts_decode_while_prefill_holds_source_slots(
    router, engine, backend
):
    actor, prefill, decode = router
    actor._model_engine = engine
    actor._transport_backend = backend
    if backend == "nixl":
        actor._sglang_bootstrap["p"] = dict(host="producer", port=12345)
    decode_started = asyncio.Event()

    async def p(*args, **kwargs):
        await asyncio.wait_for(decode_started.wait(), timeout=1)
        return {}

    async def d(*args, **kwargs):
        decode_started.set()
        await asyncio.sleep(0)
        return {"choices": []}

    prefill.generate.side_effect = p
    decode.generate.side_effect = d
    await actor._infer("generate", "prompt", {"max_tokens": 32}, request_id="r")
    p_config = prefill.generate.call_args.args[1]
    d_config = decode.generate.call_args.args[1]
    assert p_config["max_tokens"] == 1 and d_config["max_tokens"] == 32
    key = "sglang_nixl" if backend == "nixl" else "sglang_xavier"
    handoff = p_config["_pd_kv_transfer_params"][key]
    assert handoff == d_config["_pd_kv_transfer_params"][key]
    if backend == "nixl":
        assert handoff["host"] == "producer" and handoff["port"] == 12345
    if engine == "heterogeneous":
        assert handoff["heterogeneous"] is True
    assert not actor._request_set and not actor._direct_transfers


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["xavier", "nixl"])
async def test_sglang_stream_disconnect_aborts_both_roles(router, monkeypatch, backend):
    actor, prefill, decode = router
    actor._model_engine = "sglang"
    actor._transport_backend = backend
    if backend == "nixl":
        actor._sglang_bootstrap["p"] = dict(host="producer", port=12345)
    directory = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=directory))

    async def chunks():
        yield b"first"
        await asyncio.Event().wait()

    decode.generate.return_value = chunks()
    stream = await actor._infer("generate", "prompt", {"stream": True}, request_id="r")
    assert await anext(stream) == b"first"
    await stream.aclose()
    prefill.abort_request.assert_awaited_once()
    decode.abort_request.assert_awaited_once()
    decode.decrease_serve_count.assert_awaited_once()
    if backend == "nixl":
        directory.release.assert_not_awaited()
    else:
        directory.release.assert_awaited_once()
    assert not actor._request_set and not actor._direct_transfers


@pytest.mark.asyncio
async def test_native_sglang_bootstrap_refreshed_on_replica_replacement():
    actor = PDModelActor("pd", transport_backend="nixl", model_engine="sglang")
    prefill = MagicMock()
    prefill.get_sglang_pd_bootstrap = AsyncMock(
        return_value=dict(host="producer", port=12345)
    )
    await actor.add_prefill_actor("p", prefill)
    assert actor._sglang_bootstrap["p"]["port"] == 12345
    await actor.remove_prefill_actor("p")
    assert not actor._sglang_bootstrap
    prefill.get_sglang_pd_bootstrap.return_value["port"] = 23456
    await actor.add_prefill_actor("p", prefill)
    assert actor._sglang_bootstrap["p"]["port"] == 23456


@pytest.mark.asyncio
@pytest.mark.parametrize("source_published", [False, True])
@pytest.mark.parametrize(
    "mode,host_source", [("gpu", False), ("host", False), ("host", True)]
)
async def test_cross_engine_cancel_drains_source_before_directory_release(
    router, monkeypatch, source_published, mode, host_source
):
    actor, prefill, decode = router
    actor._model_engine = "heterogeneous"
    actor._handoff_mode = mode
    calls = []
    directory, sender = AsyncMock(), AsyncMock()
    directory.source.return_value = (
        dict(
            address="gpu-host",
            rank=1,
            **({"uid": "host-source"} if host_source else {}),
        )
        if source_published
        else None
    )

    async def drain(room):
        calls.append(("drain", room))

    async def release(room):
        calls.append(("release", room))

    sender.abort.side_effect = drain
    directory.release.side_effect = release
    actor_ref = AsyncMock(side_effect=[directory, sender])
    monkeypatch.setattr(xo, "actor_ref", actor_ref)
    started = asyncio.Event()

    async def pending(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    prefill.generate.side_effect = pending
    decode.generate.side_effect = pending
    task = asyncio.create_task(actor._infer("generate", "prompt", {}, request_id="r"))
    await started.wait()
    handoff = actor._direct_transfers["r"]
    assert handoff["mode"] == mode
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    expected = [("drain", handoff["room"])] if source_published else []
    assert calls == expected + [("release", handoff["room"])]
    if source_published:
        assert actor_ref.await_args_list[-1].kwargs["uid"] == (
            "host-source" if host_source else "xavier-cross-engine-transfer-1"
        )
    prefill.abort_request.assert_awaited_once()
    decode.abort_request.assert_awaited_once()
    assert not actor._request_set and not actor._direct_transfers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"logprobs": 0},
        {"prompt_logprobs": 0},
        {"guided_choice": ["yes", "no"]},
        {"json_schema": {"type": "object"}},
        {"response_format": {"type": "json_object"}},
        {"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}},
        {"return_logprob": True},
    ],
)
async def test_cross_engine_rejects_unsupported_sampling_before_scheduling(
    router, config
):
    actor, prefill, decode = router
    actor._model_engine = "heterogeneous"
    with pytest.raises(ValueError, match="structured sampling or logprobs"):
        await actor._infer("generate", "prompt", config, request_id="r")
    prefill.generate.assert_not_awaited()
    decode.generate.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key", ["min_p", "presence_penalty", "frequency_penalty", "seed"]
)
@pytest.mark.parametrize("raw", [False, True])
async def test_mlx_prefill_rejects_first_token_sampling_before_scheduling(
    router, key, raw
):
    actor, prefill, decode = router
    actor._model_engine = "heterogeneous"
    actor._mlx_prefill = True
    parameters = {key: 0}
    with pytest.raises(ValueError, match="MLX cross-engine prefill"):
        await actor._infer(
            "generate",
            "prompt",
            {} if raw else parameters,
            **({"raw_params": parameters} if raw else {}),
            request_id="r",
        )
    prefill.generate.assert_not_awaited()
    decode.generate.assert_not_awaited()
    assert not actor._request_set and not actor._direct_transfers


@pytest.mark.asyncio
@pytest.mark.parametrize("mlx_prefill", [False, True])
async def test_first_token_guard_preserves_other_routes_and_unset_parameters(
    router, mlx_prefill
):
    actor, _, _ = router
    actor._model_engine = "heterogeneous"
    actor._mlx_prefill = mlx_prefill
    actor._infer_sglang = AsyncMock(return_value="result")
    config = dict.fromkeys(
        ("min_p", "presence_penalty", "frequency_penalty", "seed"),
        None if mlx_prefill else 0,
    )
    assert await actor._infer("generate", "prompt", config, request_id="r") == "result"
    actor._infer_sglang.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_cleanup_on_disconnect(router):
    actor, prefill, decode = router
    closed = []

    async def chunks():
        try:
            yield b"first"
            yield b"second"
        finally:
            closed.append(True)

    decode.chat.return_value = chunks()
    stream = await actor._infer(
        "chat", [], {"max_tokens": 32, "stream": True}, request_id="r"
    )
    assert await anext(stream) == b"first"
    await stream.aclose()
    assert closed and not actor._request_set
    prefill.free_model_cache.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prefill", "decode", "cancel"])
async def test_failure_releases_cache(router, failure):
    actor, prefill, decode = router
    error = asyncio.CancelledError() if failure == "cancel" else RuntimeError("failed")
    target = prefill.chat if failure == "prefill" else decode.chat
    target.side_effect = error
    with pytest.raises(type(error)):
        await actor._infer("chat", [], {"max_tokens": 32}, request_id="r")
    assert not actor._request_set
    prefill.free_model_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_abort_both_stages(router):
    actor, prefill, decode = router
    actor._request_set.add("r")
    assert await actor.abort_request("r") == "DONE"
    prefill.abort_request.assert_awaited_once()
    decode.abort_request.assert_awaited_once()
    assert not actor._request_set


class FakeStage(xo.StatelessActor):
    @rpc_context
    async def decrease_serve_count(self):
        pass

    @rpc_context
    async def set_unpin_handler(self, *args):
        pass

    @rpc_context
    async def free_model_cache(self, request_id):
        pass

    @xo.generator
    @rpc_context
    async def chat(self, messages, config, **kwargs):
        if config.get("_pd_kv_transfer_params", {}).get("do_remote_decode"):
            return {"_pd_kv_transfer_params": {"do_remote_prefill": True}}
        if not config.get("stream"):
            return b"non-stream"

        async def chunks():
            yield b"one"
            yield b"two"

        return chunks()


@pytest.mark.asyncio
async def test_nested_actor_streaming():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        p = await xo.create_actor(FakeStage, address=pool.external_address, uid="p")
        d = await xo.create_actor(FakeStage, address=pool.external_address, uid="d")
        router = await xo.create_actor(
            PDModelActor, "pd", address=pool.external_address
        )
        await router.add_prefill_actor("p", p)
        await router.add_decode_actor("d", d)
        assert await router.chat([], {"stream": False}) == b"non-stream"
        stream = await router.chat([], {"stream": True})
        assert [chunk async for chunk in stream] == [b"one", b"two"]


@pytest.mark.asyncio
async def test_aborted_prefill_does_not_start_decode(router):
    actor, prefill, decode = router
    started, finish = asyncio.Event(), asyncio.Event()

    async def wait(*args, **kwargs):
        started.set()
        await finish.wait()
        return {"_pd_kv_transfer_params": {"do_remote_prefill": True}}

    prefill.chat.side_effect = wait
    task = asyncio.create_task(actor._infer("chat", [], {}, request_id="r"))
    await started.wait()
    await actor.abort_request("r")
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    decode.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovered_replica_replaces_stale_reference(router):
    actor, prefill, decode = router
    replacement = MagicMock()
    await actor.add_prefill_actor("p", replacement)
    assert actor.get_prefill_actor("p") is replacement
    assert actor._prefill_policy.schedule() is replacement
    assert (await actor.get_pd_info())["prefill_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [[], "invalid", 1])
async def test_invalid_config_rejected_before_inference(router, config):
    actor, prefill, decode = router
    with pytest.raises(TypeError, match="Generation config must be a dict or None"):
        await actor._infer("chat", "prompt", config)
    prefill.chat.assert_not_awaited()
    decode.chat.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", [False, True])
async def test_free_model_cache_engine_layout(nested):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ...model.llm.vllm.core import VLLMModel
    from ..model import ModelActor

    scheduler = Mock()
    engine = SimpleNamespace(scheduler=[scheduler])
    model = object.__new__(VLLMModel)
    model._engine = SimpleNamespace(engine=engine) if nested else engine
    actor = SimpleNamespace(_model=model)
    await ModelActor.free_model_cache(actor, "request")
    scheduler.free_seq_cache.assert_called_once_with("request")


@pytest.mark.asyncio
async def test_abort_failure_still_cleans_request(router):
    actor, prefill, decode = router
    actor._request_set.add("r")
    decode.abort_request.side_effect = RuntimeError("replica offline")
    with pytest.raises(RuntimeError, match="replica offline"):
        await actor.abort_request("r")
    assert not actor._request_set
    prefill.free_model_cache.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat", "generate"])
@pytest.mark.parametrize("backend", ["nixl", "xavier"])
async def test_nixl_handoff_preserves_metadata_and_decode_settings(
    router, backend, method
):
    import json

    actor, prefill, decode = router
    actor._transport_backend = backend
    actor._direct_handoff = backend == "xavier"
    transfer = {
        "do_remote_prefill": True,
        "remote_engine_id": "engine-p",
        "remote_block_ids": [[1, 2]],
        "remote_request_id": "internal-request",
        "remote_host": "10.0.0.1",
        "remote_port": 5000,
    }
    getattr(prefill, method).return_value = json.dumps(
        {"_pd_kv_transfer_params": transfer}
    ).encode()
    config = {"max_tokens": 64, "n": 1, "stream": False, "temperature": 0.7}
    await actor._infer(method, "input", config, request_id="r")
    p_config = getattr(prefill, method).call_args.args[1]
    assert p_config["n"] == p_config["max_tokens"] == 1
    assert p_config["_pd_kv_transfer_params"]["do_remote_decode"]
    d_config = getattr(decode, method).call_args.args[1]
    assert d_config == {**config, "_pd_kv_transfer_params": transfer}
    assert "_pd_kv_transfer_params" not in config
    decode.set_unpin_handler.assert_not_awaited()
    prefill.free_model_cache.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["nixl", "xavier"])
async def test_nixl_parallel_sampling_rejected_before_prefill(router, backend):
    actor, prefill, decode = router
    actor._transport_backend = backend
    actor._direct_handoff = backend == "xavier"
    with pytest.raises(ValueError, match="n=1"):
        await actor._infer("chat", [], {"n": 2})
    prefill.chat.assert_not_awaited()
    decode.chat.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, None, b"null", b"[]", b"{}", b"42"])
@pytest.mark.parametrize("backend", ["nixl", "xavier"])
async def test_nixl_missing_metadata_does_not_silently_recompute(
    router, backend, payload
):
    actor, prefill, decode = router
    actor._transport_backend = backend
    actor._direct_handoff = backend == "xavier"
    prefill.chat.return_value = payload
    with pytest.raises(RuntimeError, match="KV transfer metadata"):
        await actor._infer("chat", [], {})
    decode.chat.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
async def test_nixl_multiple_prefillers_and_decoders():
    actor = PDModelActor("pd", transport_backend="nixl")
    actor.address = "localhost:1234"
    prefills, decodes = [], []
    for i in range(2):
        model = MagicMock()
        model.chat = AsyncMock(
            return_value={
                "_pd_kv_transfer_params": {
                    "do_remote_prefill": True,
                    "remote_engine_id": f"p{i}",
                }
            }
        )
        prefills.append(model)
        await actor.add_prefill_actor(f"p{i}", model)
    for i in range(3):
        model = MagicMock()
        model.chat = AsyncMock(return_value=b'{"choices":[]}')
        decodes.append(model)
        await actor.add_decode_actor(f"d{i}", model)
    for i in range(6):
        await actor._infer("chat", [], {}, request_id=str(i))
    assert [p.chat.await_count for p in prefills] == [3, 3]
    for d in decodes:
        sources = {
            call.args[1]["_pd_kv_transfer_params"]["remote_engine_id"]
            for call in d.chat.call_args_list
        }
        assert sources == {"p0", "p1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["nixl", "xavier"])
async def test_nixl_stream_close_releases_decode_slot(router, backend):
    actor, prefill, decode = router
    actor._transport_backend = backend
    actor._direct_handoff = backend == "xavier"
    prefill.chat.return_value = {"_pd_kv_transfer_params": {"do_remote_prefill": True}}
    closed = []

    async def chunks():
        try:
            yield b"first"
            yield b"second"
        finally:
            closed.append(True)

    decode.chat.return_value = chunks()
    stream = await actor._infer(
        "chat", [], generate_config={"stream": True}, request_id="r"
    )
    assert await anext(stream) == b"first"
    await stream.aclose()
    assert closed == [True]
    assert not actor._request_set
    decode.decrease_serve_count.assert_awaited_once()
    prefill.free_model_cache.assert_not_awaited()


@pytest.mark.asyncio
async def test_duplicate_generation_config_rejected_before_dispatch(router):
    actor, prefill, decode = router
    with pytest.raises(TypeError, match="Generation config supplied twice"):
        await actor._infer("chat", [], {}, generate_config={})
    prefill.chat.assert_not_awaited()
    decode.chat.assert_not_awaited()
    assert not actor._request_set


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["decode", "cancel", "abort_before_decode"])
@pytest.mark.parametrize("engine", ["vllm", "sglang", "mlx"])
async def test_direct_router_releases_unclaimed_handoff(
    router, monkeypatch, failure, engine
):
    actor, prefill, decode = router
    actor._direct_handoff = True
    transfer = {
        "do_remote_prefill": True,
        "xavier_direct": {"ticket": "t", "rank": 0, "address": "127.0.0.1:1234"},
    }
    if engine in ("sglang", "mlx"):
        transfer = {
            "do_remote_prefill": True,
            f"{engine}_xavier": {
                "engine": engine,
                "ticket": "t",
                "uid": "cache",
                "address": "127.0.0.1:1234",
            },
        }

    async def finish_prefill(*args, **kwargs):
        if failure == "abort_before_decode":
            actor._request_set.discard("r")
        return {"_pd_kv_transfer_params": transfer}

    prefill.chat.side_effect = finish_prefill
    decode.chat.side_effect = (
        RuntimeError("decode failed")
        if failure == "decode"
        else asyncio.CancelledError()
    )
    peer = MagicMock(abandon_direct_gpu_v1=AsyncMock(), release_handoff=AsyncMock())
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=peer))
    with pytest.raises(RuntimeError if failure == "decode" else asyncio.CancelledError):
        await actor._infer("chat", [], {}, request_id="r")
    if engine in ("sglang", "mlx"):
        peer.release_handoff.assert_awaited_once_with("t")
        peer.abandon_direct_gpu_v1.assert_not_awaited()
    else:
        peer.abandon_direct_gpu_v1.assert_awaited_once_with("t")
    assert not actor._direct_transfers and not actor._request_set
    if failure == "abort_before_decode":
        decode.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_completed_direct_request_does_not_send_abandon_rpc(router, monkeypatch):
    actor, prefill, decode = router
    prefill.chat.return_value = {
        "_pd_kv_transfer_params": {
            "do_remote_prefill": True,
            "xavier_direct": {"ticket": "t", "rank": 0, "address": "127.0.0.1:1234"},
        }
    }
    lookup = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", lookup)
    await actor._infer("chat", [], {}, request_id="r")
    lookup.assert_not_awaited()
    assert not actor._direct_transfers


@pytest.mark.asyncio
@pytest.mark.parametrize("handoff", [{"ticket": ""}, {"ticket": "", "address": "gpu"}])
async def test_empty_vllm_handoff_needs_no_abandon_rpc(router, monkeypatch, handoff):
    actor, _, _ = router
    actor._request_set.add("r")
    actor._direct_transfers["r"] = handoff
    abandon = AsyncMock()
    monkeypatch.setattr(actor, "_abandon_vllm_handoff", abandon)
    await actor.free_prefill_model_cache("r")
    abandon.assert_not_awaited()
    assert not actor._request_set and not actor._direct_transfers


@pytest.mark.asyncio
async def test_completed_direct_stream_clears_handoff_without_abandon(
    router, monkeypatch
):
    actor, prefill, decode = router
    prefill.chat.return_value = {
        "_pd_kv_transfer_params": {
            "do_remote_prefill": True,
            "xavier_direct": {"ticket": "t", "rank": 0, "address": "127.0.0.1:1234"},
        }
    }

    async def chunks():
        yield b"one"
        yield b"two"

    decode.chat.return_value = chunks()
    lookup = AsyncMock()
    monkeypatch.setattr(xo, "actor_ref", lookup)
    stream = await actor._infer("chat", [], {"stream": True}, request_id="r")
    assert [chunk async for chunk in stream] == [b"one", b"two"]
    assert not actor._direct_transfers and not actor._request_set
    lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_empty_decode_stream_waits_for_prefill_and_propagates_failure(
    router, failure
):
    actor, prefill, decode = router
    actor._model_engine = "sglang"
    actor._transport_backend = "nixl"
    actor._sglang_bootstrap["p"] = dict(host="producer", port=12345)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def produce(*args, **kwargs):
        entered.set()
        await finish.wait()
        if failure:
            raise RuntimeError("late prefill failure")
        return {}

    async def empty():
        if False:
            yield b"unused"

    prefill.generate.side_effect = produce
    decode.generate.return_value = empty()
    stream = await actor._infer("generate", "prompt", {"stream": True}, request_id="r")
    task = asyncio.create_task(anext(stream))
    await entered.wait()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(RuntimeError if failure else StopAsyncIteration):
        await task
    decode.decrease_serve_count.assert_awaited_once()
    assert not actor._request_set and not actor._direct_transfers
    if failure:
        prefill.abort_request.assert_awaited_once()
        decode.abort_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["xavier", "nixl"])
@pytest.mark.parametrize("finish_together", [False, True])
async def test_prefill_failure_closes_already_returned_decode_stream(
    router, monkeypatch, backend, finish_together
):
    actor, prefill, decode = router
    actor._model_engine = "sglang"
    actor._transport_backend = backend
    actor._sglang_bootstrap["p"] = dict(host="producer", port=12345)
    directory = SimpleNamespace(release=AsyncMock())
    monkeypatch.setattr(xo, "actor_ref", AsyncMock(return_value=directory))
    decoded, release_prefill = asyncio.Event(), asyncio.Event()
    stream = MagicMock()
    stream.aclose = AsyncMock()
    serve_count = 0

    async def decrease(**kwargs):
        nonlocal serve_count
        serve_count -= 1

    async def produce(*args, **kwargs):
        await decoded.wait()
        if not finish_together:
            await release_prefill.wait()
        raise RuntimeError("prefill failed after decode stream return")

    async def decode_stream(*args, **kwargs):
        nonlocal serve_count
        serve_count += 1
        decoded.set()
        return stream

    prefill.generate.side_effect = produce
    decode.generate.side_effect = decode_stream
    decode.decrease_serve_count.side_effect = decrease
    task = asyncio.create_task(
        actor._infer("generate", "prompt", {"stream": True}, request_id="r")
    )
    await decoded.wait()
    if not finish_together:
        # Let the router acquire the returned stream and then fail prefill on
        # the first iteration, covering both ownership handoff windows.
        returned = await task
        task = asyncio.create_task(anext(returned))
        release_prefill.set()
    with pytest.raises(RuntimeError, match="prefill failed"):
        await task
    assert serve_count == 0
    stream.aclose.assert_awaited_once()
    decode.decrease_serve_count.assert_awaited_once()
    prefill.abort_request.assert_awaited_once()
    decode.abort_request.assert_awaited_once()
    assert not actor._request_set and not actor._direct_transfers
