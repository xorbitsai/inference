# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import xoscar as xo

from ..pd_model import PDModelActor, RoundRobinSchedulingPolicy
from ..replica_config import ReplicaConfig, validate_pd_replica_configs


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
    prefill.free_model_cache.assert_awaited_once_with("r")


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
    prefill.free_model_cache.assert_awaited_once_with("r")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["prefill", "decode", "handler", "cancel"])
async def test_failure_releases_cache(router, failure):
    actor, prefill, decode = router
    error = asyncio.CancelledError() if failure == "cancel" else RuntimeError("failed")
    target = prefill.chat if failure == "prefill" else decode.chat
    if failure == "handler":
        target = decode.set_unpin_handler
    target.side_effect = error
    with pytest.raises(type(error)):
        await actor._infer("chat", [], {"max_tokens": 32}, request_id="r")
    assert not actor._request_set
    prefill.free_model_cache.assert_awaited_once_with("r")


@pytest.mark.asyncio
async def test_abort_both_stages(router):
    actor, prefill, decode = router
    actor._request_set.add("r")
    assert await actor.abort_request("r") == "DONE"
    prefill.abort_request.assert_awaited_once()
    decode.abort_request.assert_awaited_once()
    assert not actor._request_set


class FakeStage(xo.StatelessActor):
    async def decrease_serve_count(self):
        pass

    async def set_unpin_handler(self, *args):
        pass

    async def free_model_cache(self, request_id):
        pass

    @xo.generator
    async def chat(self, messages, config, **kwargs):
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
        return b"ok"

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
    prefill.free_model_cache.assert_awaited_once_with("r")
