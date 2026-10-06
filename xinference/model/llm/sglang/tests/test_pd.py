# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from unittest.mock import AsyncMock

import pytest

from ..core import SGLANGModel
from ..pd import SGLangNixlHandoff, configure_nixl


def test_native_sglang_configuration_uses_unique_bootstrap_ports():
    ports, credentials = [], []
    for role in ("prefill", "decode"):
        config = {"dtype": "float16", "tp_size": 1}
        replica = dict(role=role, host="127.0.0.1")
        configure_nixl(config, replica, "0.5.21", 1)
        assert config["disaggregation_mode"] == role
        assert config["disaggregation_transfer_backend"] == "nixl"
        assert config["disaggregation_bootstrap_port"] == replica["port"]
        assert config["dtype"] == "float16"
        ports.append(replica["port"])
        credentials.append(config["api_key"])
    assert ports[0] != ports[1]
    assert credentials[0] != credentials[1] and all(credentials)


@pytest.mark.parametrize(
    "config,version,workers",
    [
        ({}, "0.5.20", 1),
        ({}, "0.5.21", 2),
        ({"tp_size": 2}, "0.5.21", 1),
        ({"dp_size": 2}, "0.5.21", 1),
        ({"disaggregation_mode": "prefill"}, "0.5.21", 1),
        ({"enable_lora": True}, "0.5.21", 1),
        ({"speculative_algorithm": "EAGLE"}, "0.5.21", 1),
        ({"enable_hierarchical_cache": True}, "0.5.21", 1),
        ({"tokenizer_worker_num": 2}, "0.5.21", 1),
    ],
)
def test_native_sglang_rejects_unmanaged_configuration(config, version, workers):
    with pytest.raises(ValueError):
        configure_nixl(config, dict(role="prefill", host="127.0.0.1"), version, workers)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["prefill", "decode"])
async def test_native_sglang_wrapper_forwards_actual_bootstrap(role):
    replica = dict(role=role, host="producer", port=12345)
    handoff = dict(mode="nixl", host="producer", port=12345, room=123)
    model = object.__new__(SGLANGModel)
    model.model_uid = role
    model._active_request_ids = set()
    model._nixl_handoff = SGLangNixlHandoff(replica)
    model._non_stream_generate = AsyncMock(
        return_value=dict(
            text="output",
            meta_info=dict(
                prompt_tokens=7,
                completion_tokens=1,
                finish_reason={"type": "length"},
            ),
        )
    )
    config = dict(
        max_tokens=16,
        _pd_kv_transfer_params=dict(
            do_remote_decode=role == "prefill",
            do_remote_prefill=role == "decode",
            sglang_nixl=handoff,
        ),
    )
    result = await model.async_generate("prompt", config, request_id="r")
    sent = model._non_stream_generate.call_args.kwargs
    assert sent["bootstrap_host"] == "producer"
    assert sent["bootstrap_port"] == 12345 and sent["bootstrap_room"] == 123
    assert not model._active_request_ids
    if role == "prefill":
        assert model.get_pd_bootstrap() == dict(host="producer", port=12345)
        assert result["_pd_kv_transfer_params"]["sglang_nixl"] == handoff


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"room": 0}, {"port": 0}, {"host": ""}, {"mode": "gpu"}, {"room": True}],
)
async def test_native_sglang_missing_bootstrap_cannot_fall_back_to_local_prefill(
    change,
):
    replica = dict(role="decode", host="decoder", port=23456)
    handoff = dict(mode="nixl", host="producer", port=12345, room=123)
    handoff.update(change)
    with pytest.raises(ValueError, match="bootstrap metadata"):
        await SGLangNixlHandoff(replica).accept("prompt", dict(sglang_nixl=handoff))


@pytest.mark.asyncio
async def test_native_http_requests_forward_auth_bootstrap_rid_and_abort():
    import json
    from types import SimpleNamespace

    import aiohttp
    from aiohttp import web

    model = object.__new__(SGLANGModel)
    model._model_config = {}
    configure_nixl(
        model._model_config, dict(role="decode", host="127.0.0.1"), "0.5.21", 1
    )
    credential = model._model_config["api_key"]
    model._active_request_ids = {"r"}
    received = []
    meta = dict(prompt_tokens=2, completion_tokens=1, finish_reason={"type": "length"})

    async def handle(request):
        if request.headers.get("Authorization") != f"Bearer {credential}":
            return web.Response(status=401)
        body = await request.json()
        received.append((request.path, body))
        if request.path == "/abort_request":
            return web.json_response({})
        result = dict(text="ok", meta_info=meta)
        if body.get("stream"):
            return web.Response(
                text="data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n",
                content_type="text/event-stream",
            )
        return web.json_response(result)

    app = web.Application()
    app.router.add_post("/generate", handle)
    app.router.add_post("/abort_request", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    model._engine = SimpleNamespace(url=url, generate_url=url + "/generate")
    params = dict(
        bootstrap_host="producer",
        bootstrap_port=12345,
        bootstrap_room=123,
        max_new_tokens=1,
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url + "/generate", json={}) as response:
                assert response.status == 401
        assert (
            await model._non_stream_generate("prompt", None, request_id="r", **params)
        )["text"] == "ok"
        chunks = [
            chunk
            async for chunk in model._stream_generate(
                "prompt", None, request_id="r", **params
            )
        ]
        assert chunks[0][1] == "ok"
        assert await model.abort_request("missing") == "NO_OP"
        assert await model.abort_request("r") == "DONE"
        assert len(received) == 3
        for _, body in received[:2]:
            assert body["rid"] == "r" and body["text"] == "prompt"
            assert (
                body["bootstrap_host"] == "producer" and body["bootstrap_port"] == 12345
            )
            assert body["bootstrap_room"] == 123
            assert "bootstrap_room" not in body["sampling_params"]
        assert received[2] == ("/abort_request", {"rid": "r"})
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("failure", [False, True])
def test_runtime_startup_authentication_is_scoped_and_restored(monkeypatch, failure):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ..runtime import create_runtime

    endpoint = Mock(return_value="authenticated endpoint")
    module = SimpleNamespace(RuntimeEndpoint=endpoint)
    monkeypatch.setitem(
        sys.modules, "sglang.lang.backend", SimpleNamespace(runtime_endpoint=module)
    )

    def runtime(**config):
        assert module.RuntimeEndpoint("http://replica") == "authenticated endpoint"
        if failure:
            raise RuntimeError("startup failed")
        return "runtime"

    if failure:
        with pytest.raises(RuntimeError, match="startup failed"):
            create_runtime(runtime, api_key="replica credential")
    else:
        assert create_runtime(runtime, api_key="replica credential") == "runtime"
    endpoint.assert_called_once_with("http://replica", api_key="replica credential")
    assert module.RuntimeEndpoint is endpoint


def test_runtime_without_api_key_needs_no_optional_sdk_import():
    from unittest.mock import Mock

    from ..runtime import create_runtime

    runtime = Mock(return_value="runtime")
    assert create_runtime(runtime, dtype="float16") == "runtime"
    runtime.assert_called_once_with(dtype="float16")
