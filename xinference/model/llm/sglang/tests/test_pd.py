# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from unittest.mock import AsyncMock

import pytest

from ..core import SGLANGModel
from ..pd import SGLangNixlHandoff, configure_nixl


def test_native_sglang_configuration_uses_unique_bootstrap_ports():
    ports = []
    for role in ("prefill", "decode"):
        config = {"dtype": "float16", "tp_size": 1}
        replica = dict(role=role, host="127.0.0.1")
        configure_nixl(config, replica, "0.5.21", 1)
        assert config["disaggregation_mode"] == role
        assert config["disaggregation_transfer_backend"] == "nixl"
        assert config["disaggregation_bootstrap_port"] == replica["port"]
        assert config["dtype"] == "float16"
        ports.append(replica["port"])
    assert ports[0] != ports[1]


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
