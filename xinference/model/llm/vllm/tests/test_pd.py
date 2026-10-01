# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import sys
from types import SimpleNamespace

import pytest
from packaging.version import Version

from ..pd import configure_nixl_engine, configure_nixl_environment
from ..xavier.transport import normalize_xavier_transport_backend


def test_nixl_engine_config(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "vllm.config", SimpleNamespace(KVTransferConfig=SimpleNamespace)
    )
    config = {}
    configure_nixl_engine(config, Version("0.21.0"), False)
    transfer = config["kv_transfer_config"]
    assert transfer.kv_connector == "NixlConnector"
    assert transfer.kv_role == "kv_both"
    assert transfer.kv_load_failure_policy == "fail"
    assert "enforce_eager" not in config


@pytest.mark.parametrize(
    "config",
    [
        {"tensor_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"data_parallel_size": 2},
        {"kv_transfer_config": {}},
    ],
)
def test_nixl_rejects_unsupported_config(config):
    with pytest.raises(ValueError):
        configure_nixl_engine(config, Version("0.21.0"), False)


@pytest.mark.parametrize("version,lora", [("0.20.0", False), ("0.21.0", True)])
def test_nixl_rejects_unsupported_version_and_lora(version, lora):
    with pytest.raises(ValueError):
        configure_nixl_engine({}, Version(version), lora)


def test_nixl_environment_unique_ports():
    first, second = {}, {}
    configure_nixl_environment(first, "10.0.0.1:9997")
    configure_nixl_environment(second, "10.0.0.1:9997")
    assert first["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.1"
    assert first["VLLM_NIXL_SIDE_CHANNEL_PORT"] != second["VLLM_NIXL_SIDE_CHANNEL_PORT"]


def test_backend_selection():
    assert normalize_xavier_transport_backend(None) == "xavier"
    assert normalize_xavier_transport_backend("nixl") == "nixl"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_xavier_transport_backend("typo")


@pytest.mark.asyncio
async def test_generate_passes_native_handoff_and_returns_producer_metadata(
    monkeypatch,
):
    from .. import core

    class SamplingParams:
        def __init__(self, max_tokens=1, **kwargs):
            self.max_tokens = max_tokens
            self.__dict__.update(kwargs)
            self.extra_args = None

    transfer = {"do_remote_prefill": True, "remote_block_ids": [[1, 2]]}
    received = []

    class Engine:
        async def generate(self, prompt, params, request_id, **kwargs):
            received.append(params.extra_args)
            yield SimpleNamespace(kv_transfer_params=transfer)

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.21.0"))
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    model = object.__new__(core.VLLMModel)
    model._nixl_config = {"role": "prefill"}
    model._engine = Engine()
    model._active_request_ids = set()
    model.lora_requests = []
    model.reasoning_parser = None
    model.model_uid = "p"
    model._convert_request_output_to_completion = lambda *a, **kw: {"choices": []}
    result = await core.VLLMModel.async_generate.__wrapped__(
        model,
        "prompt",
        {"max_tokens": 1, "_pd_kv_transfer_params": {"do_remote_decode": True}},
        request_id="r",
    )
    assert received == [{"kv_transfer_params": {"do_remote_decode": True}}]
    assert result["_pd_kv_transfer_params"] == transfer
    assert not model._active_request_ids
