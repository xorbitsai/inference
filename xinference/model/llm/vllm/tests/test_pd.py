# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import logging
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

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


def test_nixl_environment_unique_ports(monkeypatch):
    monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
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


@pytest.mark.parametrize("address", ["0.0.0.0:9997", ":::9997", "[::]:9997"])
def test_nixl_wildcard_uses_local_interface(monkeypatch, address):
    from .. import pd

    monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.getsockname.return_value = ("10.0.0.2", 12345)
    monkeypatch.setattr(pd.socket, "socket", Mock(return_value=sock))
    env = {}
    configure_nixl_environment(env, address)
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.2"
    sock.connect.assert_called_once_with(("8.8.8.8", 80))


def test_nixl_wildcard_offline_hostname_fallback(monkeypatch):
    from .. import pd

    monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.connect.side_effect = OSError("no route")
    monkeypatch.setattr(pd.socket, "socket", Mock(return_value=sock))
    monkeypatch.setattr(pd.socket, "gethostname", lambda: "worker")
    lookup = Mock(return_value="10.0.0.3")
    monkeypatch.setattr(pd.socket, "gethostbyname", lookup)
    env = {}
    configure_nixl_environment(env, "0.0.0.0:9997")
    lookup.assert_called_once_with("worker")
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.0.0.3"


@pytest.mark.parametrize("launch_override", [None, "10.0.0.4"])
def test_nixl_preserves_host_override(monkeypatch, launch_override):
    from .. import pd

    key = "VLLM_NIXL_SIDE_CHANNEL_HOST"
    monkeypatch.setenv(key, "10.0.0.3")
    lookup = Mock(side_effect=AssertionError("must not auto-detect explicit host"))
    monkeypatch.setattr(pd.socket, "socket", lookup)
    env = {} if launch_override is None else {key: launch_override}
    configure_nixl_environment(env, "0.0.0.0:9997")
    assert env[key] == (launch_override or "10.0.0.3")
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "backend,expected", [("NIXL", "nixl"), (" nixl ", "nixl"), (" XAVIER ", "xavier")]
)
def test_backend_normalization(backend, expected):
    assert normalize_xavier_transport_backend(backend) == expected


@pytest.mark.asyncio
async def test_chat_preserves_producer_metadata():
    from ..core import VLLMChatModel

    model = object.__new__(VLLMChatModel)
    model.model_family = MagicMock(
        model_family="test",
        chat_template="template",
        stop=None,
        stop_token_ids=None,
        model_ability=["chat"],
    )
    model.model_family.has_architecture.return_value = False
    model.reasoning_parser = None
    model._get_tokenizer = AsyncMock(return_value=object())
    model.get_full_context = Mock(return_value="prompt")
    model._get_chat_template_kwargs_from_generate_config = Mock(return_value={})
    transfer = {"do_remote_prefill": True, "remote_block_ids": [[1, 2]]}
    completion = {"choices": [], "_pd_kv_transfer_params": transfer}
    model.async_generate = AsyncMock(return_value=completion)
    model._to_chat_completion = Mock(return_value={"choices": []})
    result = await VLLMChatModel.async_chat.__wrapped__(
        model, [{"role": "user", "content": "hello"}], {}, request_id="r"
    )
    model.async_generate.assert_awaited_once_with("prompt", {}, None, request_id="r")
    assert result["_pd_kv_transfer_params"] == transfer


@pytest.mark.parametrize(
    "vllm_version,fields",
    [
        (
            "0.21.0",
            (
                "arrival_time",
                "queued_ts",
                "scheduled_ts",
                "first_token_ts",
                "last_token_ts",
                "first_token_latency",
            ),
        ),
        (
            "0.7.3",
            (
                "arrival_time",
                "first_scheduled_time",
                "first_token_time",
                "finished_time",
                "time_in_queue",
                "scheduler_time",
                "model_forward_time",
                "model_execute_time",
            ),
        ),
    ],
)
def test_pd_metrics_uses_matching_fields_at_debug(
    monkeypatch, caplog, vllm_version, fields
):
    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", Version(vllm_version))
    model = object.__new__(core.VLLMModel)
    model._nixl_config = None
    model._xavier_config = {"role": "prefill"}
    model.model_uid = "p"
    values = {field: i + 1 for i, field in enumerate(fields)}
    output = SimpleNamespace(request_id="r", metrics=SimpleNamespace(**values))
    with caplog.at_level(logging.INFO, logger=core.__name__):
        model._log_pd_request_metrics(output)
    assert not caplog.records
    with caplog.at_level(logging.DEBUG, logger=core.__name__):
        model._log_pd_request_metrics(output)
    record = caplog.records[-1]
    assert record.levelno == logging.DEBUG
    assert record.args[-1] == values
