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


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("backend", ["nixl", "xavier"])
@pytest.mark.asyncio
async def test_generate_passes_native_handoff_and_returns_producer_metadata(
    monkeypatch,
    role,
    backend,
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
    model._nixl_config = {"role": role} if backend == "nixl" else None
    model._xavier_config = (
        {"role": role, "gpu_cache_bytes": 0} if backend == "xavier" else None
    )
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
    if role == "prefill":
        assert result["_pd_kv_transfer_params"] == transfer
    else:
        assert "_pd_kv_transfer_params" not in result
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


@pytest.mark.parametrize("value", [True, 1, {}, []])
def test_backend_rejects_non_strings(value):
    with pytest.raises(ValueError, match="Unknown vLLM transfer backend"):
        normalize_xavier_transport_backend(value)


@pytest.mark.parametrize("resolved", ["127.0.1.1", "0.0.0.0", None])
def test_nixl_discovery_failure_names_override(monkeypatch, resolved):
    from .. import pd

    monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.connect.side_effect = OSError("no route")
    monkeypatch.setattr(pd.socket, "socket", Mock(return_value=sock))
    lookup = Mock(return_value=resolved)
    if resolved is None:
        lookup.side_effect = pd.socket.gaierror("no hostname")
    monkeypatch.setattr(pd.socket, "gethostbyname", lookup)
    with pytest.raises(ValueError, match="VLLM_NIXL_SIDE_CHANNEL_HOST"):
        configure_nixl_environment({}, "0.0.0.0:9997")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_native_decode_failure_propagates_before_conversion(monkeypatch, stream):
    from .. import core

    class SamplingParams:
        def __init__(self, max_tokens=1, n=1, **kwargs):
            self.max_tokens = max_tokens
            self.n = n
            self.__dict__.update(kwargs)
            self.extra_args = None

    class Engine:
        abort = AsyncMock()

        async def generate(self, *args, **kwargs):
            yield SimpleNamespace(
                finished=True,
                outputs=[SimpleNamespace(finish_reason="error", token_ids=[], text="")],
            )

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.21.0"))
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    model = object.__new__(core.VLLMModel)
    model._nixl_config = {"role": "decode"}
    model._xavier_config = None
    model._engine = Engine()
    model._active_request_ids = set()
    model.lora_requests = []
    model.reasoning_parser = None
    model.model_uid = "d"
    conversion = Mock(side_effect=AssertionError("failed output must not be converted"))
    model._convert_request_output_to_completion = conversion
    model._convert_request_output_to_completion_chunk = conversion
    with pytest.raises(RuntimeError, match="finish_reason=error"):
        result = await core.VLLMModel.async_generate.__wrapped__(
            model, "prompt", {"max_tokens": 1, "stream": stream}, request_id="failed"
        )
        if stream:
            async for _ in result:
                pytest.fail("error output yielded a success chunk")
    conversion.assert_not_called()
    assert not model._active_request_ids
    model._engine.abort.assert_awaited_once_with("failed")


@pytest.mark.parametrize(
    "multimodal,language_only", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("engine_version", ["0.21.0", "0.22.0"])
def test_nixl_load_configures_engine_args_or_rejects_multimodal(
    monkeypatch, multimodal, language_only, engine_version
):
    from .. import core

    class ReachedEngineArgs(Exception):
        pass

    def engine_args(**kwargs):
        assert kwargs["kv_transfer_config"].kv_connector == "NixlConnector"
        raise ReachedEngineArgs()

    args_factory = Mock(side_effect=engine_args)
    for name, module in {
        "vllm": SimpleNamespace(__version__=engine_version),
        "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=args_factory),
        "vllm.engine.async_llm_engine": SimpleNamespace(AsyncLLMEngine=object),
        "vllm.lora.request": SimpleNamespace(LoRARequest=object),
        "vllm.v1.executor": SimpleNamespace(Executor=object),
        "vllm.config": SimpleNamespace(KVTransferConfig=SimpleNamespace),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(core, "VLLM_VERSION", None)
    monkeypatch.setattr(core, "_init_guided_decoding_classes", lambda: None)
    monkeypatch.setattr(core, "_update_vllm_supported_lists", lambda: None)
    model = object.__new__(core.VLLMMultiModel if multimodal else core.VLLMModel)
    model.model_uid = "pd"
    model.model_path = "unused"
    model.model_spec = SimpleNamespace()
    model._n_worker = 1
    model._model_config = {}
    model._nixl_config = {"role": "prefill"}
    model._xavier_config = None
    model.lora_modules = None
    model._get_cuda_count = lambda: 1
    model._sanitize_model_config = lambda config: {
        "reasoning_content": False,
        "language_model_only": language_only,
    }
    model.prepare_parse_reasoning_content = Mock()
    model.prepare_parse_tool_calls = Mock()
    model._native_mp_route = lambda: (False, "test")
    if multimodal and not language_only:
        with pytest.raises(ValueError, match="text-only"):
            model.load()
        args_factory.assert_not_called()
    elif multimodal and Version(engine_version) < Version("0.22.0"):
        with pytest.raises(ValueError, match="requires vLLM >= 0.22.0"):
            model.load()
        args_factory.assert_not_called()
    else:
        with pytest.raises(ReachedEngineArgs):
            model.load()
        args_factory.assert_called_once()


@pytest.mark.parametrize("tools", [[], [{"type": "function"}]])
@pytest.mark.asyncio
async def test_multimodal_text_chat_preserves_pd_handoff(monkeypatch, tools):
    from .. import core

    model = object.__new__(core.VLLMMultiModel)
    model.model_family = SimpleNamespace(
        model_family="internvl", model_name="internvl", model_ability=["vision"]
    )
    model.model_uid = "p"
    model.reasoning_parser = None
    model.get_specific_prompt = Mock(return_value=("prompt", None))
    model._sanitize_chat_config = lambda config: config
    model._to_chat_completion = Mock(return_value={"choices": []})
    model._post_process_completion = Mock(return_value={"choices": []})
    transfer = {"do_remote_prefill": True, "xavier_direct": {"ticket": "t"}}
    model.async_generate = AsyncMock(return_value={"_pd_kv_transfer_params": transfer})
    monkeypatch.setattr(core, "validate_messages_media", Mock())
    result = await core.VLLMMultiModel.async_chat.__wrapped__(
        model, [{"role": "user", "content": "hello"}], {"tools": tools}
    )
    assert result["_pd_kv_transfer_params"] == transfer


@pytest.mark.parametrize(
    "worker_layout,launch_layout,expected",
    [(None, None, "DS"), ("SD", None, "SD"), ("SD", "DS", "DS")],
)
def test_nixl_conv_layout_default_and_overrides(
    monkeypatch, worker_layout, launch_layout, expected
):
    monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)
    if worker_layout:
        monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", worker_layout)
    env = {"VLLM_SSM_CONV_STATE_LAYOUT": launch_layout} if launch_layout else {}
    configure_nixl_environment(env, "10.0.0.1:9997")
    assert env["VLLM_SSM_CONV_STATE_LAYOUT"] == expected
