# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..llm_family import match_llm
from ..reasoning_parser import ReasoningParser
from ..sglang.core import SGLANGChatModel
from ..vllm.core import VLLMChatModel
from ..weight_cache import WeightCachedModel, validate_reload_patch
from .test_weight_cache import FakeCachedModel


def test_reasoning_config_dataclass_is_written_to_daemon_config(monkeypatch):
    import json

    from .. import weight_cache

    @dataclass
    class StructuredConfig:
        backend: str = "auto"
        reasoning_parser: str = "/models/reasoning_parser.py"

    config = StructuredConfig()
    daemon = weight_cache.WeightCacheDaemon(
        "vllm",
        "/models/test",
        {"structured_outputs_config": config, "kv_cache_memory_bytes": 1},
    )
    process = MagicMock()
    process.poll.return_value = None

    def start(*args, **kwargs):
        (daemon.directory / "rank.sock").touch()
        return process

    monkeypatch.setattr(weight_cache.subprocess, "Popen", start)
    monkeypatch.setattr(
        weight_cache, "_terminate_process_group", lambda *args, **kwargs: None
    )
    try:
        daemon.start()
        saved = json.loads((daemon.directory / "config.yaml").read_text())
        assert saved["structured_outputs_config"] == {
            "backend": "auto",
            "reasoning_parser": config.reasoning_parser,
        }
        assert daemon.config["structured_outputs_config"] == config
    finally:
        daemon.stop()


@pytest.mark.parametrize("engine", [VLLMChatModel, SGLANGChatModel])
@pytest.mark.parametrize("cached", [False, True])
def test_reasoning_update_keeps_engine_and_parser_snapshot(engine, cached):
    family = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
    model = engine("test", family, "/models/test", {"enable_weight_cache": cached})
    old_parser = ReasoningParser(False, "<think>", "</think>")
    model.reasoning_parser = old_parser
    model._engine = object()
    old_engine = model._engine
    model._weight_cache = MagicMock() if cached else None
    model.load = MagicMock(side_effect=AssertionError("engine must not load"))
    model._stop_engine = MagicMock(side_effect=AssertionError("engine must not stop"))
    model.reload({"reasoning_content": True}, lambda stage: None)
    assert model._engine is old_engine
    assert model.reasoning_parser is not old_parser
    assert not old_parser.reasoning_content
    assert model.reasoning_parser.reasoning_content
    assert model.reasoning_parser.enable_thinking == old_parser.enable_thinking
    assert model.reasoning_parser.extract_reasoning_content(
        "<think>analysis</think>answer"
    ) == ("analysis", "answer")
    metadata = model.get_reload_config()
    assert metadata["enabled"] is cached
    assert metadata["hot_parameters"] == ["reasoning_content"]
    assert metadata["model_config"]["reasoning_content"] is True
    assert "limit_mm_per_prompt" not in metadata["parameters"]
    if cached:
        model._weight_cache.client_config.assert_not_called()
    model.reload({"reasoning_content": False}, lambda stage: None)
    assert not model.reasoning_parser.reasoning_content
    assert model._engine is old_engine


def test_nonreasoning_model_does_not_advertise_or_accept_parser_updates():
    family = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
    model = VLLMChatModel("test", family, "/models/test", {})
    model.prepare_parse_reasoning_content(False)
    assert "reasoning_content" not in model.get_reload_config()["parameters"]
    assert model.get_reload_config()["hot_parameters"] == []
    with pytest.raises(ValueError, match="does not support reasoning-content"):
        model.reload({"reasoning_content": True}, lambda stage: None)


def make_vision_model(monkeypatch, backend="vllm"):
    class EngineArgs:
        def __init__(self, **config):
            self.config = config

        def create_engine_config(self):
            return self.config

    monkeypatch.setitem(
        sys.modules,
        "vllm.engine.arg_utils",
        SimpleNamespace(AsyncEngineArgs=EngineArgs),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.server_args",
        SimpleNamespace(ServerArgs=lambda **config: config),
    )
    model = FakeCachedModel()
    model._weight_cache_engine = backend
    model.model_path = "/models/test"
    model.model_family = SimpleNamespace(model_ability=["vision"])
    field = "limit_mm_per_prompt" if backend == "vllm" else "limit_mm_data_per_request"
    model._reload_config[field] = {"image": 2, "video": 0}
    model._weight_cache.engine_config = lambda config: config
    model.validate_reload = lambda patch: WeightCachedModel.validate_reload(
        model, patch
    )
    return model


@pytest.mark.parametrize("previous", [None, {}, {"video": 0}, {"image": 0, "video": 0}])
def test_sglang_request_limits_can_start_unlimited_or_disabled(monkeypatch, previous):
    model = make_vision_model(monkeypatch, "sglang")
    field = "limit_mm_data_per_request"
    model._reload_config[field] = previous
    assert model.get_reload_config()["parameters"][field] == "image_limits"
    cache = model._weight_cache
    model.reload({field: {"image": 8}}, lambda stage: None)
    assert model._weight_cache is cache and model.stops == 1
    assert model.loads[0][0][field] == {**(previous or {}), "image": 8}
    assert model.get_reload_config()["model_config"][field] == {
        **(previous or {}),
        "image": 8,
    }


@pytest.mark.parametrize("cached", [False, True])
def test_sglang_image_reload_capabilities_and_modality_preservation(
    monkeypatch, cached
):
    model = make_vision_model(monkeypatch, "sglang")
    field = "limit_mm_data_per_request"
    model._enable_weight_cache = cached
    if not cached:
        model._weight_cache = None
        assert field not in model.get_reload_config()["parameters"]
        with pytest.raises(ValueError, match="enable_weight_cache=true"):
            model.reload({field: {"image": 8}}, lambda stage: None)
        assert model.stops == 0
        return
    cache, engine = model._weight_cache, model.engine
    model.reload({field: {"image": 8}}, lambda stage: None)
    assert model._weight_cache is cache and model.engine is not engine
    assert model.loads[0][0][field] == {"image": 8, "video": 0}
    assert model.get_reload_config()["model_config"][field] == {"image": 8, "video": 0}
    with pytest.raises(ValueError, match="Only the image count"):
        model.validate_reload({field: {"image": 8, "video": 1}})
    model.model_family.model_ability = ["chat"]
    assert field not in model.get_reload_config()["parameters"]
    with pytest.raises(ValueError, match="requires a vision model"):
        model.validate_reload({field: {"image": 8}})


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "limits,expected",
    [
        ('{"image": 2, "video": 0}', {"image": 2, "video": 0}),
        ('{"image": 0, "video": 0}', {"image": 0, "video": 0}),
        ("", None),
        (" \t ", None),
    ],
)
def test_sglang_normalizes_limits_before_launch_and_cache_snapshot(
    monkeypatch, cached, limits, expected
):
    from .. import weight_cache
    from ..sglang import core
    from ..sglang.core import SGLANGVisionModel

    field = "limit_mm_data_per_request"
    family = match_llm("qwen2.5-vl-instruct", "pytorch", "3", "none", "modelscope")
    model = SGLANGVisionModel(
        "test",
        family,
        "/models/test",
        {
            "enable_weight_cache": cached,
            field: limits,
        },
    )
    monkeypatch.setattr(model, "_get_cuda_count", lambda: 1)
    monkeypatch.setattr(model, "prepare_parse_tool_calls", lambda: None)
    monkeypatch.setattr(core, "get_next_port", lambda: 30001)
    monkeypatch.setattr(
        core.multiprocessing, "set_start_method", lambda *args, **kw: None
    )
    monkeypatch.setattr(weight_cache, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(weight_cache, "package_version", lambda engine: "0.5.21")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    daemon = MagicMock()
    daemon.client_config.return_value = {"weight_cache_mode": "client"}
    preload = MagicMock(return_value=daemon)
    monkeypatch.setattr(weight_cache, "WeightCacheDaemon", preload)
    runtime = SimpleNamespace(pid=1, shutdown=MagicMock())
    construct = MagicMock(return_value=runtime)
    monkeypatch.setitem(
        sys.modules, "sglang", SimpleNamespace(__version__="0.5.21", Runtime=construct)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.server_args",
        SimpleNamespace(ServerArgs=lambda **config: config),
    )
    try:
        model.load()
        assert construct.call_args.kwargs[field] == expected
        zero_images = expected is not None and expected["image"] == 0
        assert (
            construct.call_args.kwargs.get("skip_server_warmup", False) is zero_images
        )
        metadata = model.get_reload_config()
        if cached:
            preload.assert_called_once()
            assert preload.call_args.args[2][field] == expected
            assert metadata["parameters"][field] == "image_limits"
            assert metadata["model_config"].get(field) == expected
            if expected is not None:
                # Native engine construction can mutate its own config, but
                # must not alter the committed reload snapshot.
                model._model_config[field]["image"] = 99
                assert model.get_reload_config()["model_config"][field] == expected
            assert model.validate_reload({field: {"image": 8}})[field] == {
                **(expected or {}),
                "image": 8,
            }
            if zero_images:
                model.reload({field: {"image": 8}}, lambda stage: None)
                assert construct.call_count == 2
                assert construct.call_args.kwargs[field] == {"image": 8, "video": 0}
                assert not construct.call_args.kwargs.get("skip_server_warmup", False)
        else:
            preload.assert_not_called()
            assert field not in metadata["parameters"]
    finally:
        model.stop()


@pytest.fixture
def vision_model(monkeypatch):
    return make_vision_model(monkeypatch)


def test_mixed_reasoning_and_engine_patch_requires_weight_cache():
    family = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
    model = VLLMChatModel("test", family, "/models/test", {})
    model.reasoning_parser = ReasoningParser(False, "<think>", "</think>")
    model._stop_engine = MagicMock()
    with pytest.raises(ValueError, match="enable_weight_cache=true"):
        model.reload({"reasoning_content": True, "max_num_seqs": 8}, lambda stage: None)
    model._stop_engine.assert_not_called()
    assert not model.reasoning_parser.reasoning_content


@pytest.mark.parametrize(
    "limits",
    [
        '{"image": 2}',
        ['{"image": 2}'],
        {"image": {"count": 2}},
        {"image": -1},
        {"image": True},
        {"image": None},
    ],
)
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "backend,field",
    [("vllm", "limit_mm_per_prompt"), ("sglang", "limit_mm_data_per_request")],
)
def test_unsupported_image_metadata_is_not_advertised(
    monkeypatch, backend, field, limits, cached
):
    model = make_vision_model(monkeypatch, backend)
    model._enable_weight_cache = cached
    model._reload_config[field] = limits
    metadata = model.get_reload_config()
    assert field not in metadata["parameters"]
    assert field not in metadata["model_config"]


def test_image_reload_requires_vision_model(vision_model):
    vision_model.model_family.model_ability = ["chat"]
    with pytest.raises(ValueError, match="requires a vision model"):
        vision_model.reload({"limit_mm_per_prompt": {"image": 8}}, lambda stage: None)
    assert vision_model.stops == 0


def test_merged_rich_modality_limit_is_rejected_in_preflight(vision_model):
    vision_model._reload_config["limit_mm_per_prompt"]["video"] = {"count": 1}
    with pytest.raises(ValueError, match="image_limits"):
        vision_model.validate_reload({"limit_mm_per_prompt": {"image": 8}})
    assert vision_model.stops == 0


def test_image_reload_preserves_other_limits_and_weight_cache(vision_model):
    model = vision_model
    cache, engine = model._weight_cache, model.engine
    model.reload({"limit_mm_per_prompt": {"image": 8}}, lambda stage: None)
    assert model._weight_cache is cache
    assert model.engine is not engine
    assert model.stops == 1
    assert model.loads[0][0]["limit_mm_per_prompt"] == {"image": 8, "video": 0}
    assert model.get_reload_config()["model_config"]["limit_mm_per_prompt"] == {
        "image": 8,
        "video": 0,
    }


def test_initial_image_limit_snapshot_survives_engine_mutation(
    vision_model, monkeypatch
):
    from .. import weight_cache

    model = vision_model
    model._weight_cache = None
    model._n_worker = 1
    model._model_config["limit_mm_per_prompt"] = {"image": 2, "video": 0}
    monkeypatch.setattr(weight_cache, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(weight_cache, "package_version", lambda engine: "0.31.0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    daemon = MagicMock()
    daemon.client_config.return_value = {}
    monkeypatch.setattr(weight_cache, "WeightCacheDaemon", lambda *args: daemon)
    model._prepare_weight_cache()
    model._model_config["limit_mm_per_prompt"]["image"] = 99
    assert (
        model.get_reload_config()["model_config"]["limit_mm_per_prompt"]["image"] == 2
    )


@pytest.mark.parametrize("previous", [0, None])
def test_enabling_absent_image_encoder_is_rejected_before_stop(vision_model, previous):
    model = vision_model
    model._reload_config["limit_mm_per_prompt"]["image"] = previous
    with pytest.raises(ValueError, match="existing positive image limit"):
        model.reload({"limit_mm_per_prompt": {"image": 8}}, lambda stage: None)
    assert model.stops == 0


def test_changing_other_modalities_is_rejected_before_stop(vision_model):
    with pytest.raises(ValueError, match="Only the image count"):
        vision_model.reload(
            {"limit_mm_per_prompt": {"image": 8, "video": 2}}, lambda stage: None
        )
    assert vision_model.stops == 0


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"image": 0},
        {"image": -1},
        {"image": True},
        {"image": 1.5},
        {"image": {"count": 8}},
        {"image": 8, "other": 1},
    ],
)
@pytest.mark.parametrize(
    "engine,field",
    [("vllm", "limit_mm_per_prompt"), ("sglang", "limit_mm_data_per_request")],
)
def test_invalid_image_limit_shape_is_rejected(limits, engine, field):
    with pytest.raises(ValueError, match="image_limits"):
        validate_reload_patch(engine, {field: limits})
