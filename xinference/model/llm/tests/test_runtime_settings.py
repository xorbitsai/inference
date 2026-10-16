# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ..llm_family import match_llm
from ..reasoning_parser import ReasoningParser
from ..sglang.core import SGLANGChatModel
from ..vllm.core import VLLMChatModel
from ..weight_cache import WeightCachedModel, validate_reload_patch
from .test_weight_cache import FakeCachedModel


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


@pytest.fixture
def vision_model(monkeypatch):
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
    model = FakeCachedModel()
    model.model_path = "/models/test"
    model.model_family = SimpleNamespace(model_ability=["vision"])
    model._reload_config["limit_mm_per_prompt"] = {"image": 2, "video": 0}
    model._weight_cache.engine_config = lambda config: config
    model.validate_reload = lambda patch: WeightCachedModel.validate_reload(
        model, patch
    )
    return model


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
def test_invalid_image_limit_shape_is_rejected(limits):
    with pytest.raises(ValueError, match="image_limits"):
        validate_reload_patch("vllm", {"limit_mm_per_prompt": limits})
