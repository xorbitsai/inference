# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""DeepSeek-V4.1 on SGLang: prompts through deepseek-recipe, images in order."""

import base64
import sys
from types import SimpleNamespace

import pytest
from packaging.version import Version

from ...reasoning_parser import ReasoningParser
from .. import core
from ..core import SGLANGVisionModel, _prompt_payload


class FakeRecipe:
    calls: list = []

    @classmethod
    def from_hf_tokenizer(cls, spec, tokenizer):
        assert spec == "dsv41"
        return cls()

    def encode(self, body, *, effort_budget=None, image_urls=None):
        FakeRecipe.calls.append((body, effort_budget))
        for message in body["messages"]:
            for part in message.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_urls.append(part["image_url"]["url"])
        return [1, 2, 3]


@pytest.fixture
def model(monkeypatch):
    FakeRecipe.calls = []
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.parser.deepseek_recipe_renderer",
        SimpleNamespace(
            DeepSeekRecipe=FakeRecipe,
            parse_effort_budget=lambda value: value,
        ),
    )
    instance = object.__new__(SGLANGVisionModel)
    instance.model_uid = "dsv41"
    instance.model_family = SimpleNamespace(
        model_name="DeepSeek-V4.1-Flash-1017",
        model_family=None,
        stop=None,
        chat_template="",
        model_ability=["chat", "vision", "reasoning", "tools"],
        has_architecture=lambda *names: "DeepseekV41ForCausalLM" in names,
    )
    instance.reasoning_parser = ReasoningParser(
        reasoning_content=True,
        enable_thinking=True,
        allow_disable_thinking=True,
        reasoning_start_tag="<think>",
        reasoning_end_tag="</think>",
    )
    instance._model_config = {}
    instance._engine = SimpleNamespace(get_tokenizer=lambda: object())
    return instance


@pytest.mark.asyncio
async def test_renders_ids_and_passes_images_as_data_urls(model, tmp_path, monkeypatch):
    monkeypatch.setenv("XINFERENCE_MEDIA_ALLOW_LOCAL_PATH", "true")
    image = tmp_path / "red.png"
    image.write_bytes(b"png-bytes")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Which colors?"},
                {"type": "image_url", "image_url": {"url": str(image)}},
                {"type": "image_url", "image_url": "data:image/jpeg;base64,AAAA"},
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    prompt, images, used_tools, context, config = await model._prepare_chat_inputs(
        messages, {"tools": tools, "temperature": 0}
    )
    assert prompt == [1, 2, 3]
    assert images == [
        "data:image/png;base64," + base64.b64encode(b"png-bytes").decode(),
        "data:image/jpeg;base64,AAAA",
    ]
    body, budget = FakeRecipe.calls[0]
    assert body["thinking"] is True and body["tools"] == tools and budget is None
    assert used_tools == tools and context["enable_thinking"] is True
    assert "tools" not in config and config["temperature"] == 0
    # The request's own messages keep their local path.
    assert messages[0]["content"][1]["image_url"]["url"] == str(image)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effort,thinking,budget",
    [("none", False, None), (75, True, 75), (None, True, None)],
)
async def test_reasoning_effort_matches_the_native_renderer(
    model, effort, thinking, budget
):
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    _, images, _, context, _ = await model._prepare_chat_inputs(
        [{"role": "user", "content": "hi"}], {"chat_template_kwargs": kwargs}
    )
    body, used_budget = FakeRecipe.calls[0]
    assert images is None
    assert body["thinking"] is thinking and used_budget == budget
    assert context["enable_thinking"] is thinking and context["thinking"] is thinking


@pytest.mark.asyncio
async def test_rejects_non_image_media(model):
    with pytest.raises(ValueError, match="text and image"):
        await model._prepare_chat_inputs(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "data:video/mp4;base64,AAAA"},
                        }
                    ],
                }
            ],
            {},
        )


def test_token_prompts_are_sent_as_input_ids():
    assert _prompt_payload([1, 2]) == {"input_ids": [1, 2]}
    assert _prompt_payload("text") == {"text": "text"}
    assert _prompt_payload("text", [4]) == {"input_ids": [4]}


@pytest.mark.parametrize("installed,ok", [("0.5.22", False), ("0.5.23", True)])
def test_vision_engine_requires_sglang_with_deepseek_v41(monkeypatch, installed, ok):
    monkeypatch.setattr(SGLANGVisionModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(SGLANGVisionModel, "_is_linux", lambda: True)
    monkeypatch.setattr(core, "_virtual_env_allows_missing_sglang", lambda: False)
    monkeypatch.setattr(core, "_get_sglang_version", lambda: Version(installed))
    monkeypatch.setattr(core, "SGLANG_INSTALLED", True)
    family = SimpleNamespace(
        model_ability=["chat", "vision"],
        architectures=["DeepseekV41ForCausalLM"],
        has_architecture=lambda *names: "DeepseekV41ForCausalLM" in names,
        matches_supported_architectures=lambda names: "DeepseekV41ForCausalLM" in names,
    )
    spec = SimpleNamespace(model_format="fp8", quantization_method=None)
    result = SGLANGVisionModel.match_json(family, spec, "fp8")
    assert (result is True) is ok
    if not ok:
        assert "0.5.23" in result[1]


def test_large_checkpoint_gets_a_longer_server_wait(model):
    assert model._get_launch_timeout() == core.DEEPSEEK_V41_SGLANG_LAUNCH_TIMEOUT
