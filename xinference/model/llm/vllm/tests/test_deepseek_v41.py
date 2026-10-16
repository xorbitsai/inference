from types import SimpleNamespace

import pytest
from packaging import version

from ...llm_family import match_llm
from .. import core
from .test_speculative_config import _model_for_config


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
def test_model_registration(hub):
    family = match_llm(
        "DeepSeek-V4.1-Flash",
        model_format="fp8",
        model_size_in_billions=552,
        quantization="fp8",
        download_hub=hub,
    )
    assert family is not None
    assert family.architectures == ["DeepseekV41ForCausalLM"]
    assert "vision" in family.model_ability
    assert family.context_length == 1048576
    assert family.tool_parser == "deepseek-v4.1"
    assert family.model_specs[0].model_id == "deepseek-ai/DeepSeek-V4.1-Flash"


@pytest.mark.parametrize(
    "vllm_version, expected",
    [("0.21.0", False), ("0.30.0.dev1", True), ("0.30.0", True)],
)
def test_multimodal_version_gate(monkeypatch, vllm_version, expected):
    monkeypatch.setattr(
        core,
        "_get_effective_vllm_version_for_family",
        lambda _: version.parse(vllm_version),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(core.VLLMMultiModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(core.VLLMMultiModel, "_is_linux", lambda: True)
    family = _model_for_config("DeepseekV41ForCausalLM").model_family
    family.model_ability = ["chat", "vision"]
    result = core.VLLMMultiModel.match_json(
        family, SimpleNamespace(model_format="fp8"), "fp8"
    )
    assert (result if isinstance(result, bool) else result[0]) is expected
    if not expected:
        assert "0.30.0" in result[1]


def test_tokenizer_and_cache_defaults(monkeypatch):
    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.30.0.dev1"))
    model = _model_for_config("DeepseekV41ForCausalLM")
    config = model._sanitize_model_config({})
    assert config["tokenizer_mode"] == "deepseek_v41"
    assert "block_size" not in config
    config = model._sanitize_model_config(
        {"tokenizer_mode": "custom", "block_size": 128}
    )
    assert config["tokenizer_mode"] == "custom"
    assert config["block_size"] == 128


@pytest.mark.asyncio
async def test_native_prompt_encoder():
    model = object.__new__(core.VLLMMultiModel)
    model.model_uid = "deepseek-v41-test"
    model.model_family = _model_for_config("DeepseekV41ForCausalLM").model_family
    model.model_family.model_name = "DeepSeek-V4.1-Flash"
    model.model_family.model_ability = ["chat", "vision", "tools"]
    model.model_family.chat_template = ""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "test.png"}},
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "weather"}}]

    class Tokenizer:
        chat_template = None

        def apply_chat_template(self, actual_messages, **kwargs):
            assert actual_messages == messages
            assert kwargs["tools"] == tools
            assert kwargs["enable_thinking"] is False
            assert kwargs["reasoning_effort"] == 25
            assert "chat_template" not in kwargs
            return "v41-native-prompt"

    tokenizer = Tokenizer()

    async def get_tokenizer(_):
        return tokenizer

    model._get_tokenizer = get_tokenizer
    template, resolved = await model._get_chat_template_and_tokenizer(
        "DeepSeek-V4.1-Flash"
    )
    assert template is None
    assert (
        model.get_full_context(
            messages,
            template,
            tokenizer=resolved,
            tools=tools,
            enable_thinking=False,
            reasoning_effort=25,
        )
        == "v41-native-prompt"
    )


@pytest.mark.asyncio
async def test_image_chat_preserves_openai_parts(monkeypatch):
    import sys
    from types import ModuleType

    model = object.__new__(core.VLLMMultiModel)
    model.model_uid = "deepseek-v41-image-test"
    model.model_family = _model_for_config("DeepseekV41ForCausalLM").model_family
    model.model_family.model_name = "DeepSeek-V4.1-Flash"
    model.model_family.model_ability = ["chat", "vision", "tools"]
    model.model_family.chat_template = ""
    model.reasoning_parser = None
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/test.png"},
                },
            ],
        }
    ]
    tools = [{"type": "function", "function": {"name": "weather"}}]

    class Tokenizer:
        chat_template = None

        def apply_chat_template(self, actual_messages, **kwargs):
            assert actual_messages == messages
            assert kwargs["tools"] == tools
            return "native-image-prompt"

    async def get_tokenizer(_):
        return Tokenizer()

    def materialize(*_):
        pass

    def process_vision_info(actual_messages, **kwargs):
        assert actual_messages[0]["content"][1]["type"] == "image"
        return ["decoded-image"], None, {}

    async def generate(inputs, *args, **kwargs):
        assert inputs["prompt"] == "native-image-prompt"
        assert inputs["multi_modal_data"] == {"image": ["decoded-image"]}
        return {"choices": []}

    media_module = ModuleType("qwen_omni_utils")
    media_module.process_vision_info = process_vision_info
    media_module.process_mm_info = None
    media_module.process_audio_info = None
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", media_module)
    monkeypatch.setattr(core, "materialize_messages_media", materialize)
    model._get_tokenizer = get_tokenizer
    model.async_generate = generate
    model._sanitize_chat_config = lambda config: config
    model._post_process_completion = lambda *args: {"ok": True}
    # Bypass only the engine-installed decorator; exercise the complete chat path.
    result = await core.VLLMMultiModel.async_chat.__wrapped__(
        model,
        messages,
        {"tools": tools, "chat_template_kwargs": {"enable_thinking": False}},
    )
    assert result == {"ok": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("use_tools", [False, True])
@pytest.mark.parametrize("as_json", [False, True])
async def test_effort_none_disables_reasoning_in_chat(
    monkeypatch, stream, use_tools, as_json
):
    import copy
    import json
    import sys
    from types import ModuleType

    from ...core import chat_context_var

    model = object.__new__(core.VLLMMultiModel)
    model.model_uid = "v41-effort-none"
    model.model_family = _model_for_config("DeepseekV41ForCausalLM").model_family
    model.model_family.model_name = "DeepSeek-V4.1-Flash"
    model.model_family.model_ability = [
        "chat",
        "vision",
        "hybrid",
        "reasoning",
        "tools",
    ]
    model.model_family.chat_template = ""
    model.model_family.reasoning_start_tag = "<think>"
    model.model_family.reasoning_end_tag = "</think>"
    model.model_family.tool_parser = "deepseek-v4.1"
    model.prepare_parse_reasoning_content(True, enable_thinking=True)
    model.prepare_parse_tool_calls()
    model._sanitize_chat_config = lambda config: config
    output = "The answer is 42."
    if use_tools:
        output += (
            '<｜DSML｜ calls><｜DSML｜ invoke name="weather">'
            '<｜DSML｜ parameter name="city" string="true">杭州'
            "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
        )

    class Tokenizer:
        chat_template = None

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["thinking"] is False
            assert kwargs["reasoning_effort"] == "none"
            return "chat-mode-prompt"

    async def get_tokenizer(_):
        return Tokenizer()

    async def generate(inputs, config, **kwargs):
        assert chat_context_var.get()["enable_thinking"] is False
        assert not model.reasoning_parser.check_content_parser()
        assert inputs["prompt"] == "chat-mode-prompt"
        completion = {
            "id": "test-completion",
            "object": "text_completion",
            "created": 1,
            "model": model.model_uid,
            "choices": [
                {"text": output, "index": 0, "logprobs": None, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if stream:

            async def chunks():
                # Split natural-language output from the tool-call region.
                for text in [output[:17], output[17:]]:
                    chunk = copy.deepcopy(completion)
                    chunk["choices"][0].update(text=text, finish_reason=None)
                    yield chunk
                completion["choices"][0]["text"] = ""
                yield completion

            return chunks()
        return completion

    media_module = ModuleType("qwen_omni_utils")
    media_module.process_vision_info = lambda *args, **kwargs: (None, None, {})
    media_module.process_mm_info = None
    media_module.process_audio_info = None
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", media_module)
    monkeypatch.setattr(core, "materialize_messages_media", lambda *args: None)
    model._get_tokenizer = get_tokenizer
    model.async_generate = generate
    template_kwargs = {
        "reasoning_effort": "none",
        "enable_thinking": True,
        "thinking": True,
    }
    config = {
        "stream": stream,
        "chat_template_kwargs": (
            json.dumps(template_kwargs) if as_json else template_kwargs
        ),
    }
    if use_tools:
        config["tools"] = [{"type": "function", "function": {"name": "weather"}}]
    token = chat_context_var.set({})
    try:
        result = await core.VLLMMultiModel.async_chat.__wrapped__(
            model, [{"role": "user", "content": "hello"}], config
        )
        if stream:
            chunks = [chunk async for chunk in result]
            deltas = [
                chunk["choices"][0]["delta"] for chunk in chunks if chunk.get("choices")
            ]
            assert (
                "".join(delta.get("content") or "" for delta in deltas)
                == "The answer is 42."
            )
            assert not any(delta.get("reasoning_content") for delta in deltas)
            calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        else:
            message = result["choices"][0]["message"]
            assert message["content"] == "The answer is 42."
            assert not message.get("reasoning_content")
            calls = message.get("tool_calls", [])
        assert len(calls) == (1 if use_tools else 0)
        if use_tools:
            assert calls[0]["function"]["name"] == "weather"
            assert json.loads(calls[0]["function"]["arguments"]) == {"city": "杭州"}
        assert template_kwargs["enable_thinking"] is True
        assert model.reasoning_parser.enable_thinking is True
    finally:
        chat_context_var.reset(token)
