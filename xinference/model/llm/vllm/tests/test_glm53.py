from types import SimpleNamespace

import pytest
from packaging import version

from ...llm_family import match_llm
from ...utils import GLM5_TOOL_CALL_FAMILY
from .. import core


def render(messages, chat_template, **kwargs):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model = object.__new__(core.VLLMChatModel)
    model.model_family = family_for()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    return model.get_full_context(
        messages, chat_template, tokenizer=tokenizer, **kwargs
    )


def family_for(hub="huggingface", model_format="fp8"):
    return match_llm(
        "glm-5.3",
        model_format=model_format,
        model_size_in_billions=753,
        quantization="FP8" if model_format == "fp8" else "none",
        download_hub=hub,
    )


@pytest.mark.parametrize(
    "hub,org", [("huggingface", "zai-org"), ("modelscope", "ZhipuAI")]
)
@pytest.mark.parametrize("model_format,suffix", [("fp8", ""), ("pytorch", "-BF16")])
def test_registration(hub, org, model_format, suffix):
    family = family_for(hub, model_format)
    assert family is not None
    assert family.model_specs[0].model_id == f"{org}/GLM-5.3{suffix}"
    assert family.context_length == 1048576
    assert family.architectures == ["GlmMoeDsaForCausalLM"]
    assert family.model_ability == ["chat", "tools", "reasoning"]
    assert family.tool_parser == "glm5"
    assert family.model_name in GLM5_TOOL_CALL_FAMILY


@pytest.mark.parametrize("vllm_version,expected", [("0.27.0", False), ("0.28.0", True)])
@pytest.mark.parametrize("custom", [False, True])
def test_vllm_version_gate(monkeypatch, vllm_version, expected, custom):
    monkeypatch.setattr(
        core,
        "_get_effective_vllm_version_for_family",
        lambda _: version.parse(vllm_version),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(core.VLLMChatModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(core.VLLMChatModel, "_is_linux", lambda: True)
    family = family_for().copy(deep=True)
    if custom:
        family.model_family = "glm-5.3"
        family.model_name = "custom-glm53"
    result = core.VLLMChatModel.match_json(
        family, SimpleNamespace(model_format="fp8"), "FP8"
    )
    assert (result if isinstance(result, bool) else result[0]) is expected
    if not expected:
        assert "0.28.0" in result[1]


@pytest.mark.parametrize(
    "effort,label",
    [
        (None, "Max"),
        ("low", "Low"),
        ("high", "High"),
        ("max", "Max"),
        ("medium", "Max"),
    ],
)
def test_reasoning_template(effort, label):
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    prompt = render(
        [{"role": "user", "content": "hello"}],
        family_for().chat_template,
        enable_thinking=False,
        **kwargs,
    )
    assert f"Reasoning Effort: {label}" in prompt
    assert prompt.endswith("<|assistant|><think>")


@pytest.mark.parametrize("clear", [None, False, True])
def test_thinking_history(clear):
    kwargs = {} if clear is None else {"clear_thinking": clear}
    prompt = render(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "private reasoning",
            },
            {"role": "user", "content": "second"},
        ],
        family_for().chat_template,
        **kwargs,
    )
    assert ("private reasoning" in prompt) is (clear is not True)


def test_tool_results_follow_call_order():
    prompt = render(
        [
            {"role": "user", "content": "look up both"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "a",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "a"}'},
                    },
                    {
                        "id": "b",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q": "b"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "result-b"},
            {"role": "tool", "tool_call_id": "a", "content": "result-a"},
        ],
        family_for().chat_template,
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    )
    assert (
        "<tool_call>lookup<arg_key>q</arg_key><arg_value>a</arg_value></tool_call>"
        in prompt
    )
    assert prompt.index("<tool_response>result-a") < prompt.index(
        "<tool_response>result-b"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize(
    "template_kwargs",
    [
        {"enable_thinking": False, "reasoning_effort": "low", "clear_thinking": True},
        '{"enable_thinking": false, "reasoning_effort": "high"}',
        {"thinking": False},
        {},
    ],
)
async def test_chat_always_parses_reasoning(
    monkeypatch, stream, custom, template_kwargs
):
    import copy
    import json

    from ...core import chat_context_var

    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    model = object.__new__(core.VLLMChatModel)
    model.model_uid = "glm53-test"
    model.model_family = family_for().copy(deep=True)
    if custom:
        model.model_family.model_name = "custom-glm53"
        model.model_family.model_family = "glm-5.3"
    model.prepare_parse_reasoning_content(True, True)
    original = copy.deepcopy(template_kwargs)
    expected = json.loads(original) if isinstance(original, str) else original

    async def get_tokenizer(_):
        return None

    def render_prompt(messages, template, **kwargs):
        assert kwargs["enable_thinking"] is True
        if "thinking" in expected:
            assert kwargs["thinking"] is True
        for key in ("reasoning_effort", "clear_thinking"):
            if key in expected:
                assert kwargs[key] == expected[key]
        return "<|assistant|><think>"

    async def generate(prompt, config, tools, request_id=None):
        # async_generate reads this config to retain special tokens, while
        # response parsing reads the request context. Both must be enabled.
        assert config["chat_template_kwargs"]["enable_thinking"] is True
        assert chat_context_var.get()["enable_thinking"] is True
        assert model.reasoning_parser.check_content_parser() is True
        base = {"id": "test", "created": 1, "model": "glm53-test"}
        if stream:

            async def chunks():
                for text, finish in [
                    ("analysis", None),
                    ("</think>answer", None),
                    ("", "stop"),
                ]:
                    yield {
                        **base,
                        "object": "text_completion",
                        "choices": [
                            {"index": 0, "text": text, "finish_reason": finish}
                        ],
                    }

            return chunks()
        return {
            **base,
            "object": "text_completion",
            "choices": [
                {"index": 0, "text": "analysis</think>answer", "finish_reason": "stop"}
            ],
            "usage": None,
        }

    model._get_tokenizer = get_tokenizer
    model.get_full_context = render_prompt
    model.async_generate = generate
    token = chat_context_var.set({})
    try:
        result = await model.async_chat(
            [{"role": "user", "content": "hello"}],
            {"stream": stream, "chat_template_kwargs": template_kwargs},
        )
        if stream:
            chunks = [chunk async for chunk in result]
            deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
            assert "".join(d.get("content") or "" for d in deltas) == "answer"
            assert "analysis" in "".join(
                d.get("reasoning_content") or "" for d in deltas
            )
        else:
            message = result["choices"][0]["message"]
            assert message["content"] == "answer"
            assert message["reasoning_content"].strip() == "analysis"
        assert template_kwargs == original
    finally:
        chat_context_var.reset(token)
