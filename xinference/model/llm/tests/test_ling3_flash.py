# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
from packaging.version import Version

from ..cache_manager import LLMCacheManager
from ..llm_family import match_llm
from ..transformers.ling3 import Ling3PytorchChatModel
from ..vllm import core as vllm_core

VARIANTS = [
    ("pytorch", "none", ""),
    ("fp8", "FP8", "-fp8"),
    ("fp4", "FP4", "-fp4"),
    ("pytorch", "Int4", "-int4"),
]
GGUF_PARTS = {"BF16": 6, "Q4_K_M": 2, "Q5_K_M": 2, "Q6_K": 3, "Q8_0": 3}


@pytest.fixture
def family():
    return match_llm("Ling-3.0-flash", "pytorch", 124, "none", "huggingface").copy(
        deep=True
    )


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize("model_format,quantization,suffix", VARIANTS)
def test_official_checkpoints(
    hub, revision, model_format, quantization, suffix, monkeypatch
):
    family = match_llm("Ling-3.0-flash", model_format, 124, quantization, hub)
    spec = family.model_specs[0]
    assert spec.model_id == f"inclusionAI/Ling-3.0-flash{suffix}"
    assert spec.model_revision == revision
    assert spec.model_hub == hub
    assert spec.activated_size_in_billions == "5_1"
    assert family.context_length == 262144
    assert "vision" not in family.model_ability
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: True)
    assert vllm_core.VLLMChatModel.match_json(family, spec, quantization) is True
    result = Ling3PytorchChatModel.match_json(family, spec, quantization)
    assert (result is True) == (model_format != "fp4")


@pytest.mark.parametrize(
    "hub,revision", [("huggingface", "main"), ("modelscope", "master")]
)
@pytest.mark.parametrize("quantization,parts", GGUF_PARTS.items())
def test_official_gguf_shards(hub, revision, quantization, parts):
    from ..llama_cpp.core import XllamaCppModel

    family = match_llm("Ling-3.0-flash", "ggufv2", 124, quantization, hub)
    spec = family.model_specs[0]
    assert spec.model_id == "inclusionAI/Ling-3.0-flash-GGUF"
    assert spec.model_revision == revision
    files, _, _ = LLMCacheManager(family)._gguf_file_names()
    assert files == [
        f"{quantization}/Ling-3.0-flash-{quantization}-{i:05d}-of-{parts:05d}.gguf"
        for i in range(1, parts + 1)
    ]
    assert XllamaCppModel.match_json(family, spec, quantization) is True
    assert vllm_core.VLLMChatModel.match_json(family, spec, quantization)[0] is False


@pytest.mark.parametrize(
    "engine_version,virtualenv,expected",
    [
        ("0.27.1", False, False),
        ("0.28.0", False, True),
        ("0.29.0", False, True),
        ("0.27.1", True, True),
    ],
)
def test_vllm_version_gate(family, monkeypatch, engine_version, virtualenv, expected):
    monkeypatch.setattr(
        vllm_core, "_virtual_env_allows_missing_vllm", lambda: virtualenv
    )
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version(engine_version))
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    result = vllm_core.VLLMChatModel.match_json(family, family.model_specs[0], "none")
    assert (result is True) == expected


def test_int4_exception_does_not_change_other_models(family, monkeypatch):
    family.architectures = ["LlamaForCausalLM"]
    result = vllm_core.VLLMChatModel.match_json(family, family.model_specs[0], "Int4")
    assert result[0] is False
    assert "pytorch format with quantization" in result[1]


@pytest.mark.parametrize(
    "engine,requirement",
    [
        ("Transformers", "transformers>=4.57.1,<5.0.0"),
        ("vllm", "vllm==0.29.0"),
        ("llama.cpp", "xllamacpp==2026.9.10809"),
    ],
)
def test_engine_install_requirements(family, engine, requirement):
    from ....core.utils import filter_virtualenv_packages_by_markers
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders

    packages = expand_engine_dependency_placeholders(family.virtualenv.packages, engine)
    prepared = filter_virtualenv_packages_by_markers(packages, engine, None)
    name = requirement.split("=")[0].rstrip(">")
    assert [p for p in prepared if p.startswith(name)] == [requirement]


@pytest.mark.parametrize("model_format,quantization,suffix", VARIANTS)
def test_preserve_checkpoint_quantization(
    model_format, quantization, suffix, monkeypatch
):
    family = match_llm("Ling-3.0-flash", model_format, 124, quantization, "huggingface")
    model = object.__new__(vllm_core.VLLMChatModel)
    model.model_family = family
    model.model_spec = family.model_specs[0]
    model._device_count = model._n_worker = 1
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version("0.29.0"))
    assert model._sanitize_model_config({})["quantization"] == (
        "fp8" if model_format == "fp8" else None
    )
    if model_format != "fp4":
        model = Ling3PytorchChatModel("ling-0", family, "/tmp/unused")
        assert model.apply_quantization_config({}) == {}
        assert model._pytorch_model_config["torch_dtype"] == "auto"
        assert model._pytorch_model_config["trust_remote_code"] is True
        assert model.allow_batch is False
        assert model._should_use_batching() is False


@pytest.fixture
def direct_model(family):
    import torch
    from transformers import BatchEncoding

    from ..utils import ChatModelMixin

    model = Ling3PytorchChatModel("ling-0", family, "/tmp/unused")
    model._batch_scheduler = None
    model.prepare_parse_reasoning_content(True, enable_thinking=True)
    model.prepare_parse_tool_calls()
    model.output_text = "Compare colors</think>They match."
    model.generation_kwargs = None

    class Tokenizer:
        pad_token_id = 156892

        def apply_chat_template(self, messages, **kwargs):
            template = kwargs.pop("chat_template")
            kwargs.pop("tokenize", None)
            return ChatModelMixin._compile_jinja_template(template).render(
                messages=messages, **kwargs
            )

        def __call__(self, prompt, **kwargs):
            model.prompt = prompt
            return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

        def encode(self, text, **kwargs):
            return list(range(len(text)))

    def generate(**kwargs):
        model.generation_kwargs = kwargs
        kwargs["streamer"].on_finalized_text(model.output_text, stream_end=True)

    model._tokenizer = Tokenizer()
    model._model = SimpleNamespace(
        generate=generate,
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1)),
    )
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
async def test_transformers_direct_chat(direct_model, stream, thinking):
    model = direct_model
    if not thinking:
        model.output_text = "They match."
    result = await model.chat(
        [{"role": "user", "content": "Compare colors."}],
        {
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "max_tokens": 12,
            "temperature": 0,
        },
    )
    if stream:
        messages = [
            chunk["choices"][0]["delta"] for chunk in result if chunk["choices"]
        ]
        content = "".join(m.get("content") or "" for m in messages)
        reasoning = "".join(m.get("reasoning_content") or "" for m in messages)
    else:
        message = result["choices"][0]["message"]
        content, reasoning = message["content"], message.get("reasoning_content") or ""
    assert content == "They match."
    assert reasoning == ("Compare colors" if thinking else "")
    assert model.generation_kwargs["max_new_tokens"] == 12
    assert model.generation_kwargs["do_sample"] is False
    assert model.generation_kwargs["use_cache"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_transformers_tools(direct_model, stream):
    model = direct_model
    model.output_text = (
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
        "<arg_value>Shanghai</arg_value>\n</tool_call>"
    )
    result = await model.chat(
        [{"role": "user", "content": "Weather?"}],
        {
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )
    if stream:
        chunks = list(result)
        calls = [
            call
            for chunk in chunks
            for choice in chunk["choices"]
            for call in choice["delta"].get("tool_calls", [])
        ]
    else:
        calls = result["choices"][0]["message"]["tool_calls"]
    assert any(call["function"].get("name") == "get_weather" for call in calls)
    assert "Shanghai" in "".join(
        call["function"].get("arguments", "") for call in calls
    )
    assert "get_weather" in model.prompt


@pytest.mark.asyncio
async def test_transformers_generation_failure_propagates(direct_model):
    def fail(**kwargs):
        raise ValueError("generation failed")

    direct_model._model.generate = fail
    with pytest.raises(ValueError, match="generation failed"):
        await direct_model.chat([{"role": "user", "content": "hello"}], {})


def test_tiny_retains_batching():
    family = match_llm("Ling-3.0-tiny", "pytorch", "7_9", "none", "huggingface")
    model = Ling3PytorchChatModel("tiny-0", family, "/tmp/unused")
    assert model.allow_batch is True
    assert model._should_use_batching() is True
