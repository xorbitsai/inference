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

# CI: linux
"""Exercise text adapters' non-streaming prompts without loading model weights."""

import copy
from unittest.mock import AsyncMock

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from .. import BUILTIN_LLM_FAMILIES
from ..mlx.core import MLXChatModel
from ..sglang.core import SGLANGChatModel
from ..transformers.core import PytorchChatModel
from ..vllm.core import VLLMChatModel


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "<s>": 1, "</s>": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="<s>",
        eos_token="</s>",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_class,family_name",
    [
        (model_class, family_name)
        for model_class in (VLLMChatModel, SGLANGChatModel, MLXChatModel)
        for family_name in (
            "llama-3.1-instruct",
            "llama-3.3-instruct",
            "qwen2.5-instruct",
        )
    ]
    # HuatuoGPT and Baichuan-M2 have no built-in MLX spec.
    + [
        (model_class, family_name)
        for model_class in (VLLMChatModel, SGLANGChatModel)
        for family_name in ("HuatuoGPT-o1-LLaMA-3.1", "Baichuan-M2")
    ],
)
@pytest.mark.parametrize("tool_choice", ["auto", None, "none", "no-tools"])
async def test_request_tools_reach_prompt(
    model_class, family_name, tool_choice, tokenizer
):
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == family_name)
    model = object.__new__(model_class)
    model.model_family = family
    model.model_uid = "tools-prompt-test"
    model.reasoning_parser = None
    model.tool_parser = None
    model.prepare_parse_tool_calls()
    model._model_config = {}
    model._model_generation_config = {}
    model._context_length = 8192
    model._tokenizer = tokenizer
    if model_class is VLLMChatModel:
        model._get_tokenizer = AsyncMock(return_value=tokenizer)

    # Only generation is mocked: render the real built-in template and run
    # the adapter's normal request controls and response conversion.
    model.async_generate = AsyncMock(
        return_value={
            "id": "completion-test",
            "model": model.model_uid,
            "created": 1,
            "choices": [{"index": 0, "text": "Hello", "finish_reason": "stop"}],
            "usage": {},
        }
    )
    config = {"stream": False}
    if tool_choice != "no-tools":
        config["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "description": "Look up the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        if tool_choice is not None:
            config["tool_choice"] = tool_choice
    original = copy.deepcopy(config)

    result = await model.async_chat(
        [{"role": "user", "content": "What is the weather in Paris?"}], config
    )

    model.async_generate.assert_awaited_once()
    prompt = model.async_generate.await_args.args[0]
    assert "What is the weather in Paris?" in prompt
    enabled = tool_choice not in ("none", "no-tools")
    assert ("lookup_weather" in prompt) is enabled
    assert ("Look up the weather for a city." in prompt) is enabled
    assert result["choices"][0]["message"]["content"] == "Hello"
    assert config == original


def test_transformers_request_tools_reach_prompt_for_baichuan_m2(tokenizer):
    # The Transformers adapter forwards tools for the same tool-call family sets.
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "Baichuan-M2")
    model = object.__new__(PytorchChatModel)
    model.model_family = family
    model.reasoning_parser = None
    model._tokenizer = tokenizer
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    prompt = model._get_full_prompt(
        [{"role": "user", "content": "What is the weather in Paris?"}], tools, {}
    )

    assert "What is the weather in Paris?" in prompt
    assert "lookup_weather" in prompt
    assert "Look up the weather for a city." in prompt
