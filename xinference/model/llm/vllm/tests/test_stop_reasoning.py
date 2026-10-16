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

from unittest.mock import AsyncMock, MagicMock

import pytest

from ..core import VLLMChatModel


@pytest.fixture
def model():
    model = object.__new__(VLLMChatModel)
    model.model_family = MagicMock(
        model_ability=["chat", "reasoning"], stop=["<end>"], stop_token_ids=[2]
    )
    model.model_family.has_architecture.return_value = False
    model.reasoning_parser = None
    model._get_chat_template_kwargs_from_generate_config = MagicMock(return_value={})
    model._get_tokenizer = AsyncMock(return_value=object())
    model.get_full_context = MagicMock(return_value="prompt")
    return model


@pytest.mark.parametrize("stop", ["10", ["10"]])
def test_reasoning_keeps_caller_stop(model, stop):
    assert model._sanitize_chat_config({"stop": stop}) == {"stop": stop}


def test_reasoning_keeps_caller_stop_token_ids(model):
    assert model._sanitize_chat_config({"stop_token_ids": [10]}) == {
        "stop_token_ids": [10]
    }


@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"stop": None, "stop_token_ids": None},
        {"stop": [], "stop_token_ids": []},
        {"stop": "", "stop_token_ids": []},
    ],
)
def test_missing_caller_stops(model, reasoning, config):
    model.model_family.model_ability = ["chat", "reasoning"] if reasoning else ["chat"]
    result = model._sanitize_chat_config(None if config is None else config.copy())
    if reasoning:
        assert not result.get("stop")
        assert not result.get("stop_token_ids")
    else:
        assert result == {"stop": ["<end>"], "stop_token_ids": [2]}
        result["stop"].append("other")
        result["stop_token_ids"].append(3)
    assert model.model_family.stop == ["<end>"]
    assert model.model_family.stop_token_ids == [2]


@pytest.mark.parametrize("stop", ["10", ["10"]])
def test_non_reasoning_keeps_caller_stops(model, stop):
    model.model_family.model_ability = ["chat"]
    config = {"stop": stop, "stop_token_ids": [10]}
    assert model._sanitize_chat_config(config.copy()) == config
    assert model.model_family.stop == ["<end>"]
    assert model.model_family.stop_token_ids == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "config", [{"stop": "10"}, {"stop": ["10"]}, {"stop_token_ids": [10]}]
)
async def test_chat_preparation_keeps_caller_stops(model, stream, config):
    config = dict(config, stream=stream)
    prompt, _, _, prepared = await model._prepare_chat_inputs(
        [{"role": "user", "content": "Count from 1 to 20"}], config.copy()
    )
    assert prompt == "prompt"
    assert prepared == config
