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
from unittest.mock import Mock

import pytest

from ..transformers.gemma4 import Gemma4ChatModel
from ..transformers.multimodal.gemma3 import Gemma3ChatModel
from ..transformers.multimodal.glm4v import Glm4VModel
from ..transformers.multimodal.minicpmv46 import MiniCPMV46Model
from ..transformers.multimodal.minicpmv47 import MiniCPMV47Model

# The generation configs of these checkpoints enable sampling.
ADAPTERS = [
    Gemma3ChatModel,
    Gemma4ChatModel,
    Glm4VModel,
    MiniCPMV46Model,
    MiniCPMV47Model,
]


def generate(**kwargs):
    from transformers import TemperatureLogitsWarper

    streamer = kwargs["streamer"]
    try:
        if kwargs.get("do_sample", True):
            TemperatureLogitsWarper(float(kwargs.get("temperature", 1.0)))
    except ValueError:
        streamer.end()
        raise
    streamer.on_finalized_text("answer", stream_end=True)


@pytest.fixture(params=ADAPTERS, ids=lambda cls: cls.__name__)
def adapter(request):
    from transformers import BatchFeature

    model = object.__new__(request.param)
    model.model_uid = "temperature-test"
    model.reasoning_parser = None
    model.tool_parser = None
    model._pytorch_model_config = {}
    model._tokenizer = Mock(encode=lambda text, **kwargs: text.split())
    model.build_inputs_from_messages = Mock(
        return_value=BatchFeature({"input_ids": [[1, 2, 3]]})
    )
    model._model = SimpleNamespace(generate=Mock(side_effect=generate))
    return model


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("temperature", [0, 1e-6, 0.7, None])
def test_temperature_reaches_generate(adapter, stream, temperature):
    config = {"stream": stream}
    if temperature is not None:
        config["temperature"] = temperature
    result = adapter.build_direct_chat_result(
        [{"role": "user", "content": "Hi"}], config
    )
    if stream:
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in result)
    else:
        text = result["choices"][0]["message"]["content"]
    assert text == "answer"

    kwargs = adapter._model.generate.call_args.kwargs
    assert kwargs.get("do_sample", True) is (temperature != 0)
    if temperature == 0:
        assert "temperature" not in kwargs
    elif temperature is not None:
        assert kwargs["temperature"] == temperature
