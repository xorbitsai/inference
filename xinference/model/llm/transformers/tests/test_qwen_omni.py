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
import importlib
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

omni = importlib.import_module("xinference.model.llm.transformers.multimodal.qwen-omni")


class StopAtThinker(Exception):
    pass


@pytest.fixture(
    params=[
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
    ]
)
def adapter(request):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    cls = getattr(transformers, request.param, None)
    if cls is None:
        pytest.skip(f"{request.param} is unavailable in this Transformers version")
    captured = {}

    def record(**kwargs):
        captured.update(kwargs)
        raise StopAtThinker

    boundary = SimpleNamespace(
        speaker_map={"Chelsie": {}},
        has_talker=True,
        thinker=SimpleNamespace(generate=record),
        talker=SimpleNamespace(codec_pad_token=0),
        config=SimpleNamespace(
            talker_config=SimpleNamespace(
                speaker_id={"chelsie": 0},
                text_config=SimpleNamespace(vocab_size=1024),
                codec_eos_token_id=0,
            )
        ),
    )
    model = object.__new__(omni.QwenOmniChatModel)
    model._model = SimpleNamespace(generate=partial(cls.generate, boundary))
    model._tokenizer = Mock()
    model.build_inputs_from_messages = Mock(
        return_value=transformers.BatchFeature({"input_ids": torch.tensor([[1, 2]])})
    )
    return model, captured


@pytest.mark.parametrize("config,expected", [({"max_tokens": 8}, 8), ({}, 512)])
def test_token_limit_reaches_thinker(adapter, config, expected):
    model, captured = adapter
    kwargs = model.build_generate_kwargs({**config, "temperature": 0.25})
    assert kwargs["speaker"] == "Chelsie"
    with pytest.raises(StopAtThinker):
        model._model.generate(**model.build_inputs_from_messages([], {}), **kwargs)
    assert captured["max_new_tokens"] == expected
    assert captured["temperature"] == 0.25


@pytest.mark.filterwarnings("error::FutureWarning")
def test_streaming_is_text_only(adapter, monkeypatch):
    model, captured = adapter
    thread = Mock()
    monkeypatch.setattr(omni, "Thread", thread)
    streamer, prompt_tokens = model.build_streaming_iter([], {"max_tokens": 8})
    kwargs = thread.call_args.kwargs["kwargs"]
    assert kwargs["streamer"] is streamer
    assert prompt_tokens == 2
    thread.return_value.start.assert_called_once_with()
    with pytest.raises(StopAtThinker):
        thread.call_args.kwargs["target"](**kwargs)
    assert not captured.get("output_hidden_states")
    assert captured["max_new_tokens"] == 8


def test_non_streaming_keeps_audio(adapter, monkeypatch):
    model, captured = adapter
    # Generation stops before audio encoding, so soundfile is not needed.
    monkeypatch.setitem(sys.modules, "soundfile", Mock())
    with pytest.raises(StopAtThinker):
        model.generate_non_streaming([], {"max_tokens": 8})
    assert captured["output_hidden_states"] is True
    assert captured["return_dict_in_generate"] is True
