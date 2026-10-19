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
import inspect
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ...context_length import ContextLengthExceeded

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
    model._pytorch_model_config = {}
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


@pytest.mark.parametrize("return_dict", [False, True])
def test_non_streaming_without_audio(adapter, monkeypatch, return_dict):
    import torch

    model, _ = adapter
    model.model_uid = "omni-thinking-0"
    generated_ids = torch.tensor([[1, 2, 3, 4]])
    result = SimpleNamespace(sequences=generated_ids) if return_dict else generated_ids
    model._model.generate = Mock(return_value=result)
    model._processor = Mock()
    model._processor.batch_decode.return_value = ["answer"]
    soundfile = Mock()
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    with patch("builtins.__import__", wraps=__import__) as imports:
        response = model.generate_non_streaming([], {"max_tokens": 2})
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "answer"},
            "finish_reason": "stop",
        }
    ]
    assert response["id"].startswith("chat")
    assert response["object"] == "chat.completion"
    assert response["model"] == "omni-thinking-0"
    assert isinstance(response["created"], int)
    assert response["usage"] == dict(
        prompt_tokens=-1, completion_tokens=-1, total_tokens=-1
    )
    model._processor.batch_decode.assert_called_once_with(
        [[3, 4]], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert not any(call.args[0] == "soundfile" for call in imports.call_args_list)
    assert soundfile.mock_calls == []


@pytest.mark.parametrize("mode", ["return_audio", "generation_mode", "non_streaming"])
@pytest.mark.parametrize(
    "limit,max_tokens,expected",
    [
        (8, 12, 1),
        (8, None, 1),
        (8, 1, 1),
        (10, 12, 3),
        (32, 8, 8),
        (7, 12, "reject"),
        (6, 12, "reject"),
        (None, None, 512),
    ],
)
def test_context_budget(adapter, monkeypatch, mode, limit, max_tokens, expected):
    import torch

    model, _ = adapter
    if limit is not None:
        model._pytorch_model_config["context_length"] = limit
    model.model_uid = "omni-context-0"
    inputs = model.build_inputs_from_messages.return_value
    inputs["input_ids"] = torch.arange(7).reshape(1, -1)
    audio = torch.zeros(4)
    generate = Mock(return_value=(torch.arange(9).reshape(1, -1), audio))
    generate.__signature__ = inspect.Signature(
        [inspect.Parameter(mode, inspect.Parameter.KEYWORD_ONLY)]
    )
    model._model = SimpleNamespace(generate=generate)
    model._processor = Mock()
    model._processor.batch_decode.return_value = ["answer"]
    soundfile = Mock()
    soundfile.write.side_effect = lambda output, *args, **kw: output.write(b"wav")
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    thread = Mock()
    monkeypatch.setattr(omni, "Thread", thread)
    config = {"temperature": 0.25, "use_audio_in_video": False}
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    def run():
        if mode == "non_streaming":
            return model.generate_non_streaming([], config)
        return model.build_streaming_iter([], config)

    if mode == "non_streaming" and expected == 1 and max_tokens != 1:
        expected = "audio_reject"
    if expected in ("reject", "audio_reject"):
        message = (
            "Prompt length 7"
            if expected == "reject"
            else "Prompt leaves 1 output token.*context_length 8.*audio generation needs at least 2 thinker tokens"
        )
        with pytest.raises(ContextLengthExceeded, match=message):
            run()
        thread.assert_not_called()
        generate.assert_not_called()
        return
    result = run()
    if mode == "non_streaming":
        kwargs = generate.call_args.kwargs
        assert result["choices"][0]["message"]["content"] == "answer"
        assert result["choices"][0]["message"]["audio"]["data"] == "d2F2"
        assert kwargs["use_audio_in_video"] is False
        assert "return_audio" not in kwargs and "generation_mode" not in kwargs
        model._processor.batch_decode.assert_called_once_with(
            [[7, 8]], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        soundfile.write.assert_called_once()
    else:
        streamer, prompt_tokens = result
        kwargs = thread.call_args.kwargs["kwargs"]
        assert kwargs["streamer"] is streamer
        assert prompt_tokens == 7
        assert kwargs[mode] == (False if mode == "return_audio" else "text")
        thread.return_value.start.assert_called_once_with()
        thread.call_args.kwargs["target"](**kwargs)
    generate.assert_called_once_with(**kwargs)
    assert kwargs["input_ids"] is inputs.input_ids
    assert kwargs["thinker_max_new_tokens"] == expected
    assert kwargs["temperature"] == 0.25
    assert kwargs["speaker"] == "Chelsie"


@pytest.mark.parametrize(
    "adapter", ["Qwen2_5OmniForConditionalGeneration"], indirect=True
)
def test_audio_context_budget_with_real_thinker(adapter, monkeypatch, request):
    import torch
    import transformers

    if not hasattr(transformers, "Qwen2_5OmniThinkerConfig"):
        pytest.skip("Qwen2_5OmniThinkerConfig is unavailable")
    request.addfinalizer(partial(torch.set_num_threads, torch.get_num_threads()))
    torch.set_num_threads(1)
    config = transformers.Qwen2_5OmniThinkerConfig(
        vision_start_token_id=151652,
        text_config=dict(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            eos_token_id=None,
            rope_theta=10000.0,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [2, 2, 2],
            },
        ),
        audio_config=dict(
            encoder_layers=1, encoder_attention_heads=2, d_model=16, output_dim=24
        ),
        vision_config=dict(depth=1, hidden_size=16, num_heads=2, out_hidden_size=24),
    )
    thinker = transformers.Qwen2_5OmniThinkerForConditionalGeneration(config).eval()
    model, _ = adapter
    model.model_uid = "omni-context-0"
    inputs = model.build_inputs_from_messages.return_value
    inputs["input_ids"] = torch.arange(1, 8).reshape(1, -1)
    model._processor = Mock()
    model._processor.batch_decode.return_value = ["answer"]
    boundary = model._model.generate.args[0]
    boundary.thinker = thinker
    boundary.speaker_map["Chelsie"] = {
        "bos_token": 1,
        "cond": torch.zeros(1, 1),
        "ref_mel": torch.zeros(1, 1, 1),
    }
    boundary.talker = SimpleNamespace(
        codec_pad_token=0,
        codec_mask_token=0,
        codec_bos_token=1,
        text_eos_token=2,
        text_pad_token=0,
        generate=Mock(
            side_effect=lambda input_ids, **kw: torch.nn.functional.pad(
                input_ids, (0, 2), value=3
            )
        ),
    )
    boundary.token2wav = Mock(return_value=torch.zeros(4))
    boundary.token2wav.dtype = torch.float
    soundfile = Mock()
    soundfile.write.side_effect = lambda output, *args, **kw: output.write(b"wav")
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    with patch.object(thinker, "generate", wraps=thinker.generate) as generate:
        model._pytorch_model_config["context_length"] = 9
        result = model.generate_non_streaming([], {"max_tokens": 12})
        assert result["choices"][0]["message"]["content"] == "answer"
        assert result["choices"][0]["message"]["audio"]["data"] == "d2F2"
        assert generate.call_args.kwargs["max_new_tokens"] == 2
        generate.reset_mock()
        model._pytorch_model_config["context_length"] = 8
        with pytest.raises(
            ContextLengthExceeded,
            match="audio generation needs at least 2 thinker tokens",
        ):
            model.generate_non_streaming([], {"max_tokens": 12})
        generate.assert_not_called()
