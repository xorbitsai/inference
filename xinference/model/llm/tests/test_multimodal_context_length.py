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
"""Launch budgets use prepared multimodal inputs before starting generation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ..context_length import ContextLengthExceeded
from ..transformers.multimodal.deepseek_vl2 import DeepSeekVL2ChatModel
from ..transformers.multimodal.gemma3 import Gemma3ChatModel
from ..transformers.multimodal.glm4_1v import Glm4_1VModel
from ..transformers.multimodal.intern_vl import InternVLChatModel
from ..transformers.multimodal.minicpmv46 import MiniCPMV46Model
from ..transformers.multimodal.minicpmv47 import MiniCPMV47Model
from ..transformers.multimodal.ovis2 import Ovis2ChatModel
from ..transformers.multimodal.qwen2_audio import Qwen2AudioChatModel
from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel
from . import test_minicpmv47, test_qwen2_vl

minicpm_adapter = test_minicpmv47.adapter
qwen_adapter = test_qwen2_vl.adapter


@pytest.fixture
def adapter(request, minicpm_adapter, qwen_adapter):
    cls = request.param
    if cls is MiniCPMV47Model:
        model = minicpm_adapter
        model.model_uid = "mini47-context-0"
        model._tokenizer = qwen_adapter._tokenizer
    elif cls is Qwen2VLChatModel:
        model = qwen_adapter
    else:
        model = object.__new__(cls)
        model.model_uid = "multimodal-context-0"
        model.reasoning_parser = None
        model._pytorch_model_config = {}
        model._tokenizer = qwen_adapter._tokenizer
        model._model = qwen_adapter._model
        model.build_inputs_from_messages = qwen_adapter.build_inputs_from_messages
    model._model.conv_template = SimpleNamespace(sep="</s>")
    model._model.generate.side_effect = lambda **kw: kw["streamer"].on_finalized_text(
        "answer", stream_end=True
    )
    return model


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "adapter,media",
    [
        (cls, media)
        for cls in (MiniCPMV47Model, Qwen2VLChatModel, Glm4_1VModel, InternVLChatModel)
        for media in (None, "image", "video")
    ]
    + [(Gemma3ChatModel, None), (MiniCPMV46Model, None)],
    indirect=["adapter"],
)
@pytest.mark.parametrize(
    "prompt_tokens,limit,max_tokens,expected",
    [
        (3, 16, 12, 12),
        (3, 8, 12, 5),
        (8, 8, 6, "reject"),
        (9, 8, 6, "reject"),
        (7, 8, 12, 1),
        (3, 8, "omitted", 5),
        (3, 8, None, 5),
        (3, 4096, "omitted", "default"),
        (3, 4096, None, "default"),
        (7, "missing", 12, 12),
        (7, None, 12, 12),
        (7, "missing", "omitted", "default"),
        (7, None, "omitted", "default"),
        (7, "missing", None, "default"),
        (7, None, None, "default"),
    ],
)
def test_context_budget(
    adapter, monkeypatch, stream, media, prompt_tokens, limit, max_tokens, expected
):
    if limit != "missing":
        adapter._pytorch_model_config["context_length"] = limit
    launch_config = deepcopy(adapter._pytorch_model_config)
    config = {"stream": stream, "mm_processor_kwargs": {"downsample_mode": "4x"}}
    if max_tokens != "omitted":
        config["max_tokens"] = max_tokens
    content = [{"type": "text", "text": "hi"}]
    if media:
        content.append(
            {
                "type": f"{media}_url",
                f"{media}_url": {"url": f"data:{media}/mp4;base64,YQ=="},
            }
        )
    messages = [{"role": "user", "content": content}]
    original = deepcopy((messages, config))
    is_mini = isinstance(adapter, MiniCPMV47Model)
    inputs = (
        adapter._processor.apply_chat_template.return_value
        if is_mini
        else adapter.build_inputs_from_messages.return_value
    )
    inputs["input_ids"] = torch.arange(prompt_tokens).reshape(1, -1)
    if expected == "reject":
        start = Mock(side_effect=AssertionError("rejected request started a thread"))
        monkeypatch.setattr("threading.Thread.start", start)
        with pytest.raises(ContextLengthExceeded, match="context_length.*8"):
            result = adapter.chat(messages, config)
            if stream:
                list(result)
        start.assert_not_called()
        adapter._model.generate.assert_not_called()
    else:
        result = adapter.chat(messages, config)
        response = list(result)[-1] if stream else result
        assert response["usage"]["prompt_tokens"] == prompt_tokens
        kwargs = adapter._model.generate.call_args.kwargs
        if expected == "default":
            expected = 1024 if isinstance(adapter, InternVLChatModel) else 512
        assert kwargs["max_new_tokens"] == expected
        for key, value in inputs.items():
            assert kwargs[key] is value
        if is_mini and media:
            options = adapter._processor.apply_chat_template.call_args.kwargs
            assert (
                options["processor_kwargs"][f"{media}s_kwargs"]["downsample_mode"]
                == "4x"
            )
            assert kwargs["downsample_mode"] == "4x"
    assert (messages, config) == original
    assert adapter._pytorch_model_config == launch_config


@pytest.mark.parametrize("adapter", [Gemma3ChatModel, MiniCPMV46Model], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
def test_image_context_budget(adapter, monkeypatch, stream):
    inputs = adapter.build_inputs_from_messages.return_value
    inputs["pixel_values"] = torch.zeros(1, 3, 2, 2)
    test_context_budget(adapter, monkeypatch, stream, "image", 7, 8, 12, 1)
    kwargs = adapter._model.generate.call_args.kwargs
    assert kwargs["do_sample"] is True
    if isinstance(adapter, MiniCPMV46Model):
        assert kwargs["downsample_mode"] == "16x"
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.8
        assert kwargs["top_k"] == 100
        assert kwargs["repetition_penalty"] == 1.05
    else:
        assert kwargs["temperature"] == 1


@pytest.mark.parametrize(
    "adapter", [Ovis2ChatModel, DeepSeekVL2ChatModel], indirect=True
)
@pytest.mark.parametrize(
    "positions,limit,max_tokens,expected",
    [
        (7, 8, 12, 1),
        (8, 8, 12, "reject"),
        (9, 8, 12, "reject"),
        (7, 8, "omitted", 1),
        (7, 8, None, 1),
        (7, 4096, "omitted", "default"),
        (7, 4096, None, "default"),
        (7, "missing", 12, 12),
        (7, None, 12, 12),
        (7, "missing", "omitted", "default"),
        (7, None, None, "default"),
    ],
)
def test_embedding_context_budget(
    adapter, monkeypatch, positions, limit, max_tokens, expected
):
    is_ovis = isinstance(adapter, Ovis2ChatModel)
    adapter._text_tokenizer = adapter._tokenizer
    adapter._model.llm = adapter._model.language = adapter._model
    adapter._model.generation_config = SimpleNamespace(eos_token_id=2)
    if not is_ovis:
        adapter._model.generate.side_effect = None
        adapter._model.generate.return_value = iter(["answer"])
    if limit != "missing":
        adapter._pytorch_model_config["context_length"] = limit
    config = {} if max_tokens == "omitted" else {"max_tokens": max_tokens}
    original = deepcopy((config, adapter._pytorch_model_config))
    inputs = adapter.build_inputs_from_messages.return_value
    inputs["input_ids"] = torch.arange(3).reshape(1, -1)
    inputs["inputs_embeds"] = torch.zeros(1, positions, 4)
    inputs["attention_mask"] = torch.ones(1, positions)
    expected_kwargs = adapter.build_generate_kwargs(config)
    if expected == "reject":
        start = Mock(side_effect=AssertionError("rejected request started a thread"))
        monkeypatch.setattr("threading.Thread.start", start)
        with pytest.raises(ContextLengthExceeded, match="context_length.*8"):
            adapter.build_streaming_iter([], config)
        start.assert_not_called()
        adapter._model.generate.assert_not_called()
    else:
        iterator, prompt_tokens = adapter.build_streaming_iter([], config)
        assert list(iterator) == ["answer"]
        assert prompt_tokens == 3
        kwargs = adapter._model.generate.call_args.kwargs
        if expected == "default":
            expected = 1024 if is_ovis else 512
        assert kwargs["max_new_tokens"] == expected
        for key, value in inputs.items():
            assert kwargs[key] is value
        for key, value in expected_kwargs.items():
            if key != "max_new_tokens":
                assert kwargs[key] == value
    assert (config, adapter._pytorch_model_config) == original


@pytest.fixture(scope="module")
def qwen2_audio_model():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2AudioConfig,
        Qwen2AudioForConditionalGeneration,
    )

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    audio = dict(d_model=8, encoder_layers=1, encoder_attention_heads=2)
    text = dict(vocab_size=16, hidden_size=8, intermediate_size=16)
    text.update(num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model = Qwen2AudioForConditionalGeneration(
        Qwen2AudioConfig(audio_config=audio, text_config=text)
    ).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(16)}))
    )
    yield model, tokenizer
    torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "checkpoint,prompt,limit,max_tokens,expected",
    [
        pytest.param(8, 300, "missing", 512, 8, id="checkpoint_8_no_budget"),
        pytest.param(8, 300, 320, 512, 8, id="checkpoint_8_budget_320"),
        pytest.param(8, 300, 305, 512, 5, id="checkpoint_8_budget_305"),
        pytest.param(50, 300, 320, 512, 20, id="checkpoint_50_budget_320"),
        pytest.param(None, 300, "missing", 512, 212, id="request_no_budget"),
        pytest.param(None, 300, 320, 512, 20, id="request_budget_320"),
        pytest.param(None, 300, 300, 512, "reject", id="exhausted_budget"),
        pytest.param(8, 512, "missing", 512, 8, id="checkpoint_at_max_length"),
        (None, 300, None, 512, 212),
        (None, 300, 1024, 512, 212),
        (None, 300, 301, 512, 1),
        (None, 300, 299, 512, "reject"),
        (None, 512, "missing", 512, "max_length"),
        (None, 513, "missing", 512, "max_length"),
        (None, 512, 1024, 512, "max_length"),
        (None, 513, 1024, 512, "max_length"),
        (None, 512, 512, 512, "reject"),
        (None, 300, "missing", "omitted", 212),
        (None, 300, "missing", None, 212),
        (None, 300, "missing", 0, 212),
        (None, 300, "missing", 304, 4),
        (None, 1, "missing", 2, 1),
    ],
)
def test_qwen2_audio_generation_limit(
    qwen2_audio_model, monkeypatch, checkpoint, prompt, limit, max_tokens, expected
):
    from transformers import BatchEncoding, GenerationConfig

    model, tokenizer = qwen2_audio_model
    model.generation_config = GenerationConfig(
        max_new_tokens=checkpoint,
        max_length=350,
        pad_token_id=0,
        eos_token_id=None,
        do_sample=False,
    )
    adapter = object.__new__(Qwen2AudioChatModel)
    adapter._model = model
    adapter._processor = SimpleNamespace(tokenizer=tokenizer)
    adapter._pytorch_model_config = {}
    if limit != "missing":
        adapter._pytorch_model_config["context_length"] = limit
    inputs = BatchEncoding({"input_ids": torch.ones((1, prompt), dtype=torch.long)})
    adapter.build_inputs_from_messages = Mock(return_value=inputs)
    config = {} if max_tokens == "omitted" else {"max_tokens": max_tokens}
    original = deepcopy(model.generation_config)
    outputs = []

    def thread_factory(*, target, kwargs):
        return SimpleNamespace(start=lambda: outputs.append(target(**kwargs)))

    thread = Mock(side_effect=thread_factory)
    monkeypatch.setattr(
        "xinference.model.llm.transformers.multimodal.qwen2_audio.Thread", thread
    )
    try:
        if isinstance(expected, str):
            error = ContextLengthExceeded if expected == "reject" else ValueError
            match = (
                "context_length" if expected == "reject" else "Input length.*max_length"
            )
            with pytest.raises(error, match=match) as caught:
                adapter.build_streaming_iter([], config)
            if expected == "max_length":
                assert type(caught.value) is ValueError
                assert str(caught.value) == (
                    f"Input length of input_ids is {prompt}, but `max_length` is set to"
                    " 512. This can lead to unexpected behavior. You should consider"
                    " increasing `max_length` or, better yet, setting `max_new_tokens`."
                )
            thread.assert_not_called()
            assert not outputs
        else:
            _, prompt_tokens = adapter.build_streaming_iter([], config)
            assert prompt_tokens == prompt
            assert outputs[0].shape[-1] - prompt == expected
    finally:
        assert model.generation_config == original
