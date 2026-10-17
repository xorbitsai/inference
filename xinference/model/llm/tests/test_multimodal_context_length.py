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
"""Launch budgets use prepared multimodal IDs before starting generation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ..context_length import ContextLengthExceeded
from ..transformers.multimodal.glm4_1v import Glm4_1VModel
from ..transformers.multimodal.intern_vl import InternVLChatModel
from ..transformers.multimodal.minicpmv47 import MiniCPMV47Model
from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel
from . import test_minicpmv47, test_qwen2_vl

minicpm_adapter = test_minicpmv47.adapter
qwen_adapter = test_qwen2_vl.adapter


@pytest.fixture(
    params=[MiniCPMV47Model, Qwen2VLChatModel, Glm4_1VModel, InternVLChatModel]
)
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
@pytest.mark.parametrize("media", [None, "image", "video"])
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
