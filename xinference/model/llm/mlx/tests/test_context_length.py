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
from unittest.mock import AsyncMock, Mock

import pytest

from ..core import MLXModel, MLXVisionModel


def make_model(vision=False, override=4096):
    family = SimpleNamespace(
        model_specs=[SimpleNamespace(model_revision="test", quantization="4bit")],
        stop=None,
        stop_token_ids=None,
    )
    cls = MLXVisionModel if vision else MLXModel
    config = {"context_length": override} if override is not None else {}
    model = cls("context-test", family, "/unused", model_config=config)
    model._context_length = 262144
    model._model = object()
    model._tokenizer = SimpleNamespace(
        encode=lambda prompt: list(range(len(prompt))), eos_token_id=99
    )
    # A cache hit must not remove cached tokens from the context budget.
    model._get_prompt_cache = Mock(return_value=[1, 2])
    model._generate_stream_inner = Mock(
        side_effect=lambda **kwargs: iter([SimpleNamespace(token=10, text="ok")])
    )
    return model


def set_batch_model(model):
    async def chunks(**kwargs):
        yield {"choices": [{"text": "ok"}], "usage": {}}

    model._batch_model = SimpleNamespace(
        generate=AsyncMock(return_value=("ok", {})),
        generate_stream=Mock(side_effect=chunks),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("input_length", [4096, 5000])
@pytest.mark.parametrize("max_tokens", [None, 8192])
async def test_text_rejects_full_context_before_returning_stream(
    batch, stream, input_length, max_tokens
):
    model = make_model()
    if batch:
        set_batch_model(model)
    with pytest.raises(ValueError, match=f"Prompt has {input_length} tokens"):
        await model.async_generate(
            "a" * input_length, {"stream": stream, "max_tokens": max_tokens}
        )
    model._get_prompt_cache.assert_not_called()
    model._generate_stream_inner.assert_not_called()
    if batch:
        model._batch_model.generate.assert_not_called()
        model._batch_model.generate_stream.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("input_length,remaining", [(4000, 96), (4095, 1)])
@pytest.mark.parametrize("max_tokens", [None, 8192, 1])
async def test_text_caps_output_using_full_prompt_including_cached_tokens(
    batch, stream, input_length, remaining, max_tokens
):
    model = make_model()
    if batch:
        set_batch_model(model)
    result = await model.async_generate(
        "a" * input_length, {"stream": stream, "max_tokens": max_tokens}
    )
    if stream:
        assert [chunk async for chunk in result]
    if batch:
        call = (
            model._batch_model.generate_stream
            if stream
            else model._batch_model.generate
        ).call_args
        assert len(call.kwargs["prompt_token_ids"]) == input_length
    else:
        call = model._generate_stream_inner.call_args
        assert call.kwargs["prompt_token_ids"] == [1, 2]
    assert call.kwargs["max_tokens"] == (1 if max_tokens == 1 else remaining)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("input_length", [4096, 5000])
@pytest.mark.parametrize("max_tokens", [None, 8192])
def test_vision_rejects_processed_image_context_before_returning_stream(
    stream, input_length, max_tokens
):
    model = make_model(vision=True)
    model._prepare_inputs = Mock(return_value=(object(), input_length, 0))
    try:
        with pytest.raises(ValueError, match=f"Prompt has {input_length} tokens"):
            model.generate(
                {"prompt": "short text", "multi_modal_data": {"image": "image"}},
                {"stream": stream, "max_tokens": max_tokens},
            )
        model._prepare_inputs.assert_called_once()
        model._generate_stream_inner.assert_not_called()
    finally:
        model.stop()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("input_length,remaining", [(4000, 96), (4095, 1)])
@pytest.mark.parametrize("max_tokens", [None, 8192, 1])
def test_vision_caps_output_and_reuses_processed_inputs(
    stream, input_length, remaining, max_tokens
):
    model = make_model(vision=True)
    processed_inputs = object()
    model._prepare_inputs = Mock(return_value=(processed_inputs, input_length, 0))
    try:
        result = model.generate(
            {"prompt": "short text", "multi_modal_data": {"image": "image"}},
            {
                "stream": stream,
                "max_tokens": max_tokens,
                "_anthropic_cache_created_tokens": 32,
            },
        )
        if stream:
            chunks = list(result)
            assert chunks
            result = chunks[-1]
        model._prepare_inputs.assert_called_once()
        kwargs = model._generate_stream_inner.call_args.kwargs
        assert kwargs["prompt_token_ids"] is processed_inputs
        assert kwargs["max_tokens"] == (1 if max_tokens == 1 else remaining)
        assert result["usage"]["cache_creation_input_tokens"] == 32
    finally:
        model.stop()


def test_omitted_context_keeps_native_max_tokens_behavior():
    model = make_model(override=None)
    assert model._get_max_tokens(4000, 8192) == 8192
    assert model._get_max_tokens(4000, None) == 262144 - 4000


def test_text_rejects_before_mutating_prompt_cache():
    model = make_model()
    with pytest.raises(ValueError, match="context_length is 4096"):
        model._prepare_inputs("a" * 5000, {"max_tokens": None})
    model._get_prompt_cache.assert_not_called()
