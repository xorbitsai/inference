# Copyright 2022-2023 XProbe Inc.
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
import pydantic
import pytest
from pydantic import create_model_from_typeddict

from ...types import (
    CreateCompletion,
    CreateCompletionCTransformers,
    CreateCompletionLlamaCpp,
    CreateCompletionTorch,
    _CreateCompletionOpenAIFallback,
    CreateChatCompletion,
    CreateChatCompletionCTransformers,
    CreateChatCompletionLlamaCpp,
    CreateChatCompletionTorch,
)


def check_fields(a, b):
    both = a.__fields__.keys() & b.__fields__.keys()
    for f in both:
        fa = a.__fields__[f]
        fb = b.__fields__[f]
        print(a, b, f)
        if fa.allow_none and not fb.allow_none:
            raise Exception(
                f"The field '{f}' allow none of {a} and {b} are not valid:\n"
                f"    {fa.allow_none} != {fb.allow_none}"
            )
        if not fa.required and fb.required:
            raise Exception(
                f"The field '{f}' required of {a} and {b} are not valid:\n"
                f"    {fa.required} != {fb.required}"
            )
        if fa.default != fb.default and fa.default is None and fb.default is not None:
            raise Exception(
                f"The field '{f}' default value of {a} and {b} are not equal:\n"
                f"    {fa.default} != {fb.default}"
            )


def test_create_completion_types():
    from openai.types.completion_create_params import CompletionCreateParamsNonStreaming

    openai_model = create_model_from_typeddict(CompletionCreateParamsNonStreaming)
    assert (
        _CreateCompletionOpenAIFallback.__fields__.keys()
        == openai_model.__fields__.keys()
    )

    with pytest.raises(pydantic.ValidationError):
        CreateCompletion()

    with pytest.raises(pydantic.ValidationError):
        CreateCompletion(model="abc", prompt="def", not_exist="jdk")

    CreateCompletion(model="abc", prompt="def")

    types = [
        CreateCompletionTorch,
        CreateCompletionLlamaCpp,
        CreateCompletionCTransformers,
    ]
    for t in types:
        t()
        assert "model" not in t.__fields__
        assert "prompt" not in t.__fields__
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            check_fields(types[i], types[j])


def test_create_chat_completion_types():
    with pytest.raises(pydantic.ValidationError):
        CreateChatCompletion()

    with pytest.raises(pydantic.ValidationError):
        CreateChatCompletion(model="abc", not_exist="jdk")

    with pytest.raises(pydantic.ValidationError):
        CreateChatCompletion(model="abc", messages=[{"role": "invalid"}])

    CreateChatCompletion(model="abc", messages=[{"role": "tool"}])

    types = [
        CreateChatCompletionTorch,
        CreateChatCompletionLlamaCpp,
        CreateChatCompletionCTransformers,
    ]
    for t in types:
        t()
        assert "model" not in t.__fields__
        assert "prompt" not in t.__fields__
        assert "messages" not in t.__fields__
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            check_fields(types[i], types[j])

    # These chat and generate share the same type.
    assert CreateChatCompletionCTransformers is CreateCompletionCTransformers
    assert CreateChatCompletionLlamaCpp is CreateCompletionLlamaCpp
    assert CreateChatCompletionTorch is CreateCompletionTorch
