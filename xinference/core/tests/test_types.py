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


import pytest
from openai.types.completion_create_params import CompletionCreateParamsNonStreaming
from pydantic import create_model_from_typeddict

from ...types import CreateCompletionOpenAI


def test_create_completion_types():
    openai_model = create_model_from_typeddict(CompletionCreateParamsNonStreaming)
    assert CreateCompletionOpenAI.__fields__.keys() == openai_model.__fields__.keys()
    CreateCompletionOpenAI(model="abc", prompt="hello", n=1)
    with pytest.raises(NotImplementedError):
        CreateCompletionOpenAI(model="abc", prompt="hello", n=2)
    with pytest.raises(NotImplementedError):
        CreateCompletionOpenAI(model="abc", prompt="hello", seed=1)
