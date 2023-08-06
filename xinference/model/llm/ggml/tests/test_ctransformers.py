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
import random
import re
import string
import time
from typing import Iterator

import pytest

from xinference.model.llm import GgmlLLMSpecV1, LLMFamilyV1
from xinference.model.llm.ggml.ctransformer import (
    CtransformerGenerateConfig,
    CtransformerModel,
)
from xinference.types import (
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)


class MockPipeline:
    def __init__(self) -> None:
        pass


class MockCtransformersModel(CtransformerModel):
    def load(self):
        self._llm = MockPipeline()

    def generate_stream(self) -> Iterator[Completion]:
        for i in range(5):
            res = f"ctransformers_test_stream_{i}"
            completion_choice = CompletionChoice(
                text=res, index=0, logprobs=None, finish_reason="test_stream"
            )
            completion_chunk = CompletionChunk(
                id=str(f"test_{i}"),
                object="text_completion",
                created=int(time.time()),
                model=self._model_uid,
                choices=[completion_choice],
            )
            completion_usage = CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            )
            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
        yield completion

    def generate(
        self, prompt: str, generate_config: CtransformerGenerateConfig
    ) -> Completion:
        completion_choice = CompletionChoice(
            text="test_ctransformers_generate",
            index=0,
            logprobs=None,
            finish_reason="test",
        )
        completion_chunk = CompletionChunk(
            id=str("test"),
            object="text_completion",
            created=int(time.time()),
            model=self._model_uid,
            choices=[completion_choice],
        )
        completion_usage = CompletionUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        completion = Completion(
            id=completion_chunk["id"],
            object=completion_chunk["object"],
            created=completion_chunk["created"],
            model=completion_chunk["model"],
            choices=completion_chunk["choices"],
            usage=completion_usage,
        )
        return completion


mock_model_spec = GgmlLLMSpecV1(
    model_format="ggmlv3",
    model_size_in_billions=6,
    quantizations=["q2_k", "q4_0"],
    model_id="test_id",
    model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
)

test_model_spec = """{
   "version":1,
   "model_name":"TestModel",
   "model_lang":[
      "en"
   ],
   "model_ability":[
      "embed", "generate"
   ],
   "model_specs":[
      {
         "model_format":"ggmlv3",
         "model_size_in_billions":6,
         "quantizations": ["q2_k", "q4_0"],
         "model_id":"test_id",
         "model_file_name_template":"TestModel.{quantization}.ggmlv3.bin"
      },
      {
         "model_format":"pytorch",
         "model_size_in_billions":3,
         "quantizations": ["int8", "int4", "none"],
         "model_id":"example/TestModel"
      }
   ],
   "prompt_style": null
}"""

mock_model_family = LLMFamilyV1.parse_raw(test_model_spec)


@pytest.fixture
def mock_AutoConfig_Pretrained(mocker):
    # Create a mock of the Child.method() and set its return value
    try:
        from ctransformers import AutoConfig, Config
    except ImportError:
        raise ImportError("ctransformers AutoConfig or Config cannot been imported.")
    mock_from_pretrained = mocker.patch.object(AutoConfig, "from_pretrained")
    config = Config()
    auto_config = AutoConfig(config=config)
    mock_from_pretrained.return_value = auto_config
    return mock_from_pretrained


@pytest.mark.parametrize(
    "model_spec, model_family", [(mock_model_spec, mock_model_family)]
)
def test_ctransformer_init(model_spec, model_family, mock_AutoConfig_Pretrained):
    quantization = "q4_0"
    uid = "".join(random.choice(string.digits) for i in range(15))
    path = "".join(
        random.choice(string.ascii_letters + string.punctuation) for i in range(100)
    )
    model = MockCtransformersModel(
        model_uid=uid,
        model_family=model_family,
        model_spec=model_spec,
        quantization=quantization,
        model_path=path,
        ctransformerModelConfig=None,
    )

    try:
        from ctransformers import AutoConfig
    except ImportError:
        raise ImportError("ctransformers AutoConfig or Config cannot been imported.")

    assert model.model_uid == uid
    assert model.quantization == quantization
    assert model.model_path == path
    assert model._ctransformer_model_config is not None
    assert isinstance(model._ctransformer_model_config, AutoConfig)

    assert isinstance(model.model_spec, GgmlLLMSpecV1)
    assert isinstance(model.model_family, LLMFamilyV1)
    assert isinstance(model.model_family.model_specs[0], GgmlLLMSpecV1)

    assert (
        model.model_family.model_specs[0].model_format == model.model_spec.model_format
    )
    assert model.model_family.model_specs[0].model_format == model_spec.model_format
    assert (
        model.model_family.model_specs[0].model_size_in_billions
        == model.model_spec.model_size_in_billions
    )
    assert (
        model.model_family.model_specs[0].model_size_in_billions
        == model_spec.model_size_in_billions
    )
    assert (
        model.model_family.model_specs[0].quantizations
        == model.model_spec.quantizations
    )
    assert model.model_family.model_specs[0].quantizations == model_spec.quantizations
    assert model.model_family.model_specs[0].model_id == model.model_spec.model_id
    assert model.model_family.model_specs[0].model_id == model_spec.model_id
    assert (
        model.model_family.model_specs[0].model_file_name_template
        == model.model_spec.model_file_name_template
    )
    assert (
        model.model_family.model_specs[0].model_file_name_template
        == model_spec.model_file_name_template
    )
    assert (
        model.model_family.model_specs[0].model_local_path
        == model.model_spec.model_local_path
    )
    assert (
        model.model_family.model_specs[0].model_local_path
        == model_spec.model_local_path
    )

    assert model._llm is None


@pytest.mark.parametrize(
    "model_spec, model_family", [(mock_model_spec, mock_model_family)]
)
def test_model_generate(model_spec, model_family, mock_AutoConfig_Pretrained):
    quantization = "q4_0"
    uid = "".join(random.choice(string.digits) for i in range(100))
    path = "".join(
        random.choice(string.ascii_letters + string.punctuation) for i in range(100)
    )
    model = MockCtransformersModel(
        model_uid=uid,
        model_family=model_family,
        model_spec=model_spec,
        quantization=quantization,
        model_path=path,
        ctransformerModelConfig=None,
    )

    assert model._llm is None

    model.load()
    assert isinstance(model._llm, MockPipeline)

    # generate with stream
    pattern = r"[0-4]"
    for completion in model.generate_stream():
        assert completion["id"].startswith("test_")
        assert re.search(pattern, completion["id"])
        assert completion["choices"][0]["text"].startswith("ctransformers_test_stream_")
        assert re.search(pattern, completion["choices"][0]["text"])
        assert completion["choices"][0]["finish_reason"] == "test_stream"
        assert completion["usage"]["prompt_tokens"] == 10
        assert completion["usage"]["completion_tokens"] == 20
        assert completion["usage"]["total_tokens"] == 30

    # generate without stream
    responses = model.generate("def Helloworld():", generate_config={"stream": True})
    assert responses["object"] == "text_completion"
    assert responses["choices"][0]["text"] == "test_ctransformers_generate"
    assert responses["choices"][0]["finish_reason"] == "test"
    assert responses["usage"]["prompt_tokens"] == 10
    assert responses["usage"]["completion_tokens"] == 20
    assert responses["usage"]["total_tokens"] == 30
