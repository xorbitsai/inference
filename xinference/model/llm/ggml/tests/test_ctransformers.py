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
import string

import pytest
from xinference_client.handler.model_handler import RESTfulGenerateModelHandle

from .....client import Client
from ....llm import GgmlLLMSpecV1, LLMFamilyV1
from ..ctransformers import CtransformersModel

mock_model_spec = GgmlLLMSpecV1(
    model_format="ggmlv3",
    model_size_in_billions=6,
    quantizations=["q2_k", "q4_0"],
    model_id="test_id",
    model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
)

test_model_spec = """{
   "version":1,
   "context_length":2048,
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


@pytest.mark.parametrize(
    "model_spec, model_family", [(mock_model_spec, mock_model_family)]
)
def test_ctransformer_init(model_spec, model_family):
    from ctransformers import AutoConfig

    quantization = "q4_0"
    uid = "".join(random.choice(string.digits) for i in range(15))
    path = "".join(
        random.choice(string.ascii_letters + string.punctuation) for i in range(100)
    )
    model = CtransformersModel(
        model_uid=uid,
        model_family=model_family,
        model_spec=model_spec,
        quantization=quantization,
        model_path=path,
        ctransformers_model_config=None,
    )

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
    assert model._llm is None


@pytest.mark.asyncio
async def test_ctransformers_generate(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    assert len(client.list_models()) == 0

    model_uid = client.launch_model(
        model_name="gpt-2",
        model_size_in_billions=1,
        model_format="ggmlv3",
        quantization="none",
    )

    assert len(client.list_models()) == 1

    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulGenerateModelHandle)

    completion = model.generate("AI is going to", generate_config={"max_tokens": 5})
    print(completion)
    assert "id" in completion
    assert "text" in completion["choices"][0]
    assert len(completion["choices"][0]["text"]) > 0

    assert completion["model"] == model_uid

    assert "finish_reason" in completion["choices"][0]
    assert completion["choices"][0]["finish_reason"] == "length"

    assert "prompt_tokens" in completion["usage"]
    assert completion["usage"]["prompt_tokens"] == 4

    assert "completion_tokens" in completion["usage"]
    assert completion["usage"]["completion_tokens"] == 5

    assert "total_tokens" in completion["usage"]
    assert completion["usage"]["total_tokens"] == 9

    client.terminate_model(model_uid=model_uid)
    assert len(client.list_models()) == 0
