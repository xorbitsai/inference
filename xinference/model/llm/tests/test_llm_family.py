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
import json

from xinference.model.llm.llm_family import (
    GgmlLLMSpecV1,
    LLMFamilyV1,
    PromptStyleV1,
    PytorchLLMSpecV1,
)


def test_deserialize_llm_family_v1():
    serialized = """{
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
         "model_size_in_billions":2,
         "quantizations": ["q4_0", "q4_1"],
         "model_id":"example/TestModel",
         "model_file_name_template":"TestModel.{quantization}.ggmlv3.bin"
      },
      {
         "model_format":"pytorch",
         "model_size_in_billions":3,
         "quantizations": ["int8", "int4", "none"],
         "model_id":"example/TestModel"
      }
   ],
   "prompt_style": {
       "style_name": "ADD_COLON_SINGLE",
       "system_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.",
       "roles": ["user", "assistant"],
       "intra_message_sep": "\\n### ",
       "inter_message_sep": "\\n### ",
       "stop": null,
       "stop_token_ids": null
   }
}"""
    model_family = LLMFamilyV1.parse_raw(serialized)
    assert isinstance(model_family, LLMFamilyV1)
    assert model_family.version == 1
    assert model_family.model_name == "TestModel"
    assert model_family.model_lang == ["en"]
    assert model_family.model_ability == ["embed", "generate"]
    assert len(model_family.model_specs) == 2

    ggml_spec = model_family.model_specs[0]
    assert ggml_spec.model_format == "ggmlv3"
    assert ggml_spec.model_size_in_billions == 2
    assert ggml_spec.model_id == "example/TestModel"
    assert ggml_spec.model_file_name_template == "TestModel.{quantization}.ggmlv3.bin"

    pytorch_spec = model_family.model_specs[1]
    assert pytorch_spec.model_format == "pytorch"
    assert pytorch_spec.model_size_in_billions == 3
    assert pytorch_spec.model_id == "example/TestModel"

    prompt_style = PromptStyleV1(
        style_name="ADD_COLON_SINGLE",
        system_prompt=(
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions."
        ),
        roles=["user", "assistant"],
        intra_message_sep="\n### ",
        inter_message_sep="\n### ",
    )
    assert prompt_style == model_family.prompt_style


def test_serialize_llm_family_v1():
    ggml_spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
    )
    pytorch_spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=3,
        quantizations=["int8", "int4", "none"],
        model_id="example/TestModel",
    )
    prompt_style = PromptStyleV1(
        style_name="ADD_COLON_SINGLE",
        system_prompt=(
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions."
        ),
        roles=["user", "assistant"],
        intra_message_sep="\n### ",
        inter_message_sep="\n### ",
    )
    model_family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="TestModel",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[ggml_spec, pytorch_spec],
        prompt_style=prompt_style,
    )

    serialized = """{"version": 1, "model_name": "TestModel", "model_lang": ["en"], "model_ability": ["embed", "generate"], "model_description": null, "model_specs": [{"model_format": "ggmlv3", "model_size_in_billions": 2, "quantizations": ["q4_0", "q4_1"], "model_id": "example/TestModel", "model_file_name_template": "TestModel.{quantization}.ggmlv3.bin", "model_local_path": null}, {"model_format": "pytorch", "model_size_in_billions": 3, "quantizations": ["int8", "int4", "none"], "model_id": "example/TestModel", "model_local_path": null}], "prompt_style": {"style_name": "ADD_COLON_SINGLE", "system_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", "roles": ["user", "assistant"], "intra_message_sep": "\n### ", "inter_message_sep": "\n### ", "stop": null, "stop_token_ids": null}}"""
    assert model_family.json() == serialized


def test_builtin_llm_families():
    import os

    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "llm_family.json"
    )
    for json_obj in json.load(open(json_path)):
        LLMFamilyV1.parse_obj(json_obj)


def test_cache_from_huggingface_pytorch():
    from ..llm_family import cache_from_huggingface

    spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=1,
        quantizations=["4-bit", "8-bit", "none"],
        model_id="facebook/opt-125m",
    )
    family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_huggingface(family, spec, quantization=None, tqdm_class=None)

    import os

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))


def test_cache_from_huggingface_ggml():
    from ..llm_family import cache_from_huggingface

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_huggingface(family, spec, quantization=None, tqdm_class=None)

    import os

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
