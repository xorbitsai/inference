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

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from ....constants import XINFERENCE_ENV_MODEL_SRC
from ...utils import is_locale_chinese_simplified, is_valid_model_uri
from ..cache_manager import LLMCacheManager as CacheManager
from ..llm_family import (
    CustomLLMFamilyV2,
    LlamaCppLLMSpecV2,
    LLMFamilyV2,
    PytorchLLMSpecV2,
    convert_model_size_to_float,
    match_llm,
    match_model_size,
)


def test_deserialize_llm_family_v1():
    serialized = """{
   "version":2,
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
         "model_format":"ggufv2",
         "model_size_in_billions":2,
         "quantization": "q4_0",
         "quantization_parts": {
            "q4_2": ["a", "b"]
         },
         "model_id":"example/TestModel",
         "model_file_name_template":"TestModel.{quantization}.bin",
         "model_file_name_split_template":"TestModel.{quantization}.bin.{part}"
      },
      {
         "model_format":"ggufv2",
         "model_size_in_billions":2,
         "quantization": "q4_1",
         "quantization_parts": {
            "q4_2": ["a", "b"]
         },
         "model_id":"example/TestModel",
         "model_file_name_template":"TestModel.{quantization}.bin",
         "model_file_name_split_template":"TestModel.{quantization}.bin.{part}"
      },
      {
         "model_format":"pytorch",
         "model_size_in_billions":3,
         "quantization": "none",
         "model_id":"example/TestModel"
      }
   ],
   "chat_template": "xyz",
   "stop_token_ids": [1, 2, 3],
   "stop": ["hello", "world"]
}"""
    model_family = LLMFamilyV2.parse_raw(serialized)
    assert isinstance(model_family, LLMFamilyV2)
    assert model_family.version == 2
    assert model_family.context_length == 2048
    assert model_family.model_name == "TestModel"
    assert model_family.model_lang == ["en"]
    assert model_family.model_ability == ["embed", "generate"]
    assert len(model_family.model_specs) == 3

    gguf_spec = model_family.model_specs[0]
    assert gguf_spec.model_format == "ggufv2"
    assert gguf_spec.model_size_in_billions == 2
    assert gguf_spec.model_id == "example/TestModel"
    assert gguf_spec.model_hub == "huggingface"
    assert gguf_spec.model_file_name_template == "TestModel.{quantization}.bin"
    assert (
        gguf_spec.model_file_name_split_template
        == "TestModel.{quantization}.bin.{part}"
    )
    assert gguf_spec.quantization_parts["q4_2"][0] == "a"
    assert gguf_spec.quantization_parts["q4_2"][1] == "b"

    pytorch_spec = model_family.model_specs[-1]
    assert pytorch_spec.model_format == "pytorch"
    assert pytorch_spec.model_size_in_billions == 3
    assert pytorch_spec.model_hub == "huggingface"
    assert pytorch_spec.model_id == "example/TestModel"

    assert model_family.chat_template == "xyz"
    assert model_family.stop_token_ids == [1, 2, 3]
    assert model_family.stop == ["hello", "world"]


def test_cache_from_huggingface_pytorch():
    spec = PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=1,
        quantization="none",
        model_id="facebook/opt-125m",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    cache_dir = CacheManager(family).cache_from_huggingface()

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_cache_from_huggingface_gguf():
    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantization="q4_0",
        model_file_name_template="README.md",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    cache_manager = CacheManager(family)

    cache_dir = cache_manager.get_cache_dir()
    shutil.rmtree(cache_dir, ignore_errors=True)

    cache_dir = cache_manager.cache_from_huggingface()

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_cache_from_uri_local():
    with open("model.bin", "w") as fd:
        fd.write("foo")

    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions=3,
        model_id="TestModel",
        model_uri=os.path.abspath(os.getcwd()),
        quantization="",
        model_file_name_template="model.bin",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="test_cache_from_uri_local",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    cache_dir = CacheManager(family).cache()
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "model.bin"))
    os.remove(cache_dir)
    os.remove("model.bin")


def test_custom_llm():
    from ..custom import get_user_defined_llm_families, register_llm, unregister_llm

    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantization="",
        model_file_name_template="README.md",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="custom-qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    register_llm(family, False)

    assert family in get_user_defined_llm_families()

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()


def test_persistent_custom_llm():
    from ....constants import XINFERENCE_MODEL_DIR
    from ..custom import get_user_defined_llm_families, register_llm, unregister_llm

    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantization="",
        model_file_name_template="README.md",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    register_llm(family, True)

    assert family in get_user_defined_llm_families()
    assert f"{family.model_name}.json" in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "v2", "llm")
    )

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()
    assert f"{family.model_name}.json" not in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "v2", "llm")
    )


def test_is_locale_chinese_simplified():
    def zh_cn():
        return ("zh_CN", "UTF-8")

    def en_us():
        return ("en_US", "UTF-8")

    with patch("locale.getdefaultlocale", side_effect=zh_cn):
        assert is_locale_chinese_simplified()

    with patch("locale.getdefaultlocale", side_effect=en_us):
        assert not is_locale_chinese_simplified()


def test_match_llm():
    assert match_llm("fake") is None
    family = match_llm("qwen1.5-chat", model_format="ggufv2")
    assert family.model_name == "qwen1.5-chat"
    assert family.model_specs[0].quantization == "q2_k"

    family = match_llm("llama-2-chat", model_format="ggufv2", quantization="Q4_0")
    assert family.model_name == "llama-2-chat"
    assert family.model_specs[0].quantization == "Q4_0"

    family = match_llm("code-llama", model_format="ggufv2", quantization="q4_0")
    assert family.model_name == "code-llama"
    assert family.model_specs[0].quantization == "Q4_0"

    family = match_llm("code-llama")
    assert family.model_name == "code-llama"
    assert family.model_specs[0].model_format == "pytorch"

    try:
        os.environ[XINFERENCE_ENV_MODEL_SRC] = "modelscope"
        family = match_llm("llama-2-chat", model_format="ggufv2")
        assert family.model_name == "llama-2-chat"
        assert family.model_specs[0].model_hub == "modelscope"
        assert family.model_specs[0].quantization == "Q4_K_M"
        assert family.model_specs[0].model_format == "ggufv2"
        # pytorch model
        family = match_llm("baichuan-2-chat", model_format="pytorch")
        assert family.model_name == "baichuan-2-chat"
        assert family.model_specs[0].model_hub == "modelscope"
        assert family.model_specs[0].quantization == "none"
        assert family.model_specs[0].model_format == "pytorch"
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)


def test_is_valid_file_uri():
    with tempfile.NamedTemporaryFile() as tmp_file:
        assert is_valid_model_uri(f"file://{tmp_file.name}") is True
    assert is_valid_model_uri(f"file://{tmp_file.name}") is False


def test_get_cache_status_pytorch():
    spec = PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=1,
        quantization="none",
        model_id="facebook/opt-125m",
        model_revision="3d2b5f275bdf882b8775f902e1bfdb790e2cfc32",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    cache_manager = CacheManager(family)

    cache_status = cache_manager.get_cache_status()
    assert not isinstance(cache_status, list)
    assert not cache_status

    cache_dir = cache_manager.cache_from_huggingface()
    cache_status = cache_manager.get_cache_status()
    assert not isinstance(cache_status, list)
    assert cache_status

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_get_cache_status_gguf():
    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantization="q4_0",
        model_file_name_template="README.md",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    cache_manager = CacheManager(family)

    cache_status = cache_manager.get_cache_status()
    assert not cache_status

    cache_dir = cache_manager.cache_from_huggingface()
    cache_status = cache_manager.get_cache_status()
    assert cache_status

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_parse_chat_template():
    from ..llm_family import BUILTIN_LLM_PROMPT_STYLE

    assert len(BUILTIN_LLM_PROMPT_STYLE) > 0
    # take some examples to assert
    assert "qwen-chat" in BUILTIN_LLM_PROMPT_STYLE
    assert "glm4-chat" in BUILTIN_LLM_PROMPT_STYLE
    assert "baichuan-2-chat" in BUILTIN_LLM_PROMPT_STYLE

    hf_spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantization="q4_0",
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )
    ms_spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantization="q4_0",
        model_id="example/TestModel",
        model_hub="modelscope",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )

    llm_family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="glm4-chat",
        chat_template="glm4-chat",
    )
    model_spec = CustomLLMFamilyV2.parse_raw(bytes(llm_family.json(), "utf8"))
    assert model_spec.model_name == llm_family.model_name

    # test vision
    llm_family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="qwen2-vl-instruct",
        chat_template="qwen2-vl-instruct",
    )
    model_spec = CustomLLMFamilyV2.parse_raw(bytes(llm_family.json(), "utf-8"))
    assert "vision" in model_spec.model_ability

    # error: missing model_family
    llm_family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        chat_template="glm4-chat",
    )
    with pytest.raises(ValueError):
        CustomLLMFamilyV2.parse_raw(bytes(llm_family.json(), "utf8"))

    # successful new model family
    llm_family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_family="xyzz",
        model_specs=[hf_spec, ms_spec],
        chat_template="glm4-chat",
    )
    model_spec = CustomLLMFamilyV2.parse_raw(bytes(llm_family.json(), "utf8"))
    assert (
        model_spec.chat_template
        == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["chat_template"]
    )
    assert (
        model_spec.stop_token_ids
        == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["stop_token_ids"]
    )
    assert model_spec.stop == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["stop"]

    # when chat_template is None, chat_template = model_family
    llm_family = CustomLLMFamilyV2(
        version=2,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="glm4-chat",
        chat_template=None,
    )
    model_spec = CustomLLMFamilyV2.parse_raw(bytes(llm_family.json(), "utf8"))
    assert (
        model_spec.chat_template
        == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["chat_template"]
    )
    assert (
        model_spec.stop_token_ids
        == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["stop_token_ids"]
    )
    assert model_spec.stop == BUILTIN_LLM_PROMPT_STYLE["glm4-chat"]["stop"]


def test_match_model_size():
    assert match_model_size("1", "1")
    assert match_model_size("1", 1)
    assert match_model_size(1, 1)
    assert not match_model_size("1", "b")
    assert not match_model_size("1", "1b")
    assert match_model_size("1.8", "1_8")
    assert match_model_size("1_8", "1.8")
    assert not match_model_size("1", "1_8")
    assert not match_model_size("1__8", "1_8")
    assert not match_model_size("1_8", 18)
    assert not match_model_size("1_8", "18")
    assert not match_model_size("1.8", 18)
    assert not match_model_size("1.8", 1)
    assert match_model_size("001", 1)


def test_convert_model_size_to_float():
    assert convert_model_size_to_float("1_8") == 1.8
    assert convert_model_size_to_float("1.8") == 1.8
    assert convert_model_size_to_float(7) == float(7)
    assert convert_model_size_to_float(1.8) == 1.8


@pytest.mark.skipif(
    True,
    reason="Current system does not support vLLM",
)
def test_quert_engine_vLLM():
    from ..llm_family import LLM_ENGINES, check_engine_by_spec_parameters

    model_name = "qwen1.5-chat"
    assert model_name in LLM_ENGINES

    assert (
        "vLLM" in LLM_ENGINES[model_name] and len(LLM_ENGINES[model_name]["vLLM"]) == 21
    )

    assert check_engine_by_spec_parameters(
        model_engine="vLLM",
        model_name=model_name,
        model_format="gptq",
        model_size_in_billions="1_8",
        quantization="Int4",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="vLLM",
            model_name=model_name,
            model_format="gptq",
            model_size_in_billions="1_8",
            quantization="Int8",
        )
        is None
    )
    assert check_engine_by_spec_parameters(
        model_engine="vLLM",
        model_name=model_name,
        model_format="pytorch",
        model_size_in_billions="1_8",
        quantization="none",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="vLLM",
            model_name=model_name,
            model_format="pytorch",
            model_size_in_billions="1_8",
            quantization="4-bit",
        )
        is None
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="vLLM",
            model_name=model_name,
            model_format="ggufv2",
            model_size_in_billions="1_8",
            quantization="q2_k",
        )
        is None
    )


@pytest.mark.skipif(
    True,
    reason="Current system does not support SGLang",
)
def test_quert_engine_SGLang():
    from ..llm_family import LLM_ENGINES, check_engine_by_spec_parameters

    model_name = "qwen1.5-chat"
    assert model_name in LLM_ENGINES

    assert (
        "SGLang" in LLM_ENGINES[model_name]
        and len(LLM_ENGINES[model_name]["SGLang"]) == 21
    )

    assert check_engine_by_spec_parameters(
        model_engine="SGLang",
        model_name=model_name,
        model_format="gptq",
        model_size_in_billions="1_8",
        quantization="Int4",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="SGLang",
            model_name=model_name,
            model_format="gptq",
            model_size_in_billions="1_8",
            quantization="Int8",
        )
        is None
    )
    assert check_engine_by_spec_parameters(
        model_engine="SGLang",
        model_name=model_name,
        model_format="pytorch",
        model_size_in_billions="1_8",
        quantization="none",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="SGLang",
            model_name=model_name,
            model_format="pytorch",
            model_size_in_billions="1_8",
            quantization="4-bit",
        )
        is None
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="SGLang",
            model_name=model_name,
            model_format="ggufv2",
            model_size_in_billions="1_8",
            quantization="q2_k",
        )
        is None
    )


def test_query_engine_general():
    from ..custom import get_user_defined_llm_families, register_llm, unregister_llm
    from ..llama_cpp.core import XllamaCppModel
    from ..llm_family import LLM_ENGINES, check_engine_by_spec_parameters

    model_name = "qwen1.5-chat"
    assert model_name in LLM_ENGINES

    assert "Transformers" in LLM_ENGINES[model_name]
    assert "llama.cpp" in LLM_ENGINES[model_name]

    assert check_engine_by_spec_parameters(
        model_engine="transformers",
        model_name=model_name,
        model_format="gptq",
        model_size_in_billions="1_8",
        quantization="Int4",
    )
    assert check_engine_by_spec_parameters(
        model_engine="transformers",
        model_name=model_name,
        model_format="gptq",
        model_size_in_billions="1_8",
        quantization="Int8",
    )
    assert check_engine_by_spec_parameters(
        model_engine="transformers",
        model_name=model_name,
        model_format="pytorch",
        model_size_in_billions="1_8",
        quantization="none",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="llama.cpp",
            model_name=model_name,
            model_format="ggufv2",
            model_size_in_billions="1_8",
            quantization="q2_k",
        )
        is XllamaCppModel
    )
    with pytest.raises(ValueError) as exif:
        check_engine_by_spec_parameters(
            model_engine="llama.cpp",
            model_name=model_name,
            model_format="ggufv2",
            model_size_in_billions="2_2",
            quantization="q2_k",
        )
    assert (
        str(exif.value)
        == "Model qwen1.5-chat cannot be run on engine llama.cpp, with format ggufv2, size 2_2 and quantization q2_k."
    )

    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantization="",
        model_file_name_template="README.md",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    register_llm(family, False)

    assert family in get_user_defined_llm_families()
    assert "custom_model" in LLM_ENGINES and "llama.cpp" in LLM_ENGINES["custom_model"]
    assert check_engine_by_spec_parameters(
        model_engine="llama.cpp",
        model_name="custom_model",
        model_format="ggufv2",
        model_size_in_billions="0_5",
        quantization="",
    )

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()
    assert "custom_model" not in LLM_ENGINES

    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions="1_8",
        model_id="null",
        quantization="default",
        model_file_name_template="qwen1_5-1_8b-chat-q4_0.gguf",
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="custom-qwen1.5-chat",
        model_lang=["en", "zh"],
        model_ability=["generate", "chat"],
        model_specs=[spec],
        chat_template="test",
        stop=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        stop_token_ids=[151643, 151644, 151645],
    )

    register_llm(family, False)

    assert family in get_user_defined_llm_families()
    assert "custom-qwen1.5-chat" in LLM_ENGINES and ["llama.cpp"] == list(
        LLM_ENGINES["custom-qwen1.5-chat"].keys()
    )

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()
    assert "custom-qwen1.5-chat" not in LLM_ENGINES


def test_multimodal_engine_available_with_virtualenv(monkeypatch):
    """Regression test for qwen3.5-style multimodal models.

    A model whose only chat/vision architecture is served by the vLLM/SGLang
    multimodal engines must still list those engines when virtualenv is enabled
    on a supported worker (GPU + Linux) even if the libraries are not installed
    locally. Previously the vision match_json returned "library is not
    installed" unconditionally, unlike the chat match_json which honored the
    virtualenv exemption.

    virtualenv can install Python packages on demand but cannot add a GPU or
    change the OS, so the hardware/OS checks stay unconditional and are
    simulated here to isolate the library-exemption behavior.
    """
    import xinference.model.llm as llm_pkg
    from xinference.model.llm.sglang import core as sglang_core
    from xinference.model.llm.vllm import core as vllm_core

    from ...utils import get_engine_params_by_name_with_virtual_env

    llm_pkg._install()

    # Simulate a supported worker: CUDA present and Linux. This isolates the
    # library-missing exemption from the (unconditional) hardware/OS gates.
    for core in (vllm_core, sglang_core):
        for model_cls in vars(core).values():
            if isinstance(model_cls, type) and hasattr(model_cls, "_has_cuda_device"):
                monkeypatch.setattr(
                    model_cls, "_has_cuda_device", classmethod(lambda cls: True)
                )
                monkeypatch.setattr(
                    model_cls, "_is_linux", classmethod(lambda cls: True)
                )

    params = get_engine_params_by_name_with_virtual_env(
        "LLM", "qwen3.5", enable_virtual_env=True
    )
    assert params is not None

    # vLLM/SGLang (GPU multimodal engines), Transformers and llama.cpp are all
    # declared virtualenv markers for qwen3.5, so on a GPU+Linux worker they
    # must be offered (as available lists, flagged virtualenv_required when the
    # lib is missing) even without the library installed locally.
    for engine in ("vLLM", "SGLang", "Transformers", "llama.cpp"):
        assert isinstance(
            params.get(engine), list
        ), f"{engine} should be available for qwen3.5 under virtualenv, got {params.get(engine)!r}"

    # MLX and LMDEPLOY are not declared for qwen3.5, so they must NOT be listed
    # as available (kept as string reasons instead).
    for engine in ("MLX", "LMDEPLOY"):
        assert not isinstance(
            params.get(engine), list
        ), f"{engine} should not be available for qwen3.5, got {params.get(engine)!r}"


def test_multimodal_vllm_engine_requires_gpu_even_with_virtualenv():
    """virtualenv must not bypass the hardware/OS gate.

    On a worker without a supported GPU, the vLLM multimodal engine must not
    match even when virtualenv is enabled, because virtualenv cannot provide an
    accelerator. Guards the correctness fix where GPU/OS checks were previously
    (incorrectly) exempted under virtualenv.
    """
    import xinference.model.llm as llm_pkg
    from xinference.model.llm.llm_family import BUILTIN_LLM_FAMILIES
    from xinference.model.llm.vllm.core import VLLMMultiModel

    llm_pkg._install()
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen3.5")
    spec = family.model_specs[0]

    if (
        VLLMMultiModel._has_cuda_device()
        or VLLMMultiModel._has_mlu_device()
        or VLLMMultiModel._has_vacc_device()
        or VLLMMultiModel._has_musa_device()
    ):
        pytest.skip("host has an accelerator; cannot assert the no-GPU gate")

    result = VLLMMultiModel.match_json(family, spec, spec.quantization)
    assert result is not True
    assert isinstance(result, tuple) and result[0] is False
