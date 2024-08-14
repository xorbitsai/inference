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
import codecs
import json
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from ....constants import XINFERENCE_ENV_MODEL_SRC
from ...utils import is_locale_chinese_simplified, valid_model_revision
from ..llm_family import (
    CustomLLMFamilyV1,
    LlamaCppLLMSpecV1,
    LLMFamilyV1,
    PromptStyleV1,
    PytorchLLMSpecV1,
    _generate_meta_file,
    _get_cache_dir,
    _get_meta_path,
    _skip_download,
    convert_model_size_to_float,
    is_valid_model_uri,
    match_llm,
    match_model_size,
    parse_uri,
)


def test_deserialize_llm_family_v1():
    serialized = """{
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
         "model_format":"ggufv2",
         "model_size_in_billions":2,
         "quantizations": ["q4_0", "q4_1"],
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
    assert model_family.context_length == 2048
    assert model_family.model_name == "TestModel"
    assert model_family.model_lang == ["en"]
    assert model_family.model_ability == ["embed", "generate"]
    assert len(model_family.model_specs) == 2

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

    pytorch_spec = model_family.model_specs[1]
    assert pytorch_spec.model_format == "pytorch"
    assert pytorch_spec.model_size_in_billions == 3
    assert pytorch_spec.model_hub == "huggingface"
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
    gguf_spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        quantization_parts={"q4_2": ["a", "b"]},
        model_id="example/TestModel",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
        model_file_name_split_template="TestModel.{quantization}.bin.{part}",
    )
    pytorch_spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=3,
        quantizations=["int8", "int4", "none"],
        model_id="example/TestModel",
        model_revision="456",
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
    llm_family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="TestModel",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[gguf_spec, pytorch_spec],
        prompt_style=prompt_style,
    )

    expected = """{"version": 1, "context_length": 2048, "model_name": "TestModel", "model_lang": ["en"], "model_ability": ["embed", "generate"], "model_description": null, "model_family": null, "model_specs": [{"model_format": "ggufv2", "model_hub": "huggingface", "model_size_in_billions": 2, "quantizations": ["q4_0", "q4_1"], "quantization_parts": {"q4_2": ["a", "b"]}, "model_id": "example/TestModel", "model_revision": "123", "model_file_name_template": "TestModel.{quantization}.bin", "model_file_name_split_template": "TestModel.{quantization}.bin.{part}", "model_uri": null}, {"model_format": "pytorch", "model_hub": "huggingface", "model_size_in_billions": 3, "quantizations": ["int8", "int4", "none"], "model_id": "example/TestModel", "model_revision": "456", "model_uri": null}], "prompt_style": {"style_name": "ADD_COLON_SINGLE", "system_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", "roles": ["user", "assistant"], "intra_message_sep": "\\n### ", "inter_message_sep": "\\n### ", "stop": null, "stop_token_ids": null}}"""
    assert json.loads(llm_family.json()) == json.loads(expected)

    llm_family_context_length = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="TestModel",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[gguf_spec, pytorch_spec],
        prompt_style=prompt_style,
    )

    assert json.loads(llm_family_context_length.json()) == json.loads(expected)


def test_builtin_llm_families():
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "llm_family.json"
    )
    for json_obj in json.load(codecs.open(json_path, "r", encoding="utf-8")):
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
        context_length=2048,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_huggingface(family, spec, quantization=None)

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_cache_from_huggingface_gguf():
    from ..llm_family import cache_from_huggingface

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=["q4_0"],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = _get_cache_dir(family, spec)
    shutil.rmtree(cache_dir)

    cache_dir = cache_from_huggingface(family, spec, quantization="q4_0")

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_cache_from_uri_local():
    from ..llm_family import cache_from_uri

    with open("model.bin", "w") as fd:
        fd.write("foo")

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=3,
        model_id="TestModel",
        model_uri=os.path.abspath(os.getcwd()),
        quantizations=[""],
        model_file_name_template="model.bin",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="test_cache_from_uri_local",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_uri(family, spec)
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "model.bin"))
    os.remove(cache_dir)
    os.remove("model.bin")


def test_meta_file():
    from ..llm_family import cache_from_huggingface

    spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=1,
        quantizations=["4-bit", "8-bit", "none"],
        model_id="facebook/opt-125m",
        model_revision="3d2b5f275bdf882b8775f902e1bfdb790e2cfc32",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_huggingface(family, spec, quantization=None)
    meta_path = _get_meta_path(cache_dir, spec.model_format, spec.model_hub, None)
    assert valid_model_revision(meta_path, "3d2b5f275bdf882b8775f902e1bfdb790e2cfc32")
    shutil.rmtree(cache_dir)


def test_parse_uri():
    scheme, path = parse_uri("dir")
    assert scheme == "file"
    assert path == "dir"

    scheme, path = parse_uri("dir/file")
    assert scheme == "file"
    assert path == "dir/file"

    scheme, path = parse_uri("s3://bucket")
    assert scheme == "s3"
    assert path == "bucket"

    scheme, path = parse_uri("s3://bucket/dir")
    assert scheme == "s3"
    assert path == "bucket/dir"


def test_legacy_cache():
    from ..llm_family import cache, get_legacy_cache_path

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=["test_legacy_cache"],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_path = get_legacy_cache_path(
        family.model_name,
        spec.model_format,
        spec.model_size_in_billions,
        quantization="test_legacy_cache",
    )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fd:
        fd.write("foo")

    assert cache(
        llm_family=family, llm_spec=spec, quantization="test_legacy_cache"
    ) == os.path.dirname(cache_path)
    shutil.rmtree(os.path.dirname(cache_path), ignore_errors=True)


def test_custom_llm():
    from ..llm_family import get_user_defined_llm_families, register_llm, unregister_llm

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom-qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    register_llm(family, False)

    assert family in get_user_defined_llm_families()

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()


def test_persistent_custom_llm():
    from ....constants import XINFERENCE_MODEL_DIR
    from ..llm_family import get_user_defined_llm_families, register_llm, unregister_llm

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    register_llm(family, True)

    assert family in get_user_defined_llm_families()
    assert f"{family.model_name}.json" in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "llm")
    )

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()
    assert f"{family.model_name}.json" not in os.listdir(
        os.path.join(XINFERENCE_MODEL_DIR, "llm")
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
    family, spec, q = match_llm("qwen1.5-chat", model_format="ggufv2")
    assert family.model_name == "qwen1.5-chat"
    assert q == "q2_k"

    family, spec, q = match_llm(
        "llama-2-chat", model_format="ggufv2", quantization="Q4_0"
    )
    assert family.model_name == "llama-2-chat"
    assert q == "Q4_0"

    family, spec, q = match_llm(
        "code-llama", model_format="ggufv2", quantization="q4_0"
    )
    assert family.model_name == "code-llama"
    assert q == "Q4_0"

    family, spec, q = match_llm("code-llama")
    assert family.model_name == "code-llama"
    assert spec.model_format == "pytorch"

    try:
        os.environ[XINFERENCE_ENV_MODEL_SRC] = "modelscope"
        family, spec, q = match_llm("llama-2-chat")
        assert family.model_name == "llama-2-chat"
        assert spec.model_hub == "modelscope"
        assert q == "Q4_K_M"
        assert spec.model_format == "ggufv2"
        # pytorch model
        family, spec, q = match_llm("baichuan-2-chat")
        assert family.model_name == "baichuan-2-chat"
        assert spec.model_hub == "modelscope"
        assert q == "none"
        assert spec.model_format == "pytorch"
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)


def test_is_valid_file_uri():
    with tempfile.NamedTemporaryFile() as tmp_file:
        assert is_valid_model_uri(f"file://{tmp_file.name}") is True
    assert is_valid_model_uri(f"file://{tmp_file.name}") is False


def test_skip_download_pytorch():
    hf_spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=3,
        quantizations=["int8", "int4", "none"],
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="456",
    )
    ms_spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=3,
        quantizations=["int8", "int4", "none"],
        model_id="example/TestModel",
        model_hub="modelscope",
        model_revision="456",
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
    llm_family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_skip_download_pytorch",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[hf_spec, ms_spec],
        prompt_style=prompt_style,
    )

    cache_dir = _get_cache_dir(llm_family, hf_spec)

    hf_meta_path = _get_meta_path(
        cache_dir, hf_spec.model_format, hf_spec.model_hub, quantization=None
    )
    ms_meta_path = _get_meta_path(
        cache_dir, ms_spec.model_format, ms_spec.model_hub, quantization=None
    )

    # since huggingface meta file exists, skip for both.
    _generate_meta_file(hf_meta_path, llm_family, hf_spec, quantization=None)
    assert os.path.exists(hf_meta_path)
    try:
        assert _skip_download(
            cache_dir,
            hf_spec.model_format,
            hf_spec.model_hub,
            hf_spec.model_revision,
            quantization=None,
        )
        assert _skip_download(
            cache_dir,
            ms_spec.model_format,
            ms_spec.model_hub,
            ms_spec.model_revision,
            quantization=None,
        )
    finally:
        os.remove(hf_meta_path)
        assert not os.path.exists(hf_meta_path)

    # since modelscope meta file exists, skip for both.
    _generate_meta_file(ms_meta_path, llm_family, ms_spec, quantization=None)
    assert os.path.exists(ms_meta_path)
    try:
        assert _skip_download(
            cache_dir,
            hf_spec.model_format,
            hf_spec.model_hub,
            hf_spec.model_revision,
            quantization=None,
        )
        assert _skip_download(
            cache_dir,
            ms_spec.model_format,
            ms_spec.model_hub,
            ms_spec.model_revision,
            quantization=None,
        )
    finally:
        os.remove(ms_meta_path)
        assert not os.path.exists(ms_meta_path)


def test_skip_download_gguf():
    hf_spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )
    ms_spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="modelscope",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
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
    llm_family = LLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_skip_download_ggml",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[hf_spec, ms_spec],
        prompt_style=prompt_style,
    )

    cache_dir = _get_cache_dir(llm_family, hf_spec)

    hf_meta_path = _get_meta_path(
        cache_dir, hf_spec.model_format, hf_spec.model_hub, quantization="q4_0"
    )
    ms_meta_path = _get_meta_path(
        cache_dir, ms_spec.model_format, ms_spec.model_hub, quantization="q4_0"
    )

    # since huggingface meta file exists, only skip when model hub is huggingface.
    _generate_meta_file(hf_meta_path, llm_family, hf_spec, quantization="q4_0")
    assert os.path.exists(hf_meta_path)
    try:
        assert _skip_download(
            cache_dir,
            hf_spec.model_format,
            hf_spec.model_hub,
            hf_spec.model_revision,
            quantization="q4_0",
        )
        assert not _skip_download(
            cache_dir,
            ms_spec.model_format,
            ms_spec.model_hub,
            ms_spec.model_revision,
            quantization="q4_0",
        )
    finally:
        os.remove(hf_meta_path)
        assert not os.path.exists(hf_meta_path)

    # since modelscope meta file exists, only skip when model hub is modelscope.
    _generate_meta_file(ms_meta_path, llm_family, ms_spec, quantization="q4_0")
    assert os.path.exists(ms_meta_path)
    try:
        assert not _skip_download(
            cache_dir,
            hf_spec.model_format,
            hf_spec.model_hub,
            hf_spec.model_revision,
            quantization="q4_0",
        )
        assert _skip_download(
            cache_dir,
            ms_spec.model_format,
            ms_spec.model_hub,
            ms_spec.model_revision,
            quantization="q4_0",
        )
    finally:
        os.remove(ms_meta_path)
        assert not os.path.exists(ms_meta_path)


def test_get_cache_status_pytorch():
    from ..llm_family import cache_from_huggingface, get_cache_status

    spec = PytorchLLMSpecV1(
        model_format="pytorch",
        model_size_in_billions=1,
        quantizations=["4-bit", "8-bit", "none"],
        model_id="facebook/opt-125m",
        model_revision="3d2b5f275bdf882b8775f902e1bfdb790e2cfc32",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="opt",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_status = get_cache_status(llm_family=family, llm_spec=spec)
    assert not isinstance(cache_status, list)
    assert not cache_status

    cache_dir = cache_from_huggingface(family, spec, quantization=None)
    cache_status = get_cache_status(llm_family=family, llm_spec=spec)
    assert not isinstance(cache_status, list)
    assert cache_status

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_get_cache_status_gguf():
    from ..llm_family import cache_from_huggingface, get_cache_status

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=["q4_0", "q5_0"],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="qwen1.5-chat",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_status = get_cache_status(llm_family=family, llm_spec=spec)
    assert isinstance(cache_status, list)
    assert not any(cache_status)

    cache_dir = cache_from_huggingface(family, spec, quantization="q4_0")
    cache_status = get_cache_status(llm_family=family, llm_spec=spec)
    assert isinstance(cache_status, list)
    assert len(cache_status) == 2
    assert cache_status[0] and not cache_status[1]

    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    assert os.path.islink(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir)


def test_parse_prompt_style():
    from ..llm_family import BUILTIN_LLM_PROMPT_STYLE

    assert len(BUILTIN_LLM_PROMPT_STYLE) > 0
    # take some examples to assert
    assert "qwen-chat" in BUILTIN_LLM_PROMPT_STYLE
    assert "chatglm3" in BUILTIN_LLM_PROMPT_STYLE
    assert "baichuan-chat" in BUILTIN_LLM_PROMPT_STYLE

    hf_spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )
    ms_spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="modelscope",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.bin",
    )

    llm_family = CustomLLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="chatglm3",
        prompt_style="chatglm3",
    )
    model_spec = CustomLLMFamilyV1.parse_raw(bytes(llm_family.json(), "utf8"))
    assert model_spec.model_name == llm_family.model_name

    # test vision
    llm_family = CustomLLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="qwen-vl-chat",
        prompt_style="qwen-vl-chat",
    )
    model_spec = CustomLLMFamilyV1.parse_raw(bytes(llm_family.json(), "utf-8"))
    assert "vision" in model_spec.model_ability

    # error: missing model_family
    llm_family = CustomLLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        prompt_style="chatglm3",
    )
    with pytest.raises(ValueError):
        CustomLLMFamilyV1.parse_raw(bytes(llm_family.json(), "utf8"))

    # wrong model_family
    llm_family = CustomLLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_family="xyzz",
        model_specs=[hf_spec, ms_spec],
        prompt_style="chatglm3",
    )
    with pytest.raises(ValueError):
        CustomLLMFamilyV1.parse_raw(bytes(llm_family.json(), "utf8"))

    # error: wrong prompt style
    llm_family = CustomLLMFamilyV1(
        version=1,
        model_type="LLM",
        model_name="test_LLM",
        model_lang=["en"],
        model_ability=["chat", "generate"],
        model_specs=[hf_spec, ms_spec],
        model_family="chatglm3",
        prompt_style="test_xyz",
    )
    with pytest.raises(ValueError):
        CustomLLMFamilyV1.parse_raw(bytes(llm_family.json(), "utf8"))


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
    from ..llama_cpp.core import LlamaCppChatModel
    from ..llm_family import (
        LLM_ENGINES,
        check_engine_by_spec_parameters,
        get_user_defined_llm_families,
        register_llm,
        unregister_llm,
    )

    assert check_engine_by_spec_parameters(
        model_engine="transformers",
        model_name="aquila2",
        model_format="pytorch",
        model_size_in_billions=7,
        quantization="none",
    )

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
    assert check_engine_by_spec_parameters(
        model_engine="transformers",
        model_name=model_name,
        model_format="pytorch",
        model_size_in_billions="1_8",
        quantization="4-bit",
    )
    assert (
        check_engine_by_spec_parameters(
            model_engine="llama.cpp",
            model_name=model_name,
            model_format="ggufv2",
            model_size_in_billions="1_8",
            quantization="q2_k",
        )
        is LlamaCppChatModel
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

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="0_5",
        model_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        prompt_style=None,
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

    spec = LlamaCppLLMSpecV1(
        model_format="ggufv2",
        model_size_in_billions="1_8",
        model_id="null",
        quantizations=["default"],
        model_file_name_template="qwen1_5-1_8b-chat-q4_0.gguf",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom-qwen1.5-chat",
        model_lang=["en", "zh"],
        model_ability=["generate", "chat"],
        model_specs=[spec],
        prompt_style={
            "style_name": "QWEN",
            "system_prompt": "You are a helpful assistant.",
            "roles": ["user", "assistant"],
            "intra_message_sep": "\n",
            "inter_message_sep": "",
            "stop": ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            "stop_token_ids": [151643, 151644, 151645],
        },
    )

    register_llm(family, False)

    assert family in get_user_defined_llm_families()
    assert "custom-qwen1.5-chat" in LLM_ENGINES and ["llama.cpp"] == list(
        LLM_ENGINES["custom-qwen1.5-chat"].keys()
    )

    unregister_llm(family.model_name)
    assert family not in get_user_defined_llm_families()
    assert "custom-qwen1.5-chat" not in LLM_ENGINES
