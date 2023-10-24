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
import os
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from ....constants import XINFERENCE_ENV_MODEL_SRC
from ...utils import is_locale_chinese_simplified
from ..llm_family import (
    AWSRegion,
    GgmlLLMSpecV1,
    LLMFamilyV1,
    PromptStyleV1,
    PytorchLLMSpecV1,
    _generate_meta_file,
    _get_cache_dir,
    _get_meta_path,
    _skip_download,
    is_self_hosted,
    is_valid_model_uri,
    match_llm,
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
    assert model_family.context_length == 2048
    assert model_family.model_name == "TestModel"
    assert model_family.model_lang == ["en"]
    assert model_family.model_ability == ["embed", "generate"]
    assert len(model_family.model_specs) == 2

    ggml_spec = model_family.model_specs[0]
    assert ggml_spec.model_format == "ggmlv3"
    assert ggml_spec.model_size_in_billions == 2
    assert ggml_spec.model_id == "example/TestModel"
    assert ggml_spec.model_hub == "huggingface"
    assert ggml_spec.model_file_name_template == "TestModel.{quantization}.ggmlv3.bin"

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
    ggml_spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
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
        model_specs=[ggml_spec, pytorch_spec],
        prompt_style=prompt_style,
    )

    expected = """{"version": 1, "context_length": 2048, "model_name": "TestModel", "model_lang": ["en"], "model_ability": ["embed", "generate"], "model_description": null, "model_specs": [{"model_format": "ggmlv3", "model_hub": "huggingface", "model_size_in_billions": 2, "quantizations": ["q4_0", "q4_1"], "model_id": "example/TestModel", "model_revision": "123", "model_file_name_template": "TestModel.{quantization}.ggmlv3.bin", "model_uri": null}, {"model_format": "pytorch", "model_hub": "huggingface", "model_size_in_billions": 3, "quantizations": ["int8", "int4", "none"], "model_id": "example/TestModel", "model_revision": "456", "model_uri": null}], "prompt_style": {"style_name": "ADD_COLON_SINGLE", "system_prompt": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.", "roles": ["user", "assistant"], "intra_message_sep": "\\n### ", "inter_message_sep": "\\n### ", "stop": null, "stop_token_ids": null}}"""
    assert json.loads(llm_family.json()) == json.loads(expected)

    llm_family_context_length = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="TestModel",
        model_lang=["en"],
        model_ability=["embed", "generate"],
        model_specs=[ggml_spec, pytorch_spec],
        prompt_style=prompt_style,
    )

    assert json.loads(llm_family_context_length.json()) == json.loads(expected)


def test_builtin_llm_families():
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


def test_cache_from_huggingface_ggml():
    from ..llm_family import cache_from_huggingface

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=["q4_0"],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="orca",
        model_lang=["en"],
        model_ability=["embed", "chat"],
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

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
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


def test_cache_from_uri_remote():
    from ..llm_family import cache_from_uri

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TestModel",
        model_uri="s3://test_bucket",
        quantizations=[""],
        model_file_name_template="model.bin",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="test_cache_from_uri_remote",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    from unittest.mock import patch

    import fsspec

    fsspec.real_filesystem = fsspec.filesystem

    def fsspec_filesystem_side_effect(scheme: str, *args, **kwargs):
        if scheme == "s3":
            mock_fs = Mock()
            mock_fs.info.return_value = {"size": 3}
            mock_fs.walk.return_value = [("test_bucket", None, ["model.bin"])]
            mock_file = MagicMock()
            mock_file_descriptor = Mock()
            mock_file_descriptor.read.side_effect = ["foo".encode(), None]
            mock_file.__enter__.return_value = mock_file_descriptor
            mock_fs.open.return_value = mock_file
            return mock_fs
        else:
            return fsspec.real_filesystem(scheme)

    with patch("fsspec.filesystem", side_effect=fsspec_filesystem_side_effect):
        cache_dir = cache_from_uri(family, spec)
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "model.bin"))
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_from_uri_remote_exception_handling():
    from ....constants import XINFERENCE_CACHE_DIR
    from ..llm_family import cache_from_uri

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TestModel",
        model_uri="s3://test_bucket",
        quantizations=[""],
        model_file_name_template="model.bin",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="test_cache_from_uri_remote_exception_handling",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    from unittest.mock import patch

    import fsspec

    fsspec.real_filesystem = fsspec.filesystem

    def fsspec_filesystem_side_effect(scheme: str, *args, **kwargs):
        if scheme == "s3":
            mock_fs = Mock()
            mock_fs.info.return_value = {"size": 3}
            mock_fs.walk.return_value = [("test_bucket", None, ["model.bin"])]
            mock_file = MagicMock()
            mock_file_descriptor = Mock()
            mock_file_descriptor.read.side_effect = Exception("Mock exception")
            mock_file.__enter__.return_value = mock_file_descriptor
            mock_fs.open.return_value = mock_file
            return mock_fs
        else:
            return fsspec.real_filesystem(scheme)

    with pytest.raises(
        RuntimeError,
        match="Failed to download model 'test_cache_from_uri_remote_exception_handling'",
    ):
        with patch("fsspec.filesystem", side_effect=fsspec_filesystem_side_effect):
            cache_from_uri(family, spec)

    cache_dir_name = (
        f"{family.model_name}-{spec.model_format}" f"-{spec.model_size_in_billions}b"
    )
    cache_dir = os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, cache_dir_name))
    assert not os.path.exists(cache_dir)


def test_legacy_cache():
    from ..llm_family import cache, get_legacy_cache_path

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=["test_legacy_cache"],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="orca",
        model_lang=["en"],
        model_ability=["embed", "chat"],
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


@pytest.mark.skip(reason="Temporary disabled")
def test_cache_from_self_hosted_storage():
    from ..llm_family import cache_from_self_hosted_storage

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="orca",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        prompt_style=None,
    )

    cache_dir = cache_from_self_hosted_storage(family, spec, quantization="")
    assert os.path.exists(cache_dir)
    assert os.path.exists(os.path.join(cache_dir, "README.md"))
    shutil.rmtree(cache_dir, ignore_errors=True)


def test_custom_llm():
    from ..llm_family import get_user_defined_llm_families, register_llm, unregister_llm

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["embed", "chat"],
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

    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="custom_model",
        model_lang=["en"],
        model_ability=["embed", "chat"],
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


def test_download_from_self_hosted_storage():
    from ....constants import XINFERENCE_ENV_MODEL_SRC
    from ..llm_family import download_from_self_hosted_storage

    assert not download_from_self_hosted_storage()

    os.environ[XINFERENCE_ENV_MODEL_SRC] = "xorbits"
    assert download_from_self_hosted_storage()
    del os.environ[XINFERENCE_ENV_MODEL_SRC]


def test_aws_region_set():
    with AWSRegion("foo"):
        assert os.environ["AWS_DEFAULT_REGION"] == "foo"

    # Ensure the region is deleted if it wasn't set before
    assert "AWS_DEFAULT_REGION" not in os.environ


def test_aws_region_restore():
    # Set an initial region
    os.environ["AWS_DEFAULT_REGION"] = "us-west-1"

    with AWSRegion("foo"):
        assert os.environ["AWS_DEFAULT_REGION"] == "foo"

    # Ensure the region is restored to its original value after exiting the context
    assert os.environ["AWS_DEFAULT_REGION"] == "us-west-1"


def test_aws_region_no_restore_if_not_set():
    # Ensure AWS_DEFAULT_REGION is not set
    if "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]

    with AWSRegion("foo"):
        assert os.environ["AWS_DEFAULT_REGION"] == "foo"

    # Ensure the region is deleted if it wasn't set before
    assert "AWS_DEFAULT_REGION" not in os.environ


def test_aws_region_exception_handling():
    with pytest.raises(ValueError):
        with AWSRegion("foo"):
            raise ValueError("Test exception")

    # Ensure the region is deleted if it wasn't set before
    assert "AWS_DEFAULT_REGION" not in os.environ


def test_is_self_hosted():
    spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=3,
        model_id="TheBloke/orca_mini_3B-GGML",
        quantizations=[""],
        model_file_name_template="README.md",
    )
    family = LLMFamilyV1(
        version=1,
        context_length=2048,
        model_type="LLM",
        model_name="orca",
        model_lang=["en"],
        model_ability=["embed", "chat"],
        model_specs=[spec],
        prompt_style=None,
    )
    assert is_self_hosted(family, spec)

    family.model_name = "foo"
    assert not is_self_hosted(family, spec)


def test_match_llm():
    assert match_llm("fake") is None
    family, spec, q = match_llm("orca", model_format="ggmlv3")
    assert family.model_name == "orca"
    assert q == "q4_0"

    family, spec, q = match_llm(
        "llama-2-chat", model_format="ggmlv3", quantization="Q4_0"
    )
    assert family.model_name == "llama-2-chat"
    assert q == "q4_0"

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
        assert q == "4-bit"
        assert spec.model_format == "pytorch"
    finally:
        os.environ.pop(XINFERENCE_ENV_MODEL_SRC)


def test_match_llm_cls():
    # TODO: implement
    pass


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


def test_skip_download_ggml():
    hf_spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="huggingface",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
    )
    ms_spec = GgmlLLMSpecV1(
        model_format="ggmlv3",
        model_size_in_billions=2,
        quantizations=["q4_0", "q4_1"],
        model_id="example/TestModel",
        model_hub="modelscope",
        model_revision="123",
        model_file_name_template="TestModel.{quantization}.ggmlv3.bin",
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
