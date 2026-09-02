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
from packaging import version
from packaging.requirements import Requirement

from ....constants import XINFERENCE_ENV_MODEL_SRC
from ...utils import is_locale_chinese_simplified, is_valid_model_uri
from ..cache_manager import LLMCacheManager as CacheManager
from ..llm_family import (
    CustomLLMFamilyV2,
    LlamaCppLLMSpecV2,
    LLMFamilyV2,
    MLXLLMSpecV2,
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


def _mlx_family_with_drafter(draft_model_id=None, draft_quantizations=None):
    spec = MLXLLMSpecV2(
        model_format="mlx",
        model_size_in_billions=2,
        quantization="bf16",
        model_id="mlx-community/gemma-4-e2b-it-bf16",
        draft_model_id=draft_model_id,
        draft_quantizations=draft_quantizations,
    )
    return LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="gemma-4",
        model_lang=["en"],
        model_ability=["chat", "vision"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )


def test_parse_bool_launch_arg():
    from ..core import parse_bool_launch_arg

    assert parse_bool_launch_arg(True) is True
    assert parse_bool_launch_arg("true") is True
    assert parse_bool_launch_arg("True") is True
    assert parse_bool_launch_arg("1") is True
    # the Web UI submits strings, "false" must not turn the switch on
    assert parse_bool_launch_arg("false") is False
    assert parse_bool_launch_arg("") is False
    assert parse_bool_launch_arg(None) is False


def test_draft_model_cache_dir():
    family = _mlx_family_with_drafter("mlx-community/gemma-4-E2B-it-assistant-bf16")

    target_cache_dir = CacheManager(family).get_cache_dir()
    draft_cache_dir = CacheManager(family, use_draft_model=True).get_cache_dir()

    assert target_cache_dir.endswith("gemma-4-mlx-2b-bf16")
    # no declared quantizations, so the drafter has a single cache dir
    assert draft_cache_dir == f"{target_cache_dir}-draft"


def test_draft_model_quantization_selection():
    family = _mlx_family_with_drafter(
        "mlx-community/gemma-4-12B-it-assistant-{draft_quantization}",
        draft_quantizations=["bf16", "8bit", "4bit"],
    )
    target_cache_dir = CacheManager(family).get_cache_dir()

    # the first declared quantization is the default
    default_manager = CacheManager(family, use_draft_model=True)
    assert default_manager._model_id == "mlx-community/gemma-4-12B-it-assistant-bf16"
    assert default_manager.get_cache_dir() == f"{target_cache_dir}-draft-bf16"

    chosen = CacheManager(family, use_draft_model=True, draft_quantization="4bit")
    assert chosen._model_id == "mlx-community/gemma-4-12B-it-assistant-4bit"
    # drafters converted differently must not share a cache dir
    assert chosen.get_cache_dir() == f"{target_cache_dir}-draft-4bit"

    with pytest.raises(ValueError, match="is not available"):
        CacheManager(family, use_draft_model=True, draft_quantization="mxfp4")


def test_gguf_drafter_without_a_file_template_is_rejected():
    # A gguf download is per file. Without a template there is nothing to ask
    # for, and falling through to the target's file names would request a name
    # that does not exist in the drafter repository.
    spec = LlamaCppLLMSpecV2(
        model_format="ggufv2",
        model_size_in_billions=12,
        quantization="Q4_K_M",
        model_id="unsloth/gemma-4-12b-it-GGUF",
        model_file_name_template="gemma-4-12b-it-{quantization}.gguf",
        draft_model_id="some/standalone-drafter-GGUF",
        draft_quantizations=["BF16"],
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="gemma-4",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    with pytest.raises(ValueError, match="draft_model_file_name_template"):
        CacheManager(family, use_draft_model=True)


def test_gguf_drafter_downloads_only_its_own_file():
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")
    spec = next(
        s
        for s in family.model_specs
        if s.model_format == "ggufv2"
        and s.model_hub == "huggingface"
        and s.model_size_in_billions == 12
        and s.quantization == "Q4_K_M"
    )
    copy = family.copy()
    copy.model_specs = [spec]

    target_files, _, _ = CacheManager(copy)._gguf_file_names()
    draft_files, final, need_merge = CacheManager(
        copy, use_draft_model=True
    )._gguf_file_names()

    assert draft_files == ["MTP/mtp-gemma-4-12b-it-BF16.gguf"]
    assert final == draft_files[0]
    assert need_merge is False
    # never the target's own file, which does not exist under that name in a
    # drafter repository
    assert not set(draft_files) & set(target_files)


def test_malformed_drafter_spec_does_not_break_listing():
    # A custom spec whose drafter fields do not add up cannot be launched, but
    # it must not take the model-listing endpoint down with it.
    from ...core import VirtualEnvSettings  # noqa: F401

    spec = MLXLLMSpecV2(
        model_format="mlx",
        model_size_in_billions=2,
        quantization="bf16",
        model_id="mlx-community/gemma-4-e2b-it-bf16",
        # templated id with nothing to fill it from: the cache manager rejects it
        draft_model_id="some/drafter-{draft_quantization}",
        draft_quantizations=[],
    )
    family = LLMFamilyV2(
        version=2,
        context_length=2048,
        model_type="LLM",
        model_name="gemma-4",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
    )

    with pytest.raises(ValueError):
        CacheManager(family, use_draft_model=True)

    from ....core.worker import WorkerActor

    class _Worker:
        pass

    specs, hubs = WorkerActor._get_spec_dicts_with_cache_status(
        _Worker(), family, CacheManager
    )

    assert len(specs) == 1
    assert specs[0]["draft_cache_status"] == []


def test_both_engine_discovery_paths_publish_draft_support():
    # The virtualenv-aware path is the default one, and it rebuilds entries from
    # specs rather than from the engine registry; the flag has to reach both or
    # the Web UI hides speculative decoding in a default deployment.
    from ...utils import (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    )

    for get_params in (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    ):
        params = get_params("LLM", "gemma-4") or {}
        available = {
            engine: entries
            for engine, entries in params.items()
            if isinstance(entries, list) and entries
        }
        assert available, get_params.__name__
        for engine, entries in available.items():
            for entry in entries:
                assert "support_draft_model" in entry, (get_params.__name__, engine)
        if "Transformers" in available:
            # it runs its own batching loop, so it can never take a drafter
            assert not available["Transformers"][0]["support_draft_model"]


def test_gemma_4_virtualenv_declares_sglang():
    from ...utils import _collect_virtualenv_engine_markers
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")

    assert "sglang" in _collect_virtualenv_engine_markers(family)


def test_engine_discovery_paths_agree_on_draft_support():
    # The default path rebuilds entries from specs, so it resolves the class a
    # different way than the registry path. They have to reach the same answer,
    # or the Web UI offers speculative decoding for an engine that rejects it —
    # or hides it for one that would take it.
    from ...utils import (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    )

    def flags(params):
        return {
            (
                engine,
                entry["model_format"],
                str(entry["model_size_in_billions"]),
            ): entry["support_draft_model"]
            for engine, entries in (params or {}).items()
            if isinstance(entries, list)
            for entry in entries
        }

    for model_name in ("gemma-4", "qwen3"):
        strict = flags(get_engine_params_by_name("LLM", model_name))
        default = flags(get_engine_params_by_name_with_virtual_env("LLM", model_name))

        assert strict, model_name
        assert default, model_name
        shared = set(strict) & set(default)
        assert shared, model_name
        for key in shared:
            assert strict[key] == default[key], (model_name, key)


def test_num_speculative_tokens_validation():
    from ..core import parse_num_speculative_tokens

    assert parse_num_speculative_tokens(None) is None
    assert parse_num_speculative_tokens("") is None
    assert parse_num_speculative_tokens(4) == 4
    # the Web UI submits additional parameters as strings
    assert parse_num_speculative_tokens("6") == 6
    # a depth of zero contradicts speculative decoding being on; substituting a
    # default for it would silently ignore what the caller asked for
    for invalid in (0, "0", -1, "abc", 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            parse_num_speculative_tokens(invalid)


@pytest.mark.parametrize(
    "draft_model_id",
    [
        "some/drafter-{draft_quantization}-{quantization}",  # KeyError
        "some/drafter-{draft_quantization}-{}",  # IndexError
        # no `{draft_quantization}` at all: these used to be passed to the hub
        # verbatim and 404 at download time instead of failing here
        "some/drafter-{quantization}",
        "some/drafter-{}",
    ],
)
def test_unexpected_placeholder_is_a_value_error(draft_model_id):
    # str.format raises KeyError or IndexError for these, which the listing
    # path's ValueError guard would not catch.
    family = _mlx_family_with_drafter(draft_model_id, draft_quantizations=["bf16"])

    with pytest.raises(ValueError, match="placeholder other than"):
        CacheManager(family, use_draft_model=True)

    from ....core.worker import WorkerActor

    class _Worker:
        pass

    specs, _hubs = WorkerActor._get_spec_dicts_with_cache_status(
        _Worker(), family, CacheManager
    )

    assert specs[0]["draft_cache_status"] == []


def test_draft_model_not_declared():
    family = _mlx_family_with_drafter()

    with pytest.raises(ValueError, match="does not declare a drafter model"):
        CacheManager(family, use_draft_model=True)


def test_builtin_gemma_4_mlx_specs_declare_drafter():
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")
    mlx_specs = [s for s in family.model_specs if s.model_format == "mlx"]

    assert mlx_specs
    for spec in mlx_specs:
        assert spec.draft_model_id, spec.model_id
        assert "assistant" in spec.draft_model_id
        assert spec.draft_quantizations
        # `{quantization}` would be substituted with the target's quantization
        # while flattening specs, which has nothing to do with the drafter
        assert "{quantization}" not in spec.draft_model_id

    # 12B is the encoder-free Unified variant and the only one whose drafter is
    # published in several conversions, so it exercises the drafter selector
    spec_12b = next(s for s in mlx_specs if s.model_size_in_billions == 12)
    assert len(spec_12b.draft_quantizations) > 1
    assert spec_12b.draft_quantizations[0] == "bf16"


def test_builtin_gemma_4_vllm_requires_mtp_dependencies():
    from ....core.utils import filter_virtualenv_packages_by_markers
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")
    assert family.virtualenv is not None

    packages = filter_virtualenv_packages_by_markers(
        family.virtualenv.packages, "vllm", "13.0", "linux"
    )
    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }

    assert requirements["vllm"].specifier.contains("0.22.0")
    assert not requirements["vllm"].specifier.contains("0.21.0")
    assert requirements["transformers"].specifier.contains("5.8.0")
    assert not requirements["transformers"].specifier.contains("5.7.0")
    assert isinstance(family.virtualenv.extra_index_url, list)
    assert "https://wheels.vllm.ai/0.22.0/cu130" in family.virtualenv.extra_index_url


def test_builtin_gemma_4_sglang_requires_supported_runtime():
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
        normalize_sglang_kernel_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")
    assert family.virtualenv is not None

    packages = expand_engine_dependency_placeholders(
        family.virtualenv.packages, "sglang"
    )
    packages = merge_virtual_env_packages(packages, None)
    packages = filter_virtualenv_packages_by_markers(
        packages, "sglang", "13.0", "linux"
    )
    packages, modern_kernel = normalize_sglang_kernel_packages(packages)

    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }

    assert requirements["sglang"].specifier.contains("0.5.13.post1")
    assert not requirements["sglang"].specifier.contains("0.5.11")
    assert requirements["transformers"].specifier.contains("5.8.1")
    assert not requirements["transformers"].specifier.contains("5.6.0")
    assert requirements["kernels"].specifier.contains("0.14.1")
    assert requirements["sglang-kernel"].specifier.contains("0.4.3")
    assert requirements["flash-attn-4"].specifier.contains("4.0.0b9")

    assert "sglang==0.5.13.post1" in packages
    assert "transformers==5.8.1" in packages
    assert "kernels==0.14.1" in packages
    assert "sglang-kernel==0.4.3" in packages
    assert "flash-attn-4==4.0.0b9" in packages
    assert not any("sgl_kernel-0.3.21" in package for package in packages)
    assert modern_kernel is True


def test_builtin_gemma_4_llama_cpp_requires_assistant_architecture():
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "gemma-4")
    assert family.virtualenv is not None

    packages = expand_engine_dependency_placeholders(
        family.virtualenv.packages, "llama.cpp"
    )
    packages = merge_virtual_env_packages(packages, None)
    packages = filter_virtualenv_packages_by_markers(
        packages, "llama.cpp", None, "darwin"
    )
    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }

    assert requirements["xllamacpp"].specifier.contains("2026.6.9713")
    assert not requirements["xllamacpp"].specifier.contains("2026.5.9294")
    assert sum(package.startswith("xllamacpp") for package in packages) == 1


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


@pytest.mark.parametrize(
    "model_name,expected_architecture,expected_formats",
    [
        (
            "Qwen3-Next-Instruct",
            "Qwen3NextForCausalLM",
            {
                "pytorch": {"none"},
                "fp8": {"fp8"},
                "awq": {"4bit", "8bit"},
            },
        ),
        (
            "Qwen3-Next-Thinking",
            "Qwen3NextForCausalLM",
            {
                "pytorch": {"none"},
                "fp8": {"fp8"},
                "awq": {"4bit", "8bit"},
            },
        ),
        (
            "MiniCPM-V-4.6",
            "MiniCPMV4_6ForConditionalGeneration",
            {
                "pytorch": {"none"},
                "bnb": {"4-bit"},
                "awq": {"Int4"},
                "gptq": {"Int4"},
            },
        ),
        (
            "MiniCPM-V-4.6-Thinking",
            "MiniCPMV4_6ForConditionalGeneration",
            {
                "pytorch": {"none"},
                "bnb": {"4-bit"},
                "awq": {"Int4"},
                "gptq": {"Int4"},
            },
        ),
        (
            "MiniMax-M3",
            "MiniMaxM3SparseForConditionalGeneration",
            {"pytorch": {"none"}},
        ),
    ],
)
def test_recent_sglang_engine_registration(
    monkeypatch, model_name, expected_architecture, expected_formats
):
    import xinference.model.llm as llm_module

    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..sglang import core as sglang_core

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == model_name
    )
    assert family.architectures == [expected_architecture]

    monkeypatch.setattr(
        llm_module,
        "SUPPORTED_ENGINES",
        {
            "SGLang": [
                sglang_core.SGLANGModel,
                sglang_core.SGLANGChatModel,
                sglang_core.SGLANGVisionModel,
            ]
        },
    )
    monkeypatch.setattr(llm_module, "LLM_ENGINES", {})
    monkeypatch.setattr(sglang_core, "SGLANG_INSTALLED", True)
    monkeypatch.setattr(sglang_core, "check_dependency_available", lambda *_args: True)
    monkeypatch.setattr(sglang_core.SGLANGModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(sglang_core.SGLANGModel, "_is_linux", lambda: True)

    llm_module.generate_engine_config_by_model_family(family)

    registrations = llm_module.LLM_ENGINES[model_name]["SGLang"]
    registered_formats = {
        registration["model_format"]: set(registration["quantizations"])
        for registration in registrations
    }
    assert registered_formats == expected_formats


@pytest.mark.parametrize(
    "model_name,expected_version,expected_formats",
    [
        ("Hy-MT2-1.8B", "0.21.0", {"pytorch", "ggufv2"}),
        ("Hy-MT2-7B", "0.21.0", {"pytorch", "ggufv2"}),
        ("Hy-MT2-30B-A3B", "0.21.0", {"pytorch"}),
        (
            "MiniCPM-V-4.6",
            "0.22.0",
            {"pytorch", "bnb", "awq", "gptq"},
        ),
        (
            "MiniCPM-V-4.6-Thinking",
            "0.22.0",
            {"pytorch", "bnb", "awq", "gptq"},
        ),
        ("MiniMax-M3", "0.24.0", {"pytorch"}),
    ],
)
def test_recent_vllm_engine_registration(
    monkeypatch, model_name, expected_version, expected_formats
):
    import xinference.model.llm as llm_module

    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..vllm import core as vllm_core

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == model_name
    )
    assert vllm_core._get_effective_vllm_version_for_family(family) == version.parse(
        expected_version
    )

    monkeypatch.setattr(
        llm_module,
        "SUPPORTED_ENGINES",
        {
            "vLLM": [
                vllm_core.VLLMModel,
                vllm_core.VLLMChatModel,
                vllm_core.VLLMMultiModel,
            ]
        },
    )
    monkeypatch.setattr(llm_module, "LLM_ENGINES", {})
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", version.parse("1.0.0"))
    monkeypatch.setattr(vllm_core.VLLMModel, "check_lib", classmethod(lambda cls: True))
    monkeypatch.setattr(vllm_core.VLLMModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(vllm_core.VLLMModel, "_is_linux", lambda: True)

    llm_module.generate_engine_config_by_model_family(family)

    registrations = llm_module.LLM_ENGINES[model_name]["vLLM"]
    assert {registration["model_format"] for registration in registrations} == (
        expected_formats
    )


@pytest.mark.parametrize(
    "model_name,engine_name,minimum_version",
    [
        ("MiniCPM-V-4.6", "vllm", "0.22.0"),
        ("MiniCPM-V-4.6-Thinking", "vllm", "0.22.0"),
        ("MiniCPM-V-4.6", "sglang", "0.5.12"),
        ("MiniCPM-V-4.6-Thinking", "sglang", "0.5.12"),
        ("MiniMax-M3", "vllm", "0.24.0"),
        ("MiniMax-M3", "sglang", "0.5.16"),
    ],
)
def test_recent_model_engine_minimum_versions(model_name, engine_name, minimum_version):
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == model_name
    )
    assert family.virtualenv is not None
    requirements = {
        requirement.name: requirement
        for package in family.virtualenv.packages
        if not package.startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }
    assert requirements[engine_name].specifier.contains(minimum_version)


def test_match_deepseek_v4_flash_0731():
    family = match_llm(
        "DeepSeek-V4-Flash-0731",
        model_format="fp8",
        model_size_in_billions=304,
        quantization="fp8",
        download_hub="huggingface",
    )

    assert family is not None
    assert family.model_name == "DeepSeek-V4-Flash-0731"
    assert family.context_length == 1048576
    assert family.architectures == ["DeepseekV4ForCausalLM"]
    assert family.tool_parser == "deepseek-v4"
    assert len(family.model_specs) == 1
    spec = family.model_specs[0]
    assert spec.model_format == "fp8"
    assert spec.model_size_in_billions == 304
    assert spec.quantization == "fp8"
    assert spec.model_hub == "huggingface"
    assert spec.model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"

    modelscope_family = match_llm(
        "DeepSeek-V4-Flash-0731",
        model_format="fp8",
        model_size_in_billions=304,
        quantization="fp8",
        download_hub="modelscope",
    )
    assert modelscope_family is not None
    assert modelscope_family.model_specs[0].model_hub == "modelscope"
    assert (
        modelscope_family.model_specs[0].model_id
        == "deepseek-ai/DeepSeek-V4-Flash-0731"
    )

    assert family.virtualenv is not None
    packages = family.virtualenv.packages
    assert 'vllm>=0.20.1 ; #engine# == "vllm"' in packages
    assert not any('#engine# == "Transformers"' in package for package in packages)
    assert not any('#engine# == "SGLang"' in package for package in packages)

    preview = match_llm(
        "DeepSeek-V4-Flash",
        model_format="pytorch",
        model_size_in_billions=284,
        quantization="none",
        download_hub="huggingface",
    )
    assert preview is not None
    assert preview.model_specs[0].model_id == "deepseek-ai/DeepSeek-V4-Flash"


def test_hy_mt2_remote_code_policy():
    import copy

    from ..llm_family import match_llm
    from ..transformers.hy_mt2 import HyMT2PytorchModel

    family = match_llm(
        "Hy-MT2-1.8B",
        model_format="pytorch",
        model_size_in_billions=2,
        quantization="none",
        download_hub="modelscope",
    )

    assert family is not None
    assert family.is_builtin is True
    assert family.architectures == ["HunYuanDenseV1ForCausalLM"]
    assert family.model_specs[0].model_hub == "modelscope"
    assert family.model_specs[0].model_id == "Tencent-Hunyuan/Hy-MT2-1.8B"
    assert (
        HyMT2PytorchModel.match_json(
            family, family.model_specs[0], family.model_specs[0].quantization
        )
        is True
    )

    builtin_model = HyMT2PytorchModel("hy-mt2-builtin-test", family, "/tmp/hy-mt2")
    assert builtin_model._pytorch_model_config["torch_dtype"] == "bfloat16"
    assert builtin_model._pytorch_model_config["trust_remote_code"] is True

    custom_family = copy.copy(family)
    custom_family.is_builtin = False
    custom_model = HyMT2PytorchModel("hy-mt2-custom-test", custom_family, "/tmp/hy-mt2")
    assert custom_model._pytorch_model_config["trust_remote_code"] is False


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


def test_qwen3_8_builtin_families_preserve_checkpoint_capabilities():
    from xinference.model.llm.llm_family import BUILTIN_LLM_FAMILIES
    from xinference.model.utils import generate_model_file_names_with_quantization_parts

    families = {family.model_name: family for family in BUILTIN_LLM_FAMILIES}
    dense = families["qwen3.8"]
    max_model = families["qwen3.8-max"]

    assert dense.context_length == max_model.context_length == 262144
    assert dense.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert set(dense.model_ability) == {
        "chat",
        "vision",
        "tools",
        "reasoning",
        "hybrid",
    }
    dense_specs = {
        (spec.model_hub, spec.model_format, spec.model_id, spec.quantization)
        for spec in dense.model_specs
    }
    assert {
        ("huggingface", "pytorch", "Qwen/Qwen3.8-27B", "none"),
        ("modelscope", "pytorch", "Qwen/Qwen3.8-27B", "none"),
        ("huggingface", "fp8", "Qwen/Qwen3.8-27B-FP8", "FP8"),
        ("modelscope", "fp8", "Qwen/Qwen3.8-27B-FP8", "FP8"),
    } <= dense_specs
    dense_gguf_specs = [
        spec for spec in dense.model_specs if spec.model_format == "ggufv2"
    ]
    assert {spec.model_hub for spec in dense_gguf_specs} == {
        "huggingface",
        "modelscope",
    }
    assert {spec.model_id for spec in dense_gguf_specs} == {"unsloth/Qwen3.8-27B-GGUF"}
    assert {spec.quantization for spec in dense_gguf_specs} == {
        "BF16",
        "IQ4_NL",
        "IQ4_XS",
        "Q3_K_M",
        "Q3_K_S",
        "Q4_0",
        "Q4_1",
        "Q4_K_M",
        "Q4_K_S",
        "Q5_K_M",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
        "UD-IQ2_M",
        "UD-IQ2_XXS",
        "UD-IQ3_XXS",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_XL",
        "UD-Q5_K_XL",
        "UD-Q6_K_XL",
        "UD-Q8_K_XL",
    }
    assert all(
        spec.multimodal_projectors == ["mmproj-BF16.gguf", "mmproj-F16.gguf"]
        for spec in dense_gguf_specs
    )
    assert all(
        spec.quantization_parts["BF16"] == ["00001-of-00002", "00002-of-00002"]
        and spec.model_file_name_split_template
        == "{quantization}/Qwen3.8-27B-{quantization}-{part}.gguf"
        for spec in dense_gguf_specs
        if spec.quantization == "BF16"
    )
    dense_bf16 = next(
        spec
        for spec in dense_gguf_specs
        if spec.model_hub == "huggingface" and spec.quantization == "BF16"
    )
    file_names, final_file_name, need_merge = (
        generate_model_file_names_with_quantization_parts(dense_bf16)
    )
    assert file_names == [
        "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
    ]
    assert final_file_name == "Qwen3.8-27B-BF16.gguf"
    assert need_merge is True

    dense_mlx_specs = [spec for spec in dense.model_specs if spec.model_format == "mlx"]
    assert {spec.model_hub for spec in dense_mlx_specs} == {
        "huggingface",
        "modelscope",
    }
    assert {spec.quantization for spec in dense_mlx_specs} == {
        "4bit",
        "8bit",
        "bf16",
    }
    assert all(
        spec.model_id == f"mlx-community/Qwen3.8-27B-{spec.quantization}"
        for spec in dense_mlx_specs
    )
    assert "reasoning_effort|default('xhigh')" in dense.chat_template

    assert max_model.architectures == ["Qwen3_5MoeForCausalLM"]
    assert set(max_model.model_ability) == {"chat", "tools", "reasoning"}
    assert "vision" not in max_model.model_ability
    assert "hybrid" not in max_model.model_ability
    max_specs = {
        (
            spec.model_hub,
            spec.model_format,
            spec.model_id,
            spec.quantization,
            spec.model_size_in_billions,
            spec.activated_size_in_billions,
        )
        for spec in max_model.model_specs
    }
    assert {
        (
            "huggingface",
            "pytorch",
            "Qwen/Qwen3.8-2.4T-A95B",
            "none",
            2400,
            95,
        ),
        (
            "modelscope",
            "pytorch",
            "Qwen/Qwen3.8-2.4T-A95B",
            "none",
            2400,
            95,
        ),
        (
            "huggingface",
            "fp8",
            "Qwen/Qwen3.8-2.4T-A95B-FP8",
            "FP8",
            2400,
            95,
        ),
        (
            "modelscope",
            "fp8",
            "Qwen/Qwen3.8-2.4T-A95B-FP8",
            "FP8",
            2400,
            95,
        ),
    } <= max_specs
    max_gguf_specs = [
        spec for spec in max_model.model_specs if spec.model_format == "ggufv2"
    ]
    max_gguf_part_counts = {
        "BF16": 140,
        "Q8_0": 56,
        "UD-IQ1_M": 13,
        "UD-IQ1_S": 12,
        "UD-IQ2_XS": 16,
        "UD-IQ2_XXS": 15,
        "UD-IQ3_XXS": 21,
        "UD-IQ4_XS": 29,
        "UD-Q1_0": 10,
    }
    assert {spec.model_hub for spec in max_gguf_specs} == {
        "huggingface",
        "modelscope",
    }
    assert {spec.model_id for spec in max_gguf_specs} == {
        "unsloth/Qwen3.8-2.4T-A95B-GGUF"
    }
    assert {spec.quantization for spec in max_gguf_specs} == set(max_gguf_part_counts)
    assert all(
        spec.model_size_in_billions == 2400
        and spec.activated_size_in_billions == 95
        and spec.model_file_name_template
        == f"Qwen3.8-2.4T-A95B-{spec.quantization}.gguf"
        and spec.model_file_name_split_template
        == "{quantization}/Qwen3.8-2.4T-A95B-{quantization}-{part}.gguf"
        and len(spec.quantization_parts[spec.quantization])
        == max_gguf_part_counts[spec.quantization]
        for spec in max_gguf_specs
    )
    max_ud_q1_0 = next(
        spec
        for spec in max_gguf_specs
        if spec.model_hub == "modelscope" and spec.quantization == "UD-Q1_0"
    )
    file_names, final_file_name, need_merge = (
        generate_model_file_names_with_quantization_parts(max_ud_q1_0)
    )
    assert file_names[0] == ("UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf")
    assert file_names[-1] == ("UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00010-of-00010.gguf")
    assert final_file_name == "Qwen3.8-2.4T-A95B-UD-Q1_0.gguf"
    assert need_merge is True
    assert "Disabling thinking is not supported." in max_model.chat_template


def test_qwen3_8_local_engine_virtualenv_requirements():
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ..llm_family import BUILTIN_LLM_FAMILIES

    families = {family.model_name: family for family in BUILTIN_LLM_FAMILIES}
    for family_name in ("qwen3.8", "qwen3.8-max"):
        family = families[family_name]
        assert family.virtualenv is not None
        packages = expand_engine_dependency_placeholders(
            family.virtualenv.packages, "llama.cpp"
        )
        packages = merge_virtual_env_packages(packages, None)
        packages = filter_virtualenv_packages_by_markers(
            packages, "llama.cpp", None, "darwin"
        )
        requirements = {
            requirement.name: requirement
            for package in packages
            if not package.lstrip().startswith("#")
            for requirement in [Requirement(package.split(";", 1)[0].strip())]
        }
        assert requirements["xllamacpp"].specifier.contains("2026.8.10229")
        assert not requirements["xllamacpp"].specifier.contains("2026.7.10068")
        assert sum(package.startswith("xllamacpp") for package in packages) == 1

    dense = families["qwen3.8"]
    assert dense.virtualenv is not None
    packages = expand_engine_dependency_placeholders(dense.virtualenv.packages, "mlx")
    packages = merge_virtual_env_packages(packages, None)
    packages = filter_virtualenv_packages_by_markers(packages, "mlx", None, "darwin")
    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }
    assert requirements["mlx-lm"].specifier.contains("0.24.0")
    assert requirements["mlx-vlm"].specifier.contains("0.6.13")
    assert not requirements["mlx-vlm"].specifier.contains("0.6.12")


def test_qwen3_8_max_requires_vllm_027_without_virtualenv(monkeypatch):
    from xinference.model.llm.llm_family import BUILTIN_LLM_FAMILIES
    from xinference.model.llm.vllm import core as vllm_core

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == "qwen3.8-max"
    )
    spec = next(spec for spec in family.model_specs if spec.model_format == "pytorch")
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", version.parse("0.26.0"))
    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: False)

    result = vllm_core.VLLMChatModel.match_json(family, spec, spec.quantization)
    assert result == (False, "Qwen3_5MoeForCausalLM requires vLLM >= 0.27.0")

    monkeypatch.setattr(vllm_core, "_virtual_env_allows_missing_vllm", lambda: True)
    assert vllm_core.VLLMChatModel.match_json(family, spec, spec.quantization) is True


def test_ornith_15_builtin_family_matches_modelscope_qwen3_5_moe():
    from ....core.model import XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel

    families = {family.model_name: family for family in BUILTIN_LLM_FAMILIES}
    family = families["Ornith-1.5-35B-A3B"]

    assert family.context_length == 262144
    assert family.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert family.model_ability == [
        "chat",
        "vision",
        "tools",
        "reasoning",
        "hybrid",
    ]

    spec = family.model_specs[0]
    assert spec.model_format == "pytorch"
    assert spec.model_size_in_billions == 35
    assert spec.activated_size_in_billions == 3
    assert spec.model_hub == "modelscope"
    assert spec.model_id == "ornith-ai/Ornith-1.5-35B-A3B"
    assert spec.quantization == "none"

    modelscope_family = match_llm(
        "Ornith-1.5-35B-A3B",
        model_format="pytorch",
        model_size_in_billions=35,
        quantization="none",
        download_hub="modelscope",
    )
    assert modelscope_family is not None
    assert modelscope_family.model_specs[0].model_hub == "modelscope"
    assert modelscope_family.model_specs[0].model_id == spec.model_id

    assert Qwen2VLChatModel.match_json(family, spec, spec.quantization) is True
    assert "Ornith-1.5-35B-A3B" in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
    assert (
        'transformers>=5.8.1 ; #engine# == "Transformers"' in family.virtualenv.packages
    )
    assert (
        'qwen-vl-utils!=0.0.9 ; #engine# == "Transformers"'
        in family.virtualenv.packages
    )
    assert 'sglang>=0.5.9 ; #engine# == "sglang"' in family.virtualenv.packages
    assert 'vllm==0.21.0 ; #engine# == "vllm"' in family.virtualenv.packages


def test_ornith_15_397b_builtin_family_matches_modelscope_qwen3_5_moe():
    from ....core.model import XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel

    families = {family.model_name: family for family in BUILTIN_LLM_FAMILIES}
    family = families["Ornith-1.5-397B"]

    assert family.context_length == 262144
    assert family.architectures == ["Qwen3_5MoeForConditionalGeneration"]
    assert family.model_ability == [
        "chat",
        "vision",
        "tools",
        "reasoning",
        "hybrid",
    ]

    spec = family.model_specs[0]
    assert spec.model_format == "pytorch"
    assert spec.model_size_in_billions == 397
    # The target model metadata does not provide a trustworthy activated
    # parameter count; do not inherit qwen3.5's A17B value.
    assert spec.activated_size_in_billions is None
    assert spec.model_hub == "modelscope"
    assert spec.model_id == "ornith-ai/Ornith-1.5-397B"
    assert spec.quantization == "none"

    modelscope_family = match_llm(
        "Ornith-1.5-397B",
        model_format="pytorch",
        model_size_in_billions=397,
        quantization="none",
        download_hub="modelscope",
    )
    assert modelscope_family is not None
    assert modelscope_family.model_specs[0].model_hub == "modelscope"
    assert modelscope_family.model_specs[0].model_id == spec.model_id

    assert Qwen2VLChatModel.match_json(family, spec, spec.quantization) is True
    assert "Ornith-1.5-397B" in XINFERENCE_BATCHING_ALLOWED_VISION_MODELS
    assert family.tool_parser == "qwen"
    assert "<function=example_function_name>" in family.chat_template
    assert (
        'transformers>=5.8.1 ; #engine# == "Transformers"' in family.virtualenv.packages
    )
    assert (
        'qwen-vl-utils!=0.0.9 ; #engine# == "Transformers"'
        in family.virtualenv.packages
    )
    assert 'sglang>=0.5.9 ; #engine# == "sglang"' in family.virtualenv.packages
    assert 'qwen-vl-utils!=0.0.9 ; #engine# == "sglang"' in family.virtualenv.packages
    assert 'vllm==0.21.0 ; #engine# == "vllm"' in family.virtualenv.packages


def test_ornith_15_397b_matches_vllm_021_multimodal(monkeypatch):
    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..vllm import core as vllm_core

    family = next(
        family
        for family in BUILTIN_LLM_FAMILIES
        if family.model_name == "Ornith-1.5-397B"
    )
    spec = family.model_specs[0]

    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", version.parse("0.21.0"))
    monkeypatch.setattr(
        vllm_core.VLLMMultiModel, "_has_cuda_device", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(
        vllm_core.VLLMMultiModel, "_is_linux", classmethod(lambda cls: True)
    )
    vllm_core._update_vllm_supported_lists()

    assert "Qwen3_5MoeForConditionalGeneration" in (
        vllm_core.VLLM_SUPPORTED_MULTI_MODEL_LIST
    )
    assert vllm_core.VLLMMultiModel.match_json(family, spec, spec.quantization) is True


def test_kimi_k3_tool_family_registration():
    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..utils import KIMI_K3_TOOL_CALL_FAMILY

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == "Kimi-K3"
    )
    assert "tools" in family.model_ability
    assert family.tool_parser == "kimi-k3"
    assert family.model_name in KIMI_K3_TOOL_CALL_FAMILY


def test_kimi_k3_virtualenv_engine_discovery(monkeypatch):
    import xinference.model.llm as llm_pkg
    from xinference.model.llm.vllm import core as vllm_core

    from ...utils import get_engine_params_by_name_with_virtual_env

    llm_pkg._install()
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", None)
    monkeypatch.setattr(
        vllm_core,
        "VLLM_SUPPORTED_MULTI_MODEL_LIST",
        [
            architecture
            for architecture in vllm_core.VLLM_SUPPORTED_MULTI_MODEL_LIST
            if architecture != "KimiK3ForConditionalGeneration"
        ],
    )
    monkeypatch.setattr(
        vllm_core.VLLMMultiModel,
        "_has_cuda_device",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        vllm_core.VLLMMultiModel,
        "_is_linux",
        classmethod(lambda cls: True),
    )
    params = get_engine_params_by_name_with_virtual_env(
        "LLM", "Kimi-K3", enable_virtual_env=True
    )

    assert params is not None
    assert isinstance(params.get("vLLM"), list), params.get("vLLM")
    family = next(
        family
        for family in llm_pkg.BUILTIN_LLM_FAMILIES
        if family.model_name == "Kimi-K3"
    )
    assert 'vllm>=0.27.0 ; #engine# == "vllm"' in family.virtualenv.packages
    assert vllm_core._get_virtualenv_vllm_version(family) == version.parse("0.27.0")
