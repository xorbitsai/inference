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
# CI: linux

import os
import shutil
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from packaging import version
from packaging.requirements import Requirement

from xinference._model_catalog import load_model_catalog

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


@pytest.mark.parametrize(
    "spec_cls,model_format",
    [
        (LlamaCppLLMSpecV2, "ggufv2"),
        (PytorchLLMSpecV2, "pytorch"),
        (MLXLLMSpecV2, "mlx"),
    ],
)
@pytest.mark.parametrize("size,expected", [("5_1", "5_1"), ("5", 5), (5, 5)])
def test_model_sizes_preserve_radix(spec_cls, model_format, size, expected):
    spec = spec_cls(
        model_format=model_format,
        model_size_in_billions=size,
        activated_size_in_billions=size,
        quantization="none",
        model_file_name_template="model.gguf",
    )
    assert spec.model_size_in_billions == expected
    assert spec.activated_size_in_billions == expected


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


@pytest.mark.parametrize("model_name", ["Spark-X2.5", "Spark-X2.5-Base"])
def test_builtin_spark_x2_5_sglang_requires_official_runtime(model_name):
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
        normalize_sglang_kernel_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    assert family.virtualenv is not None

    packages = expand_engine_dependency_placeholders(
        family.virtualenv.packages, "sglang"
    )
    packages = merge_virtual_env_packages(packages, None)
    packages = filter_virtualenv_packages_by_markers(
        packages, "sglang", "13.0", "linux"
    )
    packages, modern_kernel = normalize_sglang_kernel_packages(packages)

    assert (
        "sglang @ git+https://github.com/sgl-project/sglang.git@20621aa1"
        "#subdirectory=python" in packages
    )
    assert "torch==2.13.0+cu130" in packages
    assert "nvidia-cusparselt-cu13==0.8.1" in packages
    assert "sglang-kernel==0.4.6.post1" in packages
    assert "sgl-deep-gemm==0.1.5.post3" in packages
    assert "flashinfer_python[cu13]==0.6.17" in packages
    assert not any("sgl_kernel-0.3.21" in package for package in packages)
    assert modern_kernel is True


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_builtin_gemma_4_llama_cpp_requires_metal_bf16_compatible_runtime(platform):
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
        packages, "llama.cpp", None, platform
    )
    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }

    assert requirements["xllamacpp"].specifier.contains("2026.10.11870")
    # Gemma-4 Q4 GGUFs still contain a BF16 projection tensor. Older runtimes
    # dispatch its Metal kernel even when the default shader language omits it.
    for unsupported_version in ["2026.5.9294", "2026.6.9713", "2026.7.10068"]:
        assert not requirements["xllamacpp"].specifier.contains(unsupported_version)
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


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    return _isolated_hub(tmp_path, monkeypatch)


def _isolated_hub(tmp_path, monkeypatch):
    """A cache, LLM registry and engine table for one test, and small downloads.

    Models registered at startup are not visible and what the test registers is
    gone with it. A snapshot download writes README.md (repository id and
    revision) and the files in `shipped[repo_id]`; a file download writes that
    file, holding the repository id and revision. Either fails after writing if
    the repository is in `interrupted`; `downloads` lists (repo_id, revision).
    """
    import sys
    import types

    import huggingface_hub

    from .... import constants
    from ... import llm
    from ...cache_manager import CacheManager as BaseCacheManager
    from ...custom import RegistryManager
    from .. import custom, llm_family
    from ..llama_cpp.core import XllamaCppModel

    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(BaseCacheManager, "is_initialized", False)
    # An empty registry, family list and engine table: none of the models
    # registered at startup are visible, and what the test registers goes away.
    engines = {}
    monkeypatch.setattr(llm, "LLM_ENGINES", engines)
    monkeypatch.setattr(llm_family, "LLM_ENGINES", engines)
    monkeypatch.setattr(custom, "UD_LLM_FAMILIES", [])
    registry = custom.LLMModelRegistry()
    monkeypatch.setattr(RegistryManager, "_instances", {"llm": registry})
    # llama.cpp is listed for a registration only if xllamacpp is installed;
    # these tests build the engine object but never load it.
    monkeypatch.setattr(XllamaCppModel, "check_lib", classmethod(lambda cls: True))
    hub = SimpleNamespace(downloads=[], shipped={}, interrupted=set())

    def download(repo_id, revision, local_dir, files):
        hub.downloads.append((repo_id, revision))
        root = local_dir or str(tmp_path / "hub" / f"{repo_id}-{revision}")
        for name, content in files.items():
            os.makedirs(os.path.dirname(os.path.join(root, name)), exist_ok=True)
            with open(os.path.join(root, name), "w") as fd:
                fd.write(content)
        if repo_id in hub.interrupted:
            raise ConnectionError("connection lost")
        return root

    def snapshot_download(repo_id, revision=None, local_dir=None, **kwargs):
        files = {"README.md": f"{repo_id}@{revision}", **hub.shipped.get(repo_id, {})}
        return download(repo_id, revision, local_dir, files)

    def file_download(repo_id, name, revision, local_dir):
        root = download(repo_id, revision, local_dir, {name: f"{repo_id}@{revision}"})
        return os.path.join(root, name)

    def hf_hub_download(repo_id, filename=None, revision=None, local_dir=None, **kw):
        return file_download(repo_id, filename, revision, local_dir)

    def model_file_download(model_id, file_path, revision=None, **kwargs):
        return file_download(model_id, file_path, revision, None)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    for name, attributes in {
        "modelscope": {},
        "modelscope.hub": {},
        "modelscope.hub.file_download": {"model_file_download": model_file_download},
        "modelscope.hub.snapshot_download": {"snapshot_download": snapshot_download},
    }.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return hub


def _hub_family(name, *sources):
    """A custom family with one pytorch spec per (model_id, model_revision, model_hub)."""
    return CustomLLMFamilyV2(
        version=2,
        context_length=2048,
        model_name=name,
        model_family="my-family",
        model_lang=["en"],
        model_ability=["generate"],
        model_specs=[
            PytorchLLMSpecV2(
                model_format="pytorch",
                model_size_in_billions=1,
                quantization="none",
                model_id=model_id,
                model_revision=revision,
                model_hub=hub,
            )
            for model_id, revision, hub in sources
        ],
    )


def _register(name, *sources):
    from ..custom import register_llm

    register_llm(_hub_family(name, *sources), persist=False)


def _reregister(name, *sources):
    from ..custom import unregister_llm

    unregister_llm(name)  # the downloaded directory stays on disk
    _register(name, *sources)


def _launch_cache(name, hub="huggingface"):
    # what create_llm_model_instance asks the cache manager for
    return CacheManager(match_llm(name, "pytorch", 1, "none", hub)).cache()


def _readme(path, name="README.md"):
    with open(os.path.join(path, name)) as fd:
        return fd.read()


SOURCE_A = ("org/model-a", None, "huggingface")


@pytest.mark.parametrize(
    "new_source",
    [("org/model-b", None, "huggingface"), ("org/model-a", "v2", "huggingface")],
    ids=["other-repository", "other-revision"],
)
def test_reregistered_custom_llm_does_not_serve_the_files_of_its_old_source(
    fake_hub, new_source
):
    _register("my-llm", SOURCE_A)
    old_dir = _launch_cache("my-llm")
    assert _readme(old_dir) == "org/model-a@None"

    _reregister("my-llm", new_source)
    with pytest.raises(ValueError, match="my-llm.*org/model-a"):
        _launch_cache("my-llm")
    # nothing was fetched into the old directory, and nothing was deleted
    assert fake_hub.downloads == [("org/model-a", None)]
    assert _readme(old_dir) == "org/model-a@None"

    # a name without a cache is fetched from the new source
    _register("my-other-llm", new_source)
    assert _readme(_launch_cache("my-other-llm")) == "%s@%s" % new_source[:2]


def test_reregistered_custom_llm_with_the_same_source_reuses_its_cache(fake_hub):
    _register("my-llm", SOURCE_A)
    cache_dir = _launch_cache("my-llm")
    _reregister("my-llm", SOURCE_A)
    assert _launch_cache("my-llm") == cache_dir
    assert fake_hub.downloads == [("org/model-a", None)]


def test_custom_llm_cache_without_a_record_is_reused_whatever_files_it_holds(fake_hub):
    # cached before sources were recorded; a repository's own file that is named
    # like a record (here an empty object) is just one of its files
    _register("my-llm", ("org/model-b", None, "huggingface"))
    matched = match_llm("my-llm", "pytorch", 1, "none", "huggingface")
    cache_dir = CacheManager(matched).get_cache_dir()
    os.makedirs(cache_dir)
    shipped = {"README.md": "org/model-a@None", "__model_source.json": "{}"}
    for name, content in shipped.items():
        with open(os.path.join(cache_dir, name), "w") as fd:
            fd.write(content)
    assert _readme(_launch_cache("my-llm")) == "org/model-a@None"
    assert fake_hub.downloads == []


def test_custom_llm_model_uri_is_not_served_a_downloaded_directory(fake_hub, tmp_path):
    from ..custom import register_llm, unregister_llm

    _register("my-llm", SOURCE_A)
    old_dir = _launch_cache("my-llm")
    unregister_llm("my-llm")

    mine = tmp_path / "mine"
    os.makedirs(mine)
    with open(mine / "README.md", "w") as fd:
        fd.write("mine")
    family = _hub_family("my-llm", (None, None, "huggingface"))
    family.model_specs[0].model_uri = str(mine)
    register_llm(family, persist=False)
    with pytest.raises(ValueError, match="my-llm.*org/model-a"):
        _launch_cache("my-llm")
    assert _readme(old_dir) == "org/model-a@None"


@pytest.mark.parametrize(
    "quantization,requested_quantization",
    [("none", "none"), ("4-bit", None), ("4-bit", "4-bit")],
)
def test_custom_llm_hub_variants_share_the_download_after_the_registration_is_restored(
    fake_hub, monkeypatch, quantization, requested_quantization
):
    import sys
    import types

    from ..custom import register_llm, unregister_llm

    def no_download(*args, **kwargs):
        pytest.fail("the files are already cached")

    for name, attributes in {
        "modelscope": {},
        "modelscope.hub": {},
        "modelscope.hub.file_download": {"model_file_download": no_download},
        "modelscope.hub.snapshot_download": {"snapshot_download": no_download},
    }.items():
        module = types.ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    # one registration, one directory, two hubs; the ids are templates. A third
    # entry whose template cannot be expanded must not stand in the way
    variants = (
        ("org/model-{quantization}", None, "huggingface"),
        ("org-ms/model-{quantization}", None, "modelscope"),
        ("org/{unknown}", None, "csghub"),
    )

    def register_variants():
        family = _hub_family("my-llm", *variants)
        for spec in family.model_specs:
            spec.quantization = quantization
        register_llm(family, persist=False)

    def launch(hub):
        return CacheManager(
            match_llm("my-llm", "pytorch", 1, requested_quantization, hub)
        ).cache()

    register_variants()
    cache_dir = launch("huggingface")  # match_llm expands only the selected entry
    assert launch("modelscope") == cache_dir

    # Restored as registered: the unselected entry is a template again.
    unregister_llm("my-llm")
    register_variants()
    assert launch("modelscope") == cache_dir
    expected_quantization = requested_quantization or "none"
    assert fake_hub.downloads == [(f"org/model-{expected_quantization}", None)]


@pytest.mark.parametrize("partial_directory", [False, True])
def test_custom_llm_hub_variants_share_one_download_while_cache_is_incomplete(
    fake_hub, monkeypatch, partial_directory
):
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    _register("my-llm", SOURCE_A, ("org-ms/model-a", None, "modelscope"))
    first = CacheManager(match_llm("my-llm", "pytorch", 1, "none", "huggingface"))
    second = CacheManager(match_llm("my-llm", "pytorch", 1, "none", "modelscope"))
    cache_dir = first.get_cache_dir()
    downloading = threading.Event()
    second_started = threading.Event()
    finish_download = threading.Event()
    downloads = []

    def cache_from_hub(manager):
        if manager.get_cache_status():
            return cache_dir
        downloads.append(manager._model_hub)
        if manager is first:
            if partial_directory:
                os.makedirs(cache_dir)
                with open(os.path.join(cache_dir, "README.md"), "w") as fd:
                    fd.write("incomplete")
            downloading.set()
            assert finish_download.wait(5)
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "README.md"), "w") as fd:
            fd.write(manager._model_id)
        return cache_dir

    monkeypatch.setattr(CacheManager, "cache_from_huggingface", cache_from_hub)
    monkeypatch.setattr(CacheManager, "cache_from_modelscope", cache_from_hub)

    def launch_second():
        second_started.set()
        return second.cache()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.cache)
        try:
            assert downloading.wait(5)
            second_result = pool.submit(launch_second)
            assert second_started.wait(5)
            # Another hub must wait both before creation and while files arrive.
            with pytest.raises(TimeoutError):
                second_result.result(timeout=0.2)
        finally:
            finish_download.set()
        assert first_result.result(timeout=5) == cache_dir
        assert second_result.result(timeout=5) == cache_dir

    assert downloads == ["huggingface"]
    assert _readme(cache_dir) == "org/model-a"
    with open(first._source_record_path(), encoding="utf-8") as fd:
        # a snapshot is one unit: the record names its source for the whole directory
        assert json.load(fd) == {
            ".": {
                "model_hub": "huggingface",
                "model_id": "org/model-a",
                "model_revision": None,
            }
        }


def test_built_in_and_drafter_caches_are_not_recorded(fake_hub):
    from ..custom import register_llm

    built_in = CacheManager(
        match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    )
    built_in.cache()
    family = _hub_family("my-llm", SOURCE_A)
    family.model_specs[0].draft_model_id = "org/drafter"
    register_llm(family, persist=False)
    matched = match_llm("my-llm", "pytorch", 1, "none", "huggingface")
    drafter = CacheManager(matched, use_draft_model=True)
    assert _readme(drafter.cache()) == "org/drafter@None"
    # reused on the next launch: the drafter's repository is not the model's source
    assert drafter.cache() == drafter.get_cache_dir()
    for manager in (built_in, drafter):
        assert not os.path.exists(manager._source_record_path())
    assert fake_hub.downloads[1:] == [("org/drafter", None)]


def test_failed_provenance_write_never_leaves_a_download_without_a_record(
    fake_hub, monkeypatch
):
    import json

    from .. import cache_manager

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    _register("my-llm", SOURCE_A)
    with monkeypatch.context() as broken:
        broken.setattr(
            cache_manager, "json", SimpleNamespace(load=json.load, dump=refuse)
        )
        with pytest.raises(OSError, match="disk full"):
            _launch_cache("my-llm")
    assert fake_hub.downloads == []  # nothing was downloaded

    # so there is nothing that could be served under another source later
    _reregister("my-llm", ("org/model-b", None, "huggingface"))
    assert _readme(_launch_cache("my-llm")) == "org/model-b@None"
    assert fake_hub.downloads == [("org/model-b", None)]


def test_interrupted_download_cannot_be_reused_for_another_source(
    fake_hub, monkeypatch
):
    from ... import utils

    # the huggingface_hub < 0.23 branch downloads straight into the cache directory
    monkeypatch.setattr(utils, "IS_NEW_HUGGINGFACE_HUB", False)
    monkeypatch.setattr(utils, "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS", 1)
    fake_hub.interrupted.add("org/model-a")
    _register("my-llm", SOURCE_A)
    with pytest.raises(RuntimeError, match="after multiple retries"):
        _launch_cache("my-llm")  # leaves the files that arrived

    _reregister("my-llm", ("org/model-b", None, "huggingface"))
    with pytest.raises(ValueError, match="my-llm.*org/model-a"):
        _launch_cache("my-llm")
    assert fake_hub.downloads == [("org/model-a", None)]


@pytest.mark.parametrize(
    "damage", ["{", "{}", None], ids=["malformed", "not-a-source", "unreadable"]
)
def test_unreadable_or_invalid_provenance_is_not_taken_for_a_cache_from_before_records(
    fake_hub, damage
):
    _register("my-llm", SOURCE_A)
    manager = CacheManager(match_llm("my-llm", "pytorch", 1, "none", "huggingface"))
    cache_dir = manager.cache()
    record_path = manager._source_record_path()
    os.remove(record_path)
    if damage is None:
        os.makedirs(record_path)  # a directory where the record should be
    else:
        with open(record_path, "w") as fd:
            fd.write(damage)

    # not even the source it was downloaded from can be confirmed any more
    _reregister("my-llm", SOURCE_A)
    with pytest.raises(ValueError, match="cannot be read or is invalid"):
        _launch_cache("my-llm")
    assert fake_hub.downloads == [("org/model-a", None)]
    assert _readme(cache_dir) == "org/model-a@None"


def test_repository_file_named_like_a_record_is_neither_trusted_nor_displaced(fake_hub):
    import json

    # a repository that ships "__model_source.json", here claiming to be model-b
    claim = {
        "model_hub": "huggingface",
        "model_id": "org/model-b",
        "model_revision": None,
    }
    fake_hub.shipped["org/model-a"] = {"__model_source.json": json.dumps(claim)}
    _register("my-llm", SOURCE_A)
    cache_dir = _launch_cache("my-llm")
    with open(os.path.join(cache_dir, "__model_source.json")) as fd:
        assert json.load(fd) == claim  # still the repository's file

    _reregister("my-llm", ("org/model-b", None, "huggingface"))
    with pytest.raises(
        ValueError, match="holds files downloaded from huggingface:org/model-a"
    ):
        _launch_cache("my-llm")


def _gguf_family(name, *sources):
    """A custom family with one GGUF spec per (model_hub, model_id, file name).

    The specs share a cache directory: same name, format, size and quantization.
    """
    return CustomLLMFamilyV2(
        version=2,
        context_length=2048,
        model_name=name,
        model_family="my-family",
        model_lang=["en"],
        model_ability=["generate"],
        model_specs=[
            LlamaCppLLMSpecV2(
                model_format="ggufv2",
                model_size_in_billions=1,
                quantization="q4",
                model_id=model_id,
                model_hub=hub,
                model_file_name_template=file_name,
            )
            for hub, model_id, file_name in sources
        ],
    )


def _register_gguf(name, *sources, family=None):
    """Register the family, replacing an earlier registration of the name.

    The directory downloaded for the earlier registration stays on disk.
    """
    from ..custom import get_user_defined_llm_families, register_llm, unregister_llm

    if name in {f.model_name for f in get_user_defined_llm_families()}:
        unregister_llm(name)
    register_llm(family or _gguf_family(name, *sources), persist=False)


def _launch_gguf(name, hub, **kwargs):
    """A launch through the real entry point: engine selection, then the cache manager."""
    from ...core import create_model_instance

    model = create_model_instance(
        f"{name}-1-0",
        "LLM",
        name,
        "llama.cpp",
        "ggufv2",
        1,
        "q4",
        download_hub=hub,
        enable_virtual_env=False,
        **kwargs,
    )
    return model.model_path


def _gguf_manager(name, hub="huggingface"):
    return CacheManager(match_llm(name, "ggufv2", 1, "q4", hub))


def _source(hub, model_id, model_revision=None):
    return {"model_hub": hub, "model_id": model_id, "model_revision": model_revision}


GGUF_A = ("huggingface", "org/model-a", "a.gguf")
GGUF_B = ("modelscope", "org-ms/model-b", "b.gguf")


@pytest.mark.parametrize(
    "file_name,reused_name",
    [
        ("model.gguf", "model.gguf"),
        ("q4/model.gguf", "q4/model.gguf"),
        ("model.gguf", "MODEL.GGUF"),
    ],
)
def test_gguf_added_to_an_unrecorded_directory_is_not_served_to_another_source(
    fake_hub, file_name, reused_name
):
    import json

    _register_gguf("my-llm", ("huggingface", "org/model-a", file_name))
    cache_dir = _gguf_manager("my-llm").get_cache_dir()
    os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, "old.gguf"), "w") as fd:
        fd.write("legacy")
    with open(os.path.join(cache_dir, "mmproj.gguf"), "w") as fd:
        fd.write("legacy projector")
    assert (
        _launch_gguf("my-llm", "huggingface", multimodal_projector="mmproj.gguf")
        == cache_dir
    )
    assert _readme(cache_dir, "mmproj.gguf") == "legacy projector"
    assert _readme(cache_dir, file_name) == "org/model-a@None"
    with open(_gguf_manager("my-llm")._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {file_name: _source("huggingface", "org/model-a")}
    if not os.path.exists(os.path.join(cache_dir, reused_name)):
        pytest.skip("the cache volume distinguishes filename case")

    _register_gguf("my-llm", ("huggingface", "org/model-b", reused_name))
    with pytest.raises(
        ValueError, match="my-llm.*downloaded from huggingface:org/model-a"
    ):
        _launch_gguf("my-llm", "huggingface")
    assert fake_hub.downloads == [("org/model-a", None)]
    assert _readme(cache_dir, file_name) == "org/model-a@None"

    # The download did not claim the legacy file.
    _register_gguf("my-llm", ("huggingface", "org/model-b", "old.gguf"))
    assert _launch_gguf("my-llm", "huggingface") == cache_dir
    assert fake_hub.downloads == [("org/model-a", None)]


def test_gguf_file_is_accepted_when_only_the_hub_it_came_from_is_registered(fake_hub):
    import json

    _register_gguf("my-llm", GGUF_A, GGUF_B)
    cache_dir = _launch_gguf("my-llm", "huggingface")
    assert _launch_gguf("my-llm", "modelscope") == cache_dir
    assert fake_hub.downloads == [("org/model-a", None), ("org-ms/model-b", None)]
    assert _readme(cache_dir, "a.gguf") == "org/model-a@None"
    assert _readme(cache_dir, "b.gguf") == "org-ms/model-b@None"
    with open(_gguf_manager("my-llm")._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {
            "a.gguf": _source("huggingface", "org/model-a"),
            "b.gguf": _source("modelscope", "org-ms/model-b"),
        }

    _register_gguf("my-llm", GGUF_B)
    assert _launch_gguf("my-llm", "modelscope") == cache_dir
    assert len(fake_hub.downloads) == 2

    _register_gguf("my-llm", ("modelscope", "org-ms/model-c", "b.gguf"))
    with pytest.raises(ValueError, match="downloaded from modelscope:org-ms/model-b"):
        _launch_gguf("my-llm", "modelscope")
    assert len(fake_hub.downloads) == 2


@pytest.mark.parametrize("cached_main", [False, True])
def test_gguf_projector_source_is_checked_before_reuse_or_download(
    fake_hub, cached_main
):
    _register_gguf("my-llm", GGUF_A, GGUF_B)
    cache_dir = _launch_gguf(
        "my-llm", "huggingface", multimodal_projector="mmproj.gguf"
    )
    assert (
        _launch_gguf("my-llm", "huggingface", multimodal_projector="mmproj.gguf")
        == cache_dir
    )
    if cached_main:
        _launch_gguf("my-llm", "modelscope")
        _register_gguf("my-llm", GGUF_B)
    # A missing main file must not authorize overwriting A's projector,
    # even while both sources remain declared.
    before = list(fake_hub.downloads)
    with pytest.raises(ValueError, match="downloaded from huggingface:org/model-a"):
        _launch_gguf("my-llm", "modelscope", multimodal_projector="mmproj.gguf")
    assert fake_hub.downloads == before
    assert _readme(cache_dir, "mmproj.gguf") == "org/model-a@None"


def test_gguf_cache_hit_requires_one_declared_source_for_weights_and_projector(
    fake_hub,
):
    _register_gguf("my-llm", GGUF_A, GGUF_B)
    cache_dir = _launch_gguf(
        "my-llm", "huggingface", multimodal_projector="mmproj.gguf"
    )
    assert _launch_gguf("my-llm", "modelscope") == cache_dir
    record_path = _gguf_manager("my-llm")._source_record_path()
    with open(record_path, "rb") as fd:
        before = fd.read()

    # Both repositories remain declared, but their files cannot be used together.
    with pytest.raises(ValueError, match="different sources"):
        _launch_gguf("my-llm", "modelscope", multimodal_projector="mmproj.gguf")
    assert fake_hub.downloads == [
        ("org/model-a", None),
        ("org/model-a", None),
        ("org-ms/model-b", None),
    ]
    assert _readme(cache_dir, "b.gguf") == "org-ms/model-b@None"
    assert _readme(cache_dir, "mmproj.gguf") == "org/model-a@None"
    with open(record_path, "rb") as fd:
        assert fd.read() == before


def test_split_gguf_cache_hit_requires_one_declared_source_for_all_shards(fake_hub):
    sources = (
        ("huggingface", "org/model-a", "model-1-of-2.gguf"),
        ("modelscope", "org-ms/model-b", "model-2-of-2.gguf"),
    )
    _register_gguf("my-llm", *sources)
    cache_dir = _launch_gguf("my-llm", "huggingface")
    assert _launch_gguf("my-llm", "modelscope") == cache_dir
    record_path = _gguf_manager("my-llm")._source_record_path()
    with open(record_path, "rb") as fd:
        before = fd.read()
    family = _gguf_family("my-llm", *sources)
    for spec in family.model_specs:
        spec.model_file_name_split_template = "model-{part}.gguf"
        spec.quantization_parts = {"q4": ["1-of-2", "2-of-2"]}
    _register_gguf("my-llm", family=family)

    with pytest.raises(ValueError, match="different sources"):
        _launch_gguf("my-llm", "huggingface")
    assert fake_hub.downloads == [("org/model-a", None), ("org-ms/model-b", None)]
    assert _readme(cache_dir, "model-1-of-2.gguf") == "org/model-a@None"
    assert _readme(cache_dir, "model-2-of-2.gguf") == "org-ms/model-b@None"
    with open(record_path, "rb") as fd:
        assert fd.read() == before


@pytest.mark.parametrize("split", [False, True], ids=["single", "split"])
@pytest.mark.parametrize("legacy", [False, True], ids=["recorded", "legacy"])
def test_gguf_cache_hit_reuses_one_declared_source_and_retains_unknown_files(
    fake_hub, split, legacy
):
    import json

    family = _gguf_family(
        "my-llm",
        ("huggingface", "org/model-a", "model.gguf"),
        ("modelscope", "org-ms/model-b", "model.gguf"),
    )
    if split:
        for spec in family.model_specs:
            spec.model_file_name_split_template = "model-{part}.gguf"
            spec.quantization_parts = {"q4": ["1-of-2", "2-of-2"]}
    _register_gguf("my-llm", family=family)
    cache_dir = _launch_gguf(
        "my-llm", "huggingface", multimodal_projector="mmproj.gguf"
    )
    manager = _gguf_manager("my-llm")
    with open(manager._source_record_path(), encoding="utf-8") as fd:
        records = json.load(fd)
    if legacy:
        del records["mmproj.gguf"]
        if split:
            del records["model-2-of-2.gguf"]
        manager._record_source(records)
    before = list(fake_hub.downloads)

    # A hit can reuse another declared hub's coherent set, including unknown files.
    assert (
        _launch_gguf("my-llm", "modelscope", multimodal_projector="mmproj.gguf")
        == cache_dir
    )
    assert fake_hub.downloads == before
    assert _readme(cache_dir, "mmproj.gguf") == "org/model-a@None"
    if split:
        assert _readme(cache_dir, "model-2-of-2.gguf") == "org/model-a@None"
    with open(manager._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == records


def test_gguf_model_uri_is_not_served_a_downloaded_directory(fake_hub, tmp_path):
    _register_gguf("my-llm", GGUF_A)
    cache_dir = _launch_gguf("my-llm", "huggingface")

    mine = tmp_path / "mine"
    os.makedirs(mine)
    with open(mine / "c.gguf", "w") as fd:
        fd.write("mine")
    family = _gguf_family("my-llm", ("huggingface", None, "c.gguf"))
    family.model_specs[0].model_uri = str(mine)
    _register_gguf("my-llm", family=family)
    # The directory would be served whole, whether or not it holds c.gguf.
    with pytest.raises(
        ValueError, match="local model_uri.*downloaded from huggingface:org/model-a"
    ):
        _launch_gguf("my-llm", "huggingface")
    assert _readme(cache_dir, "a.gguf") == "org/model-a@None"


def test_gguf_cache_without_a_record_is_reused(fake_hub):
    # cached before sources were recorded, as far as the selected file goes
    _register_gguf("my-llm", ("huggingface", "org/model-b", "model.gguf"))
    cache_dir = _gguf_manager("my-llm").get_cache_dir()
    os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, "model.gguf"), "w") as fd:
        fd.write("legacy")
    assert _launch_gguf("my-llm", "huggingface") == cache_dir
    assert _readme(cache_dir, "model.gguf") == "legacy"
    assert fake_hub.downloads == []
    assert not os.path.exists(_gguf_manager("my-llm")._source_record_path())


def test_removed_gguf_cache_is_recorded_afresh(fake_hub):
    import json

    _register_gguf("my-llm", GGUF_A, GGUF_B)
    cache_dir = _launch_gguf("my-llm", "huggingface")
    _launch_gguf("my-llm", "modelscope")
    shutil.rmtree(cache_dir)  # the record stays behind

    _register_gguf("my-llm", ("huggingface", "org/model-c", "a.gguf"))
    assert (
        _readme(_launch_gguf("my-llm", "huggingface"), "a.gguf") == "org/model-c@None"
    )
    with open(_gguf_manager("my-llm")._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {"a.gguf": _source("huggingface", "org/model-c")}


@pytest.mark.parametrize("legacy_directory", [False, True])
@pytest.mark.parametrize("leaves_file", [False, True])
def test_interrupted_gguf_download_cannot_be_reused_for_another_source(
    fake_hub, monkeypatch, legacy_directory, leaves_file
):
    import huggingface_hub

    from ... import utils

    # the huggingface_hub < 0.23 branch downloads straight into the cache directory
    monkeypatch.setattr(utils, "IS_NEW_HUGGINGFACE_HUB", False)
    monkeypatch.setattr(utils, "XINFERENCE_DOWNLOAD_MAX_ATTEMPTS", 1)
    fake_hub.interrupted.add("org/model-a")
    _register_gguf("my-llm", ("huggingface", "org/model-a", "model.gguf"))
    if legacy_directory:
        os.makedirs(_gguf_manager("my-llm").get_cache_dir())

    def fail_before_writing(repo_id, **kwargs):
        fake_hub.downloads.append((repo_id, kwargs.get("revision")))
        raise ConnectionError("connection lost")

    with monkeypatch.context() as interrupted:
        if not leaves_file:
            interrupted.setattr(huggingface_hub, "hf_hub_download", fail_before_writing)
        with pytest.raises(RuntimeError, match="after multiple retries"):
            _launch_gguf("my-llm", "huggingface")

    _register_gguf("my-llm", ("huggingface", "org/model-b", "model.gguf"))
    if leaves_file:
        with pytest.raises(
            ValueError, match="my-llm.*downloaded from huggingface:org/model-a"
        ):
            _launch_gguf("my-llm", "huggingface")
        assert fake_hub.downloads == [("org/model-a", None)]
    else:
        cache_dir = _launch_gguf("my-llm", "huggingface")
        assert _readme(cache_dir, "model.gguf") == "org/model-b@None"
        assert fake_hub.downloads == [("org/model-a", None), ("org/model-b", None)]


@pytest.mark.parametrize("hub", ["huggingface", "modelscope"])
@pytest.mark.parametrize(
    "recorded_source,record_name",
    [
        (None, "model-2-of-2.gguf"),
        ("org/model-a", "model-2-of-2.gguf"),
        ("org/model-b", "model-2-of-2.gguf"),
        ("org/model-a", "recorded-shard.gguf"),
        ("org/model-b", "recorded-shard.gguf"),
    ],
    ids=[
        "legacy",
        "other-source",
        "selected-source",
        "other-source-alias",
        "selected-source-alias",
    ],
)
def test_split_gguf_completion_reuses_only_shards_recorded_for_selected_source(
    fake_hub, hub, recorded_source, record_name
):
    import json

    family = _gguf_family("my-llm", (hub, "org/model-b", "model.gguf"))
    family.model_specs[0].model_file_name_split_template = "model-{part}.gguf"
    family.model_specs[0].quantization_parts = {"q4": ["1-of-2", "2-of-2"]}
    _register_gguf("my-llm", family=family)
    manager = _gguf_manager("my-llm", hub)
    cache_dir = manager.get_cache_dir()
    os.makedirs(cache_dir)
    shard = "model-2-of-2.gguf"
    content = f"{recorded_source or 'org/model-a'}@None"
    with open(os.path.join(cache_dir, shard), "w") as fd:
        fd.write(content)
    if record_name != shard:
        os.link(os.path.join(cache_dir, shard), os.path.join(cache_dir, record_name))
    if recorded_source:
        manager._record_source({record_name: _source(hub, recorded_source)})

    if recorded_source == "org/model-b":
        assert _launch_gguf("my-llm", hub) == cache_dir
        assert fake_hub.downloads == [("org/model-b", None)]
        for name in ("model-1-of-2.gguf", shard, "model.gguf"):
            assert _readme(cache_dir, name) == "org/model-b@None"
    else:
        # Do not combine a new first shard with a second shard from A.
        message = (
            f"downloaded from {hub}:org/model-a"
            if recorded_source
            else "split GGUF.*no source record"
        )
        with pytest.raises(ValueError, match=message):
            _launch_gguf("my-llm", hub)
        assert fake_hub.downloads == []
        assert not os.path.exists(os.path.join(cache_dir, "model-1-of-2.gguf"))
        if recorded_source:
            with open(manager._source_record_path(), encoding="utf-8") as fd:
                assert json.load(fd) == {record_name: _source(hub, recorded_source)}
        else:
            assert not os.path.exists(manager._source_record_path())
    assert _readme(cache_dir, shard) == content


@pytest.mark.parametrize("relative_home", [False, True], ids=["absolute", "relative"])
def test_gguf_reregistration_checks_source_with_relative_cache_root(
    fake_hub, tmp_path, monkeypatch, relative_home
):
    import json

    from .... import constants

    monkeypatch.chdir(tmp_path)
    home = "home" if relative_home else str(tmp_path / "home")
    monkeypatch.setenv("XINFERENCE_HOME", home)
    monkeypatch.setattr(constants, "XINFERENCE_HOME", home)
    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", os.path.join(home, "cache"))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", os.path.join(home, "model"))
    _register_gguf("my-llm", ("huggingface", "org/model-a", "model.gguf"))
    cache_dir = _launch_gguf("my-llm", "huggingface")
    assert os.path.isabs(cache_dir) is not relative_home
    with open(_gguf_manager("my-llm")._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {"model.gguf": _source("huggingface", "org/model-a")}
    assert _launch_gguf("my-llm", "huggingface") == cache_dir

    _register_gguf("my-llm", ("huggingface", "org/model-b", "model.gguf"))
    with pytest.raises(
        ValueError, match="my-llm.*downloaded from huggingface:org/model-a"
    ):
        _launch_gguf("my-llm", "huggingface")
    assert fake_hub.downloads == [("org/model-a", None)]
    assert _readme(cache_dir, "model.gguf") == "org/model-a@None"


@pytest.mark.parametrize("selected_file", [None, "model.gguf", "model-2-of-2.gguf"])
def test_split_gguf_is_tied_to_the_file_the_runtime_opens(fake_hub, selected_file):
    import json

    def split_family(model_id):
        family = _gguf_family("my-llm", ("huggingface", model_id, "model.gguf"))
        spec = family.model_specs[0]
        spec.model_file_name_split_template = "model-{part}.gguf"
        spec.quantization_parts = {"q4": ["1-of-2", "2-of-2"]}
        return family

    _register_gguf("my-llm", family=split_family("org/model-a"))
    cache_dir = _launch_gguf("my-llm", "huggingface")
    assert fake_hub.downloads == [("org/model-a", None)] * 2  # one per part
    with open(_gguf_manager("my-llm")._source_record_path(), encoding="utf-8") as fd:
        assert json.load(fd) == {
            name: _source("huggingface", "org/model-a")
            for name in ("model-1-of-2.gguf", "model-2-of-2.gguf", "model.gguf")
        }

    _register_gguf("my-llm", family=split_family("org/model-a"))
    assert _launch_gguf("my-llm", "huggingface") == cache_dir
    assert len(fake_hub.downloads) == 2
    family = (
        split_family("org/model-b")
        if selected_file is None
        else _gguf_family("my-llm", ("huggingface", "org/model-b", selected_file))
    )
    _register_gguf("my-llm", family=family)
    with pytest.raises(
        ValueError, match="my-llm.*downloaded from huggingface:org/model-a"
    ):
        _launch_gguf("my-llm", "huggingface")
    assert len(fake_hub.downloads) == 2


def test_unreadable_provenance_blocks_a_gguf_download_into_the_directory(fake_hub):
    _register_gguf("my-llm", GGUF_A, GGUF_B)
    cache_dir = _launch_gguf("my-llm", "huggingface")
    with open(_gguf_manager("my-llm")._source_record_path(), "w") as fd:
        fd.write("{")

    # b.gguf would be added to a directory whose record cannot be updated
    with pytest.raises(ValueError, match="cannot be read or is invalid"):
        _launch_gguf("my-llm", "modelscope")
    assert fake_hub.downloads == [("org/model-a", None)]
    assert not os.path.exists(os.path.join(cache_dir, "b.gguf"))


def test_models_registered_before_the_fixture_do_not_reach_the_tests(
    tmp_path, monkeypatch
):
    from ... import llm
    from ...custom import RegistryManager
    from .. import custom, llm_family
    from ..custom import get_user_defined_llm_families

    # A disposable registry stands in for the one the server fills at startup
    # (never the developer's own), holding a valid registration of my-llm.
    with monkeypatch.context() as startup:
        _isolated_hub(tmp_path / "startup", startup)
        _register("my-llm", SOURCE_A)
        registry, engines = RegistryManager.get_registry("llm"), llm.LLM_ENGINES
        assert list(engines) == ["my-llm"]

        with monkeypatch.context() as isolated:
            hub = _isolated_hub(tmp_path / "isolated", isolated)
            assert get_user_defined_llm_families() == []
            assert llm.LLM_ENGINES == {}
            assert llm_family.LLM_ENGINES is llm.LLM_ENGINES
            # the same name registers without a duplicate-name error
            _register("my-llm", SOURCE_A)
            assert _readme(_launch_cache("my-llm")) == "org/model-a@None"
            assert hub.downloads == [("org/model-a", None)]
            assert list(llm.LLM_ENGINES) == ["my-llm"]

        # the outer registry, family list and engine table are back, untouched
        assert RegistryManager.get_registry("llm") is registry
        assert custom.UD_LLM_FAMILIES is registry.models
        assert llm.LLM_ENGINES is engines and llm_family.LLM_ENGINES is engines
        assert [f.model_name for f in get_user_defined_llm_families()] == ["my-llm"]
        assert list(engines) == ["my-llm"]


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


@pytest.fixture
def isolated_cache_home(tmp_path, monkeypatch):
    return _isolate_cache_home(tmp_path, monkeypatch)


def _isolate_cache_home(tmp_path, monkeypatch):
    """Cache, persisted registrations and LLM registry that exist for this test only."""
    from .... import constants
    from ... import llm
    from ...cache_manager import CacheManager as BaseCacheManager
    from ...custom import RegistryManager
    from .. import custom, llm_family

    monkeypatch.setattr(constants, "XINFERENCE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setattr(BaseCacheManager, "is_initialized", False)
    # An empty registry and engine table: none of the models registered at
    # startup are visible, and what the test registers is gone with them.
    engines = {}
    monkeypatch.setattr(llm, "LLM_ENGINES", engines)
    monkeypatch.setattr(llm_family, "LLM_ENGINES", engines)
    monkeypatch.setattr(custom, "UD_LLM_FAMILIES", [])
    registry = custom.LLMModelRegistry()
    monkeypatch.setattr(RegistryManager, "_instances", {"llm": registry})
    return tmp_path


def _weights_dir(path, marker):
    os.makedirs(path)
    for name in ("config.json", marker):
        with open(os.path.join(path, name), "w") as fd:
            fd.write(name)
    return str(path)


def _pytorch_spec(size="0_5", quantization="none", **source):
    return PytorchLLMSpecV2(
        model_format="pytorch",
        model_size_in_billions=size,
        quantization=quantization,
        **source,
    )


def _custom_family(name, *specs):
    return CustomLLMFamilyV2(
        version=2,
        context_length=2048,
        model_name=name,
        model_family="my-family",
        model_lang=["en"],
        model_ability=["generate"],
        model_specs=list(specs),
    )


def test_custom_llm_cannot_share_the_cache_directory_of_a_builtin(isolated_cache_home):
    from ..custom import get_user_defined_llm_families, register_llm

    builtin = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    assert builtin is not None
    builtin_dir = CacheManager(builtin).get_cache_dir()
    _weights_dir(builtin_dir, "BUILTIN_WEIGHTS")  # as left by a download
    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")

    # "." and "_" give the same directory name, so this is the built-in's cache
    twin = _custom_family("qwen2_5-instruct", _pytorch_spec(model_uri=mine))
    assert CacheManager(twin).get_cache_dir() == builtin_dir

    with pytest.raises(ValueError, match="qwen2.5-instruct.*cache directory"):
        register_llm(twin, persist=True)
    assert twin not in get_user_defined_llm_families()
    assert not os.path.exists(
        isolated_cache_home / "model" / "v2" / "llm" / "qwen2_5-instruct.json"
    )
    # the built-in is still served from its own files
    assert "BUILTIN_WEIGHTS" in os.listdir(CacheManager(builtin).cache())


@pytest.mark.parametrize("model_format", ["pytorch", "ggufv2", "mlx"])
def test_custom_twin_of_a_builtin_spec_is_refused_in_every_format(
    isolated_cache_home, model_format
):
    from ..custom import register_llm
    from ..llm_family import BUILTIN_LLM_FAMILIES

    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")
    family, spec = next(
        (family, spec)
        for family in BUILTIN_LLM_FAMILIES
        if "." in family.model_name
        for spec in family.model_specs
        if spec.model_format == model_format
    )
    twin = _custom_family(
        family.model_name.replace(".", "_"),
        spec.copy(update={"model_id": None, "model_uri": mine}),
    )
    with pytest.raises(ValueError, match="cache directory"):
        register_llm(twin, persist=False)


def test_custom_llm_with_its_own_cache_directory_keeps_its_own_weights(
    isolated_cache_home,
):
    from ..custom import register_llm

    builtin = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    builtin_dir = CacheManager(builtin).get_cache_dir()
    _weights_dir(builtin_dir, "BUILTIN_WEIGHTS")
    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")

    # the same twin name, but a size no built-in spec has: another directory
    twin = _custom_family("qwen2_5-instruct", _pytorch_spec("0_6", model_uri=mine))
    register_llm(twin, persist=False)
    register_llm(_custom_family("my-qwen", _pytorch_spec(model_uri=mine)), False)
    matched = match_llm("qwen2_5-instruct", "pytorch", "0_6", "none", "huggingface")
    assert CacheManager(matched).get_cache_dir() != builtin_dir
    assert "CUSTOM_WEIGHTS" in os.listdir(CacheManager(matched).cache())
    assert "BUILTIN_WEIGHTS" in os.listdir(CacheManager(builtin).cache())

    # one registration may list the same weights on two hubs: one directory
    hubs = _custom_family(
        "my-two-hubs",
        _pytorch_spec(model_id="org/my-model", model_hub="huggingface"),
        _pytorch_spec(model_id="org/my-model", model_hub="modelscope"),
    )
    register_llm(hubs, persist=False)
    # a family without specs names no directory
    register_llm(_custom_family("my-no-specs"), persist=False)


def test_custom_llms_register_unless_their_cache_directories_coincide(
    isolated_cache_home,
):
    from ..custom import register_llm

    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")
    register_llm(_custom_family("my.model", _pytorch_spec(model_uri=mine)), False)
    with pytest.raises(ValueError, match="my.model.*cache directory"):
        register_llm(_custom_family("my_model", _pytorch_spec(model_uri=mine)), False)
    # not the same format, size and quantization: not the same directory
    other = _custom_family("my_model", _pytorch_spec("1", "4-bit", model_uri=mine))
    register_llm(other, persist=False)


def test_startup_skips_a_persisted_registration_with_a_colliding_cache_directory(
    isolated_cache_home,
):
    from .. import register_custom_model
    from ..custom import get_user_defined_llm_families

    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")
    persisted = isolated_cache_home / "model" / "v2" / "llm"
    os.makedirs(persisted)
    for name in ("qwen2_5-instruct", "my-qwen"):
        family = _custom_family(name, _pytorch_spec(model_uri=mine))
        (persisted / f"{name}.json").write_text(family.json())

    with pytest.warns(UserWarning, match="qwen2_5-instruct.json has error.*cache"):
        register_custom_model()
    names = {family.model_name for family in get_user_defined_llm_families()}
    assert "my-qwen" in names
    assert "qwen2_5-instruct" not in names
    assert (persisted / "qwen2_5-instruct.json").exists()


@pytest.mark.parametrize("ignores_case", [True, False])
def test_custom_llm_named_like_a_builtin_up_to_case(
    isolated_cache_home, monkeypatch, ignores_case
):
    from .. import custom
    from ..custom import get_user_defined_llm_families, register_llm

    # stands in for the cache volume, so the result does not depend on the host
    monkeypatch.setattr(custom, "_volume_ignores_case", lambda path: ignores_case)
    builtin = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    assert builtin is not None
    mine = _weights_dir(isolated_cache_home / "mine", "CUSTOM_WEIGHTS")
    twin = _custom_family("Qwen2_5-instruct", _pytorch_spec(model_uri=mine))
    builtin_name = os.path.basename(CacheManager(builtin).get_cache_dir())
    twin_name = os.path.basename(CacheManager(twin).get_cache_dir())
    assert twin_name != builtin_name and twin_name.lower() == builtin_name

    if ignores_case:  # one directory on such a volume
        with pytest.raises(ValueError, match="qwen2.5-instruct.*cache directory"):
            register_llm(twin, persist=True)
        assert twin not in get_user_defined_llm_families()
    else:
        register_llm(twin, persist=False)
        assert twin in get_user_defined_llm_families()


@pytest.mark.parametrize("remove_parent", [False, True], ids=["empty", "missing"])
def test_startup_reloads_a_case_distinct_registration_after_cache_cleanup(
    isolated_cache_home, monkeypatch, remove_parent
):
    from .. import register_custom_model
    from ..custom import get_user_defined_llm_families, register_llm

    parent = isolated_cache_home / "cache" / "v2"
    parent.mkdir(parents=True)
    if _directory_ignores_case(parent):
        pytest.skip("This regression requires a case-sensitive cache directory")

    builtin = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none", "huggingface")
    assert builtin is not None
    twin = _custom_family("Qwen2_5-instruct", _pytorch_spec(model_id="org/custom"))
    builtin_dir = CacheManager.get_cache_dir_for_spec(
        builtin.model_name, builtin.model_specs[0]
    )
    twin_dir = CacheManager.get_cache_dir_for_spec(twin.model_name, twin.model_specs[0])
    assert builtin_dir != twin_dir and builtin_dir.casefold() == twin_dir.casefold()
    _weights_dir(builtin_dir, "BUILTIN_WEIGHTS")
    _weights_dir(twin_dir, "CUSTOM_WEIGHTS")
    assert not os.path.samefile(builtin_dir, twin_dir)

    register_llm(twin, persist=True)
    persisted = isolated_cache_home / "model" / "v2" / "llm" / f"{twin.model_name}.json"
    definition = persisted.read_text()

    # Establish that the persisted definition reloads while both caches exist.
    _isolate_cache_home(isolated_cache_home, monkeypatch)
    register_custom_model()
    assert get_user_defined_llm_families() == [twin]

    shutil.rmtree(builtin_dir)
    shutil.rmtree(twin_dir)
    assert list(parent.iterdir()) == []
    if remove_parent:
        parent.rmdir()
    assert parent.exists() == (not remove_parent)

    # A fresh startup must load the same JSON without either model's cache.
    _isolate_cache_home(isolated_cache_home, monkeypatch)
    register_custom_model()
    assert get_user_defined_llm_families() == [twin]
    assert persisted.read_text() == definition
    assert list(parent.iterdir()) == []


def _directory_ignores_case(path) -> bool:
    """Ground truth from two exclusive mkdir calls, independent of the probe."""
    original = path / "Probe"
    alternate = path / "pROBE"
    original.mkdir()
    try:
        try:
            alternate.mkdir()
        except FileExistsError:
            return True
        alternate.rmdir()
        return False
    finally:
        original.rmdir()


def test_volume_case_probe_matches_what_the_volume_does(tmp_path):
    from ..custom import _volume_ignores_case

    # The ground truth concerns entries inside tmp_path, not its own name.
    (tmp_path / "Probe").mkdir()
    try:
        (tmp_path / "pROBE").mkdir()
        expected = False
    except FileExistsError:
        expected = True

    for name in ("Probe", "Missing"):
        assert _volume_ignores_case(str(tmp_path / name)) is expected


@pytest.mark.parametrize("ignores_case", [True, False])
def test_volume_case_probe_uses_the_cache_parent(tmp_path, ignores_case):
    from .. import custom

    parent = str(tmp_path / "v2")
    probe = os.path.join(parent, "Probe")
    alternate = os.path.join(parent, "pROBE")
    stat = os.stat_result((0, 1, 1, 0, 0, 0, 0, 0, 0, 0))

    def lstat(path):
        if path == probe or (ignores_case and path == alternate):
            return stat
        raise FileNotFoundError(path)

    # Model a mount, symlink target or NTFS directory whose rules differ
    # from the directory containing its name.
    with patch.object(custom, "os", wraps=os) as filesystem:
        filesystem.path.isdir.side_effect = lambda path: path == parent or (
            not ignores_case and path == str(tmp_path / "V2")
        )
        filesystem.path.samefile.return_value = True
        filesystem.scandir.return_value = nullcontext(
            [SimpleNamespace(name="Probe", path=probe)]
        )
        filesystem.lstat.side_effect = lstat
        assert (
            custom._volume_ignores_case(os.path.join(parent, "Missing")) is ignores_case
        )
        filesystem.scandir.assert_called_once_with(parent)
        filesystem.makedirs.assert_not_called()


@pytest.mark.parametrize("state", ["empty", "missing", "no-case-entry"])
def test_volume_case_probe_without_a_usable_entry(tmp_path, state):
    from ..custom import _volume_ignores_case

    parent = tmp_path / "v2"
    if state != "missing":
        parent.mkdir()
    if state == "no-case-entry":
        (parent / "1234").mkdir()
    before = {"1234"} if state == "no-case-entry" else set()

    observed = _volume_ignores_case(str(parent / "Missing"))
    assert {entry.name for entry in parent.iterdir()} == before
    assert observed is _directory_ignores_case(parent)


def test_volume_case_probe_can_write_without_listing(tmp_path):
    from .. import custom

    expected = _directory_ignores_case(tmp_path)
    with patch.object(custom, "os", wraps=os) as filesystem:
        filesystem.scandir.side_effect = PermissionError("denied")
        assert custom._volume_ignores_case(str(tmp_path / "Missing")) is expected
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["makedirs", "mkdtemp", "lstat", "rmdir"])
def test_volume_case_probe_errors_are_conservative(tmp_path, operation):
    from .. import custom

    with (
        patch.object(custom, "os", wraps=os) as filesystem,
        patch.object(custom, "tempfile", wraps=tempfile) as temporary,
    ):
        owner = temporary if operation == "mkdtemp" else filesystem
        getattr(owner, operation).side_effect = PermissionError("denied")
        assert custom._volume_ignores_case(str(tmp_path / "Missing")) is True
        if operation in ("lstat", "rmdir"):
            filesystem.rmdir.assert_called_once()

    remaining = list(tmp_path.iterdir())
    if operation == "rmdir":
        # The simulated cleanup failure leaves only the directory we own.
        assert len(remaining) == 1
        assert remaining[0].name.startswith(".xinference-CaseProbe-")
        remaining[0].rmdir()
    else:
        assert remaining == []


def test_volume_case_probe_disappearing_entry_is_not_case_evidence(tmp_path):
    from .. import custom

    probe = str(tmp_path / "Probe")
    stat = os.stat_result((0, 1, 1, 0, 0, 0, 0, 0, 0, 0))
    with (
        patch.object(custom, "os", wraps=os) as filesystem,
        patch.object(custom.tempfile, "mkdtemp", side_effect=PermissionError("denied")),
    ):
        filesystem.scandir.return_value = nullcontext(
            [SimpleNamespace(name="Probe", path=probe)]
        )
        filesystem.lstat.side_effect = [stat, FileNotFoundError(), FileNotFoundError()]
        assert custom._volume_ignores_case(str(tmp_path / "Missing")) is True
        assert filesystem.lstat.call_count == 3


def test_volume_case_probe_distinguishes_hard_linked_entries(tmp_path):
    from .. import custom

    stat = os.stat_result((0, 1, 1, 0, 0, 0, 0, 0, 0, 0))
    with patch.object(custom, "os", wraps=os) as filesystem:
        filesystem.scandir.return_value = nullcontext(
            [
                SimpleNamespace(name=name, path=str(tmp_path / name))
                for name in ("Probe", "pROBE")
            ]
        )
        filesystem.lstat.return_value = stat
        assert custom._volume_ignores_case(str(tmp_path / "Missing")) is False
        filesystem.makedirs.assert_not_called()


def test_unknown_cache_case_rules_reject_new_and_persisted_aliases(
    isolated_cache_home, monkeypatch
):
    from .. import custom, register_custom_model
    from ..custom import get_user_defined_llm_families, register_llm

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(custom.tempfile, "mkdtemp", denied)
    twin = _custom_family("Qwen2_5-instruct", _pytorch_spec(model_id="org/custom"))
    persisted = isolated_cache_home / "model" / "v2" / "llm" / f"{twin.model_name}.json"
    with pytest.raises(ValueError, match="qwen2.5-instruct.*cache directory"):
        register_llm(twin, persist=True)
    assert get_user_defined_llm_families() == []
    assert not persisted.exists()

    # An existing JSON is not proof that case-only paths are distinct here.
    persisted.parent.mkdir(parents=True)
    persisted.write_text(twin.json())
    definition = persisted.read_text()
    with pytest.warns(UserWarning, match="Qwen2_5-instruct.json has error.*cache"):
        register_custom_model()
    assert get_user_defined_llm_families() == []
    assert persisted.read_text() == definition

    independent = _custom_family("my-qwen", _pytorch_spec(model_id="org/custom"))
    register_llm(independent, persist=False)
    assert get_user_defined_llm_families() == [independent]


def test_models_registered_before_the_fixture_do_not_reach_the_test(
    isolated_cache_home, monkeypatch
):
    from ... import llm
    from ...custom import RegistryManager
    from .. import custom, llm_family
    from ..custom import get_user_defined_llm_families, register_llm

    # Seed a disposable outer registry, never the developer's startup registry.
    family = _custom_family("registered-at-startup", _pytorch_spec(model_id="org/m"))
    register_llm(family, persist=False)
    registry = RegistryManager.get_registry("llm")
    engines = llm.LLM_ENGINES
    with monkeypatch.context() as isolated:
        _isolate_cache_home(isolated_cache_home, isolated)
        assert get_user_defined_llm_families() == []
        same_name = _custom_family(family.model_name, _pytorch_spec("1"))
        register_llm(same_name, persist=False)
        assert get_user_defined_llm_families() == [same_name]

    assert RegistryManager.get_registry("llm") is registry
    assert custom.UD_LLM_FAMILIES is registry.models
    assert llm.LLM_ENGINES is engines
    assert llm_family.LLM_ENGINES is engines
    assert get_user_defined_llm_families() == [family]


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


@pytest.fixture(
    params=["Ornith-1.5-35B-A3B", "Ornith-1.5-397B", "custom-modelscope-local"]
)
def modelscope_only_local_family(request, monkeypatch):
    from ...custom import RegistryManager
    from ..custom import register_llm
    from ..llm_family import BUILTIN_LLM_FAMILIES

    monkeypatch.setenv(XINFERENCE_ENV_MODEL_SRC, "huggingface")
    if request.param != "custom-modelscope-local":
        yield next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == request.param)
        return

    family = CustomLLMFamilyV2(
        version=2,
        context_length=2048,
        model_name=request.param,
        model_family="qwen2.5-instruct",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[
            PytorchLLMSpecV2(
                model_format="pytorch",
                model_size_in_billions=1,
                quantization="none",
                model_hub="modelscope",
                model_id="test/local-modelscope",
            )
        ],
    )
    register_llm(family, persist=False)
    try:
        yield family
    finally:
        RegistryManager.get_registry("llm").unregister(
            family.model_name, remove_file=False
        )


@pytest.fixture
def local_llm_factory(monkeypatch):
    from ...core import create_model_instance
    from .. import llm_family

    check_virtual_env = llm_family.check_engine_by_spec_parameters_with_virtual_env

    class LocalModel:
        def __init__(self, uid, family, path, config):
            self.model_family = family
            self.model_path = path

    def select_engine(*args, **kwargs):
        if "llm_family" in kwargs:
            assert kwargs["llm_family"].model_name == args[1]
            assert len(kwargs["llm_family"].model_specs) == 1
            assert check_virtual_env(*args, **kwargs) is not None
        return LocalModel

    def no_download(*args, **kwargs):
        pytest.fail("Local model matching must not detect a hub or download weights")

    monkeypatch.setattr(llm_family, "check_engine_by_spec_parameters", select_engine)
    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters_with_virtual_env", select_engine
    )
    monkeypatch.setattr(CacheManager, "cache", no_download)
    monkeypatch.setattr("xinference.model.utils.auto_detect_download_hub", no_download)
    return create_model_instance


@pytest.mark.parametrize("enable_virtual_env", [False, True])
@pytest.mark.parametrize(
    "local_path,download_hub,expected_hub",
    [
        (True, None, "modelscope"),
        (None, None, None),
        ("", None, None),
        (True, "huggingface", "modelscope"),
        (True, "modelscope", "modelscope"),
    ],
)
def test_local_model_path_modelscope_only_launch(
    modelscope_only_local_family,
    local_llm_factory,
    tmp_path,
    enable_virtual_env,
    local_path,
    download_hub,
    expected_hub,
):
    family = modelscope_only_local_family
    spec = family.model_specs[0]
    kwargs = dict(
        model_uid="local-model",
        model_type="LLM",
        model_name=family.model_name,
        model_engine="Transformers",
        model_format=spec.model_format,
        model_size_in_billions=spec.model_size_in_billions,
        quantization=spec.quantization,
        download_hub=download_hub,
        model_path=str(tmp_path) if local_path is True else local_path,
        enable_virtual_env=enable_virtual_env,
    )
    if expected_hub is None:
        with pytest.raises(ValueError, match="Model not found"):
            local_llm_factory(**kwargs)
    else:
        model = local_llm_factory(**kwargs)
        assert model.model_family.model_name == family.model_name
        assert model.model_family.model_specs[0].model_hub == expected_hub
        assert model.model_path == str(tmp_path)


@pytest.mark.parametrize(
    "model_src", [None, "huggingface", "modelscope", "openmind_hub", "csghub", "auto"]
)
def test_local_model_path_preserves_source_preference(
    monkeypatch, tmp_path, local_llm_factory, model_src
):
    from .. import llm_family

    if model_src is None:
        monkeypatch.delenv(XINFERENCE_ENV_MODEL_SRC, raising=False)
    else:
        monkeypatch.setenv(XINFERENCE_ENV_MODEL_SRC, model_src)
    monkeypatch.setattr(
        "xinference.model.utils.is_locale_chinese_simplified", lambda: False
    )
    family = llm_family.get_llm_family_by_name("qwen2.5-instruct").copy(deep=True)
    # Source preference must hold even when other hubs are registered first.
    family.model_specs.sort(key=lambda spec: spec.model_hub == "huggingface")
    monkeypatch.setattr(llm_family, "BUILTIN_LLM_FAMILIES", [family])
    model = local_llm_factory(
        "local-model",
        "LLM",
        "qwen2.5-instruct",
        "Transformers",
        "pytorch",
        "0_5",
        "none",
        model_path=str(tmp_path),
        enable_virtual_env=True,
    )
    expected_hub = "modelscope" if model_src == "modelscope" else "huggingface"
    assert model.model_family.model_specs[0].model_hub == expected_hub
    # Non-local matching still follows the environment.
    if model_src in ("huggingface", "modelscope"):
        family = match_llm("qwen2.5-instruct", "pytorch", "0_5", "none")
        assert family.model_specs[0].model_hub == model_src


def test_local_model_path_preserves_modelscope_gguf_filename(
    monkeypatch, tmp_path, local_llm_factory
):
    from ..gguf_model_metadata import resolve_gguf_model_path

    monkeypatch.setenv(XINFERENCE_ENV_MODEL_SRC, "modelscope")
    model_file = tmp_path / "ggml-model-Q2_K.gguf"
    model_file.touch()
    model = local_llm_factory(
        "local-model",
        "LLM",
        "tiny-llama",
        "llama.cpp",
        "ggufv2",
        1,
        "Q2_K",
        model_path=str(tmp_path),
        enable_virtual_env=False,
    )
    spec = model.model_family.model_specs[0]
    assert resolve_gguf_model_path(str(tmp_path), spec) == str(model_file)
    assert spec.model_hub == "modelscope"


@pytest.mark.parametrize("download_hub", [None, "huggingface", "modelscope"])
def test_local_model_path_repository_lookup(
    modelscope_only_local_family, tmp_path, download_hub
):
    from ....core.worker import WorkerActor

    family = modelscope_only_local_family
    spec = family.model_specs[0]
    payload = dict(
        model_name=family.model_name,
        model_format=spec.model_format,
        model_size_in_billions=spec.model_size_in_billions,
        quantization=spec.quantization,
        model_path=str(tmp_path),
        download_hub=download_hub,
    )
    assert WorkerActor._resolve_model_download_repositories(payload) == []


def test_local_model_path_unknown_family(tmp_path, local_llm_factory):
    with pytest.raises(ValueError, match="Model not found, name: unknown-local-model"):
        local_llm_factory(
            "local-model",
            "LLM",
            "unknown-local-model",
            "Transformers",
            model_path=str(tmp_path),
        )


@pytest.mark.parametrize(
    "model_name,expected_architecture,expected_formats",
    [
        (
            "Qwen3-VL-Instruct",
            "Qwen3VLMoeForConditionalGeneration",
            {
                ("pytorch", 235): {"none"},
                ("fp8", 235): {"fp8"},
                ("awq", 235): {"Int4"},
                ("pytorch", 30): {"none"},
                ("fp8", 30): {"fp8"},
                ("awq", 30): {"4bit", "8bit"},
                ("pytorch", 32): {"none"},
                ("fp8", 32): {"fp8"},
                ("awq", 32): {"Int4"},
                ("pytorch", 8): {"none"},
                ("fp8", 8): {"fp8"},
                ("awq", 8): {"4bit", "8bit"},
                ("pytorch", 4): {"none"},
                ("fp8", 4): {"fp8"},
                ("awq", 4): {"4bit", "8bit"},
                ("pytorch", 2): {"none"},
                ("fp8", 2): {"fp8"},
            },
        ),
        (
            "Qwen3-VL-Thinking",
            "Qwen3VLMoeForConditionalGeneration",
            {
                ("pytorch", 235): {"none"},
                ("fp8", 235): {"fp8"},
                ("awq", 235): {"Int4"},
                ("pytorch", 30): {"none"},
                ("fp8", 30): {"fp8"},
                ("awq", 30): {"4bit", "8bit"},
            },
        ),
        (
            "Qwen3-Next-Instruct",
            "Qwen3NextForCausalLM",
            {
                ("pytorch", 80): {"none"},
                ("fp8", 80): {"fp8"},
                ("awq", 80): {"4bit", "8bit"},
            },
        ),
        (
            "Qwen3-Next-Thinking",
            "Qwen3NextForCausalLM",
            {
                ("pytorch", 80): {"none"},
                ("fp8", 80): {"fp8"},
                ("awq", 80): {"4bit", "8bit"},
            },
        ),
        (
            "MiniCPM-V-4.6",
            "MiniCPMV4_6ForConditionalGeneration",
            {
                ("pytorch", 1): {"none"},
                ("bnb", 1): {"4-bit"},
                ("awq", 1): {"Int4"},
                ("gptq", 1): {"Int4"},
            },
        ),
        (
            "MiniCPM-V-4.6-Thinking",
            "MiniCPMV4_6ForConditionalGeneration",
            {
                ("pytorch", 1): {"none"},
                ("bnb", 1): {"4-bit"},
                ("awq", 1): {"Int4"},
                ("gptq", 1): {"Int4"},
            },
        ),
        (
            "MiniMax-M3",
            "MiniMaxM3SparseForConditionalGeneration",
            {("pytorch", 428): {"none"}},
        ),
        (
            "minicpm5-2b",
            "LlamaForCausalLM",
            {("pytorch", 2): {"none"}, ("awq", 2): {"Int4"}},
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
        (
            registration["model_format"],
            registration["model_size_in_billions"],
        ): set(registration["quantizations"])
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
        ("minicpm5-2b", "0.21.0", {"pytorch", "awq", "ggufv2"}),
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
        ("minicpm5-2b", "vllm", "0.21.0"),
        ("minicpm5-2b", "sglang", "0.5.16"),
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


def test_minicpm5_2b_builtin_family_has_all_published_formats():
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(
        family for family in BUILTIN_LLM_FAMILIES if family.model_name == "minicpm5-2b"
    )
    specs = {
        spec.model_format: {
            (candidate.quantization, candidate.model_id)
            for candidate in family.model_specs
            if candidate.model_hub == "huggingface"
            and candidate.model_format == spec.model_format
        }
        for spec in family.model_specs
        if spec.model_hub == "huggingface"
    }

    assert specs == {
        "pytorch": {("none", "openbmb/MiniCPM5-2B")},
        "ggufv2": {
            ("F16", "openbmb/MiniCPM5-2B-GGUF"),
            ("Q4_K_M", "openbmb/MiniCPM5-2B-GGUF"),
            ("Q8_0", "openbmb/MiniCPM5-2B-GGUF"),
        },
        "mlx": {("4bit", "openbmb/MiniCPM5-2B-MLX")},
        # Despite repository name, config.json declares AWQ GEMM packing.
        "awq": {("Int4", "openbmb/MiniCPM5-2B-GPTQ")},
    }
    assert family.tool_parser == "minicpm5"
    assert "<think>\\n\\n</think>" in family.chat_template


def test_spark_x2_5_builtin_families_cover_official_hubs_and_formats():
    from ..llm_family import BUILTIN_LLM_FAMILIES
    from ..sglang.core import SGLANG_SUPPORTED_CHAT_MODELS, SGLANG_SUPPORTED_MODELS
    from ..transformers.spark_x2_5 import SparkX25PytorchChatModel, SparkX25PytorchModel
    from ..utils import GLM5_TOOL_CALL_FAMILY
    from ..vllm import core as vllm_core

    families = {family.model_name: family for family in BUILTIN_LLM_FAMILIES}
    instruction = families["Spark-X2.5"]
    base = families["Spark-X2.5-Base"]

    assert instruction.context_length == 1048576
    assert instruction.model_ability == ["chat", "tools", "reasoning", "hybrid"]
    assert instruction.architectures == ["Spark2_5ForCausalLM"]
    assert instruction.tool_parser == "glm5"
    assert "Spark2_5ForCausalLM" in vllm_core.VLLM_SUPPORTED_CHAT_MODELS
    assert "Spark2_5ForCausalLM" in vllm_core.VLLM_SUPPORTED_MODELS
    assert "Spark2_5ForCausalLM" in SGLANG_SUPPORTED_CHAT_MODELS
    assert "Spark2_5ForCausalLM" in SGLANG_SUPPORTED_MODELS
    assert instruction.model_name in GLM5_TOOL_CALL_FAMILY

    expected_instruction_specs = {
        ("pytorch", "1_7", "none", "XHToken/Spark-X2.5-1.7B"),
        ("pytorch", 4, "none", "XHToken/Spark-X2.5-4B"),
        ("pytorch", "1_7", "Int8", "XHToken/Spark-X2.5-1.7B-INT8"),
        ("pytorch", 4, "Int8", "XHToken/Spark-X2.5-4B-INT8"),
        ("fp8", "1_7", "FP8", "XHToken/Spark-X2.5-1.7B-FP8"),
        ("fp8", 4, "FP8", "XHToken/Spark-X2.5-4B-FP8"),
        ("ggufv2", "1_7", "Q4_K_M", "XHToken/Spark-X2.5-1.7B-GGUF"),
        ("ggufv2", "1_7", "Q8_0", "XHToken/Spark-X2.5-1.7B-GGUF"),
        ("ggufv2", 4, "Q4_K_M", "XHToken/Spark-X2.5-4B-GGUF"),
        ("ggufv2", 4, "Q8_0", "XHToken/Spark-X2.5-4B-GGUF"),
    }
    for model_hub, revision in (("huggingface", "main"), ("modelscope", "master")):
        specs = {
            (
                spec.model_format,
                spec.model_size_in_billions,
                spec.quantization,
                spec.model_id,
            )
            for spec in instruction.model_specs
            if spec.model_hub == model_hub
        }
        assert specs == expected_instruction_specs
        assert all(
            spec.model_revision == revision
            for spec in instruction.model_specs
            if spec.model_hub == model_hub
        )

    assert base.model_ability == ["generate"]
    for family in (instruction, base):
        assert family.virtualenv is not None
        packages = family.virtualenv.packages
        assert 'transformers==4.57.1 ; #engine# == "Transformers"' in packages
        assert 'vllm==0.29.0 ; #engine# == "vllm"' in packages
        assert 'transformers==5.16.1 ; #engine# == "vllm"' in packages
        assert 'tokenizers==0.23.2 ; #engine# == "vllm"' in packages
        assert 'vllm-spark2-5-plugin==0.1.0 ; #engine# == "vllm"' in packages
        assert 'openai>=2.25.0 ; #engine# == "vllm"' in packages

    huggingface_base_specs = {
        (spec.model_size_in_billions, spec.model_id, spec.model_revision)
        for spec in base.model_specs
        if spec.model_hub == "huggingface"
    }
    assert huggingface_base_specs == {
        ("1_7", "XHToken/Spark-X2.5-1.7B-Base", "main"),
        (4, "XHToken/Spark-X2.5-4B-Base", "main"),
    }
    modelscope_base_specs = {
        (spec.model_size_in_billions, spec.model_id, spec.model_revision)
        for spec in base.model_specs
        if spec.model_hub == "modelscope"
    }
    assert modelscope_base_specs == {
        ("1_7", "XHToken/Spark-X2.5-1.7B-Base", "master"),
        (4, "XHToken/Spark-X2.5-4B-Base", "master"),
    }

    instruction_spec = next(
        spec
        for spec in instruction.model_specs
        if spec.model_hub == "huggingface"
        and spec.model_format == "pytorch"
        and spec.quantization == "none"
    )
    base_spec = next(
        spec for spec in base.model_specs if spec.model_hub == "huggingface"
    )
    assert (
        SparkX25PytorchChatModel.match_json(
            instruction, instruction_spec, instruction_spec.quantization
        )
        is True
    )
    assert (
        SparkX25PytorchModel.match_json(base, base_spec, base_spec.quantization) is True
    )

    vllm_model = object.__new__(vllm_core.VLLMModel)
    vllm_model.model_family = instruction
    vllm_model.model_spec = instruction_spec
    vllm_model._device_count = 1
    vllm_model._n_worker = 1
    vllm_model._shard = 0
    vllm_model._address = None
    with patch.object(vllm_core, "VLLM_VERSION", version.parse("0.21.0")):
        assert vllm_model._sanitize_model_config({})["trust_remote_code"] is True

    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "prompt"

    tokenizer = Tokenizer()
    model = SparkX25PytorchChatModel.__new__(SparkX25PytorchChatModel)
    model.model_family = instruction
    model.reasoning_parser = None
    model._tokenizer = tokenizer
    assert (
        model._get_full_prompt(
            [{"role": "user", "content": "hello"}],
            [{"type": "function", "function": {"name": "weather"}}],
            {},
        )
        == "prompt"
    )
    assert tokenizer.calls[0][1]["tools"] == [
        {"type": "function", "function": {"name": "weather"}}
    ]
    assert "chat_template" not in tokenizer.calls[0][1]


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


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_internvl_engines_after_lmdeploy_removal(enable_virtual_env):
    from ...utils import (
        get_engine_params_by_name,
        get_engine_params_by_name_with_virtual_env,
    )
    from ..llm_family import (
        check_engine_by_spec_parameters,
        check_engine_by_spec_parameters_with_virtual_env,
    )

    if enable_virtual_env:
        params = get_engine_params_by_name_with_virtual_env(
            "LLM", "InternVL3", enable_virtual_env=True
        )
        check_engine = check_engine_by_spec_parameters_with_virtual_env
        error = "not listed in virtualenv packages"
    else:
        params = get_engine_params_by_name("LLM", "InternVL3", enable_virtual_env=False)
        check_engine = check_engine_by_spec_parameters
        error = "cannot be run on engine"

    assert "LMDEPLOY" not in params
    assert isinstance(params["Transformers"], list)
    for engine in ("LMDEPLOY", "lmdeploy"):
        with pytest.raises(ValueError, match=error):
            check_engine(engine, "InternVL3", "pytorch", 1, "none")


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
    from xinference.model.llm.mlx import core as mlx_core
    from xinference.model.llm.sglang import core as sglang_core
    from xinference.model.llm.vllm import core as vllm_core

    from ... import utils as model_utils

    llm_pkg._install()

    monkeypatch.setattr(model_utils, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(mlx_core, "sys", SimpleNamespace(platform="linux"))

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

    params = model_utils.get_engine_params_by_name_with_virtual_env(
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

    # MLX cannot run on this simulated Linux worker.
    assert not isinstance(
        params.get("MLX"), list
    ), f"MLX should not be available for qwen3.5, got {params.get('MLX')!r}"


@pytest.mark.parametrize("lib_available", [False, True])
@pytest.mark.parametrize("model_name", ["qwen3.5", "qwen3.6"])
@pytest.mark.parametrize(
    "host_platform,processor",
    [("darwin", "arm"), ("darwin", "i386"), ("linux", "x86_64")],
)
def test_qwen3_5_and_3_6_mlx_engine_available_with_virtualenv(
    monkeypatch, host_platform, processor, lib_available, model_name
):
    from ... import utils as model_utils
    from .. import llm_family
    from ..mlx import core as mlx_core

    monkeypatch.setattr(model_utils, "sys", SimpleNamespace(platform=host_platform))
    monkeypatch.setattr(mlx_core, "sys", SimpleNamespace(platform=host_platform))
    monkeypatch.setattr(
        mlx_core, "platform", SimpleNamespace(processor=lambda: processor)
    )
    # Start discovery without pre-populated engines so the result does not
    # depend on the host or optional libraries used at test collection time.
    monkeypatch.setattr(llm_family, "LLM_ENGINES", {model_name: {}})
    monkeypatch.setattr(
        llm_family, "SUPPORTED_ENGINES", {"MLX": llm_family.MLX_CLASSES}
    )
    for model_cls in llm_family.MLX_CLASSES:
        monkeypatch.setattr(
            model_cls,
            "check_lib",
            classmethod(lambda cls: lib_available),
        )

    params = model_utils.get_engine_params_by_name_with_virtual_env(
        "LLM", model_name, enable_virtual_env=True
    )
    assert params is not None
    mlx_params = params["MLX"]
    if host_platform != "darwin" or processor != "arm":
        assert isinstance(mlx_params, str)
        return

    assert isinstance(mlx_params, list)
    expected_sizes = {"0_8", 2, 4, 9, 27, 35, 122, 397}
    if model_name == "qwen3.6":
        expected_sizes = {35}
    assert {param["model_size_in_billions"] for param in mlx_params} == expected_sizes
    assert all(param["model_format"] == "mlx" for param in mlx_params)
    assert all("4bit" in param["quantizations"] for param in mlx_params)
    assert all(
        param.get("virtualenv_required", False) is not lib_available
        for param in mlx_params
    )


@pytest.mark.parametrize("engine", ["MLX", "Transformers"])
def test_qwen3_5_mlx_virtualenv_requirements(engine):
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen3.5")
    assert family.virtualenv is not None
    packages = expand_engine_dependency_placeholders(family.virtualenv.packages, engine)
    packages = merge_virtual_env_packages(packages, None)
    packages = filter_virtualenv_packages_by_markers(packages, engine, None, "darwin")
    requirements = {
        requirement.name: requirement
        for package in packages
        if not package.lstrip().startswith("#")
        for requirement in [Requirement(package.split(";", 1)[0].strip())]
    }
    if engine != "MLX":
        assert "mlx-lm" not in requirements
        assert "mlx-vlm" not in requirements
        return

    assert "mlx-lm" in requirements
    assert requirements["mlx-vlm"].specifier.contains("0.3.12")
    assert not requirements["mlx-vlm"].specifier.contains("0.3.11")


def test_builtin_mlx_families_have_virtualenv_dependencies(monkeypatch):
    from ....core.utils import (
        filter_virtualenv_packages_by_markers,
        merge_virtual_env_packages,
    )
    from ....core.virtual_env_manager import expand_engine_dependency_placeholders
    from ... import utils as model_utils
    from .. import llm_family
    from ..mlx import core as mlx_core

    # These records need catalog or loader fixes before enabling their MLX
    # paths: DeepSeek V4 is only implemented in mlx-vlm, Kimi's registered MLX
    # quant is text-only, and GLM-5/5.1 incorrectly advertise vision ability.
    incompatible_models = {
        "DeepSeek-V4-Flash",
        "DeepSeek-V4-Pro",
        "Kimi-K2.5",
        "glm-5",
        "glm-5.1",
    }
    families = [
        family
        for family in llm_family.BUILTIN_LLM_FAMILIES
        if any(spec.model_format == "mlx" for spec in family.model_specs)
        and family.model_name not in incompatible_models
    ]
    assert families
    monkeypatch.setattr(model_utils, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(mlx_core, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(mlx_core, "platform", SimpleNamespace(processor=lambda: "arm"))
    monkeypatch.setattr(
        llm_family, "LLM_ENGINES", {family.model_name: {} for family in families}
    )
    monkeypatch.setattr(
        llm_family, "SUPPORTED_ENGINES", {"MLX": llm_family.MLX_CLASSES}
    )
    for model_cls in llm_family.MLX_CLASSES:
        monkeypatch.setattr(model_cls, "check_lib", classmethod(lambda cls: False))

    for family in families:
        params = model_utils.get_engine_params_by_name_with_virtual_env(
            "LLM", family.model_name, enable_virtual_env=True
        )
        assert params is not None, family.model_name
        assert isinstance(params.get("MLX"), list), (family.model_name, params)
        assert all(param["virtualenv_required"] for param in params["MLX"])
        for spec in family.model_specs:
            if spec.model_format != "mlx":
                continue
            packages = family.virtualenv.packages.copy() if family.virtualenv else []
            spec_virtualenv = getattr(spec, "virtualenv", None)
            if spec_virtualenv:
                packages.extend(spec_virtualenv.packages)
            packages = expand_engine_dependency_placeholders(packages, "MLX")
            packages = merge_virtual_env_packages(packages, None)
            packages = filter_virtualenv_packages_by_markers(
                packages, "MLX", None, "darwin"
            )
            requirements = {
                Requirement(package.split(";", 1)[0].strip()).name
                for package in packages
                if not package.lstrip().startswith("#")
            }
            assert "mlx-lm" in requirements, family.model_name
            if "vision" in family.model_ability:
                assert {"mlx-vlm", "mlx_vlm"} & requirements, family.model_name


@pytest.mark.parametrize(
    "model_name,minimum_version,unsupported_version",
    [("qwen3.6", "0.5.0", "0.4.4"), ("MiniMax-M3", "0.6.4", "0.6.3")],
)
def test_new_mlx_vision_models_require_supported_runtime_versions(
    model_name, minimum_version, unsupported_version
):
    from ....core.utils import filter_virtualenv_packages_by_markers
    from ..llm_family import BUILTIN_LLM_FAMILIES

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name)
    assert family.virtualenv is not None
    packages = filter_virtualenv_packages_by_markers(
        family.virtualenv.packages, "MLX", None, "darwin"
    )
    requirement = next(
        Requirement(package.split(";", 1)[0].strip())
        for package in packages
        if package.startswith("mlx-vlm")
    )
    assert requirement.specifier.contains(minimum_version)
    assert not requirement.specifier.contains(unsupported_version)


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
        "Q4_0",
        "Q4_1",
        "Q8_0",
        "UD-IQ2_XXS",
        "UD-IQ3_XXS",
        "UD-IQ4_XS",
        "UD-Q2_K_XL",
        "UD-Q3_K_XL",
        "UD-Q4_K_M",
        "UD-Q4_K_S",
        "UD-Q4_K_XL",
        "UD-Q5_K_M",
        "UD-Q5_K_S",
        "UD-Q5_K_XL",
        "UD-Q6_K",
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
    (
        file_names,
        final_file_name,
        need_merge,
    ) = generate_model_file_names_with_quantization_parts(dense_bf16)
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
    (
        file_names,
        final_file_name,
        need_merge,
    ) = generate_model_file_names_with_quantization_parts(max_ud_q1_0)
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
    assert requirements["mlx-lm"].specifier.contains("0.31.3")
    assert not requirements["mlx-lm"].specifier.contains("0.31.2")
    assert requirements["llguidance"].specifier.contains("1.7.0")
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


def test_register_builtin_model_is_idempotent():
    # Worker.update_model_type() calls register_builtin_model() again on
    # every runtime hub-refresh, on the same already-imported process. The
    # LLM_ENGINES entries were already guarded against duplication (see
    # generate_engine_config_by_model_family's already_exists check); the
    # SUPPORTED_ENGINES class lists were not.
    import xinference.model.llm as llm_module

    from ..llm_family import BUILTIN_LLM_FAMILIES, LLM_ENGINES, SUPPORTED_ENGINES

    llm_module.register_builtin_model()
    model_name = next(iter(LLM_ENGINES))
    baseline_classes = {
        engine: list(classes) for engine, classes in SUPPORTED_ENGINES.items()
    }
    baseline_engine_entries = sum(
        len(specs) for specs in LLM_ENGINES[model_name].values()
    )
    baseline_family_count = len(BUILTIN_LLM_FAMILIES)

    for _ in range(3):
        llm_module.register_builtin_model()

    assert {
        engine: list(classes) for engine, classes in SUPPORTED_ENGINES.items()
    } == baseline_classes
    assert (
        sum(len(specs) for specs in LLM_ENGINES[model_name].values())
        == baseline_engine_entries
    )
    # BUILTIN_LLM_FAMILIES itself must not grow either: load_model_family_from_json
    # unconditionally appended every family on every refresh, independent of the
    # engine-class and engine-entry guards above.
    assert len(BUILTIN_LLM_FAMILIES) == baseline_family_count


def test_register_builtin_model_preserves_and_removes_downloaded_provenance(
    tmp_path, monkeypatch
):
    import json
    import os

    from .... import constants
    from .. import register_builtin_model
    from ..core import LLM_VERSION_INFOS
    from ..llm_family import (
        BUILTIN_LLM_FAMILIES,
        BUILTIN_LLM_MODEL_GENERATE_FAMILIES,
        LLM_ENGINES,
    )

    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path))

    import xinference.model.llm as llm_module

    spec_path = os.path.join(os.path.dirname(llm_module.__file__), "models")
    downloaded_entry = next(
        entry
        for entry in load_model_catalog(spec_path)
        if "generate" in entry["model_ability"]
    )
    model_name = "downloaded-only-llm-refresh-test"
    downloaded_entry["model_name"] = model_name

    builtin_dir = os.path.join(str(tmp_path), "v2", "builtin", "llm")
    os.makedirs(builtin_dir, exist_ok=True)
    catalog_path = os.path.join(builtin_dir, "llm_models.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump([downloaded_entry], f)

    register_builtin_model()
    active = [f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name]
    assert len(active) == 1
    assert active[0].is_builtin is False
    assert model_name in BUILTIN_LLM_MODEL_GENERATE_FAMILIES
    assert model_name in LLM_ENGINES
    assert model_name in LLM_VERSION_INFOS

    register_builtin_model()
    active = [f for f in BUILTIN_LLM_FAMILIES if f.model_name == model_name]
    assert len(active) == 1
    assert active[0].is_builtin is False

    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    register_builtin_model()
    assert not any(f.model_name == model_name for f in BUILTIN_LLM_FAMILIES)
    assert model_name not in BUILTIN_LLM_MODEL_GENERATE_FAMILIES
    assert model_name not in LLM_ENGINES
    assert model_name not in LLM_VERSION_INFOS
