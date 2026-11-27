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
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ...utils import generate_model_file_names_with_quantization_parts
from ..llm_family import BUILTIN_LLM_FAMILIES, match_llm
from ..utils import ChatModelMixin

MESSAGES = [
    {"role": "system", "content": "Translate into English."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "Hello"},
    {"role": "user", "content": "谢谢"},
]


@pytest.mark.parametrize(
    "size, expected",
    [
        (
            "1.8B",
            "<｜hy_begin▁of▁sentence｜>Translate into English."
            "<｜hy_place▁holder▁no▁3｜><｜hy_User｜>你好<｜hy_Assistant｜>"
            "Hello<｜hy_place▁holder▁no▁2｜><｜hy_User｜>谢谢<｜hy_Assistant｜>",
        ),
        (
            "7B",
            "<|startoftext|>Translate into English.<|extra_4|>你好<|extra_0|>"
            "Hello<|eos|><|startoftext|>谢谢<|extra_0|>",
        ),
        (
            "30B-A3B",
            "<｜hy_begin▁of▁sentence｜>Translate into English."
            "<｜reasoning_mode｜>reasoning_effort:no_think"
            "<｜hy_User｜>你好<｜hy_Assistant｜><think></think>Hello<eos:6124c78e>"
            "<｜hy_User｜>谢谢<｜hy_Assistant｜><think></think>",
        ),
    ],
)
@pytest.mark.parametrize("generation", ["Hy-MT2", "Hy-MT3"])
def test_prompt_uses_the_model_own_special_tokens(generation, size, expected):
    name = f"{generation}-{size}"
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    mixin = ChatModelMixin()
    mixin.tool_parser = None
    mixin.model_family = SimpleNamespace(
        model_name=name, model_family=None, model_ability=family.model_ability
    )

    assert mixin.get_full_context(MESSAGES, family.chat_template) == expected


@pytest.mark.parametrize(
    "name, size, stop_token_ids, gguf_quantizations",
    [
        ("Hy-MT3-1.8B", 2, [120020], ("Q4_K_M", "Q8_0")),
        ("Hy-MT3-7B", 7, [127960, 127967], ("IQ2", "Q4_K_M", "Q8_0")),
        ("Hy-MT3-30B-A3B", 30, [120025], ("IQ2", "Q4_K_M", "Q8_0")),
    ],
)
@pytest.mark.parametrize(
    "hub, namespace, revision",
    [("huggingface", "tencent", "main"), ("modelscope", "Tencent-Hunyuan", "master")],
)
def test_hy_mt3_matches_all_weight_variants(
    name, size, stop_token_ids, gguf_quantizations, hub, namespace, revision
):
    variants = [
        ("pytorch", "none", ""),
        ("fp8", "fp8", "-FP8"),
        *(("ggufv2", quantization, "-GGUF") for quantization in gguf_quantizations),
    ]
    for model_format, quantization, suffix in variants:
        family = match_llm(
            name,
            model_format=model_format,
            model_size_in_billions=size,
            quantization=quantization,
            download_hub=hub,
        )
        assert family is not None, (name, hub, model_format, quantization)
        assert family.stop_token_ids == stop_token_ids
        (spec,) = family.model_specs
        assert spec.model_hub == hub
        assert spec.model_id == f"{namespace}/{name}{suffix}"
        assert spec.model_revision == revision

        if model_format == "ggufv2":
            filename = f"{name}-{quantization}.gguf"
            assert generate_model_file_names_with_quantization_parts(spec) == (
                [filename],
                filename,
                False,
            )


@pytest.mark.parametrize("size", ["1.8B", "7B", "30B-A3B"])
@pytest.mark.parametrize(
    "enable_virtual_env, registered",
    [(False, True), (True, True), (True, False)],
)
def test_hy_mt3_engine_formats(
    monkeypatch, tmp_path, size, enable_virtual_env, registered
):
    from packaging.version import Version

    from .... import constants
    from .. import generate_engine_config_by_model_family
    from ..cache_manager import LLMCacheManager
    from ..core import create_llm_model_instance
    from ..llama_cpp.core import XllamaCppModel
    from ..llm_family import LLM_ENGINES
    from ..sglang import core as sglang_core
    from ..transformers.hy_mt2 import HyMT2PytorchModel
    from ..trt_llm.core import TRTLLMModel
    from ..vllm import core as vllm_core

    name = f"Hy-MT3-{size}"
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", registered)
    monkeypatch.setattr(
        vllm_core, "VLLM_VERSION", Version("0.29.0") if registered else None
    )
    monkeypatch.setattr(
        vllm_core,
        "VLLM_SUPPORTED_CHAT_MODELS",
        list(family.architectures) if registered else [],
    )
    monkeypatch.setattr(sglang_core, "SGLANG_INSTALLED", registered)
    monkeypatch.setitem(
        sys.modules, "xllamacpp", SimpleNamespace(__version__="2026.10.11870")
    )
    engine_classes = {
        "Transformers": HyMT2PytorchModel,
        "vLLM": vllm_core.VLLMChatModel,
        "SGLang": sglang_core.SGLANGChatModel,
        "llama.cpp": XllamaCppModel,
        "TRT-LLM": TRTLLMModel,
    }
    engines = {name: {}}
    if registered:
        for engine_class in engine_classes.values():
            monkeypatch.setattr(
                engine_class, "check_lib", classmethod(lambda cls: True)
            )
        generate_engine_config_by_model_family(family, engines)
    monkeypatch.setitem(LLM_ENGINES, name, engines[name])
    cache = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(LLMCacheManager, "cache", cache)

    for spec in family.model_specs:
        supported_engines = {
            "pytorch": {"Transformers", "vLLM", "SGLang"},
            "fp8": {"vLLM", "SGLang"},
            "ggufv2": {"llama.cpp"},
        }[spec.model_format]
        if spec.model_format == "fp8" and size != "30B-A3B":
            supported_engines.add("Transformers")
        launch_options = dict(
            model_uid="hy-mt3-test",
            model_name=name,
            model_format=spec.model_format,
            model_size_in_billions=spec.model_size_in_billions,
            quantization=spec.quantization,
            enable_virtual_env=enable_virtual_env,
        )
        for engine, engine_class in engine_classes.items():
            cache.reset_mock()
            if engine in supported_engines:
                model = create_llm_model_instance(model_engine=engine, **launch_options)
                assert type(model) is engine_class
                cache.assert_called_once()
            else:
                error = f"cannot be run on engine {engine}"
                if engine == "TRT-LLM" and enable_virtual_env:
                    error = "Engine TRT-LLM is not listed in virtualenv packages"
                with pytest.raises(ValueError, match=error):
                    create_llm_model_instance(model_engine=engine, **launch_options)
                cache.assert_not_called()


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_hy_mt3_cache_only_allows_older_runtime_for_supported_checkpoint(
    monkeypatch, tmp_path, enable_virtual_env
):
    from .... import constants
    from ..cache_manager import LLMCacheManager
    from ..core import create_llm_model_instance
    from ..llama_cpp.core import XllamaCppModel
    from ..llm_family import LLM_ENGINES

    name = "Hy-MT3-30B-A3B"
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    monkeypatch.setitem(LLM_ENGINES, name, {})
    monkeypatch.setitem(
        sys.modules, "xllamacpp", SimpleNamespace(__version__="2026.7.10068")
    )
    cache = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(LLMCacheManager, "cache", cache)

    launch_options = dict(
        model_uid="hy-mt3-cache-only-test",
        model_name=name,
        model_format="ggufv2",
        model_size_in_billions=30,
        quantization="Q4_K_M",
        enable_virtual_env=enable_virtual_env,
        _xinference_cache_only=True,
    )
    with pytest.raises(ValueError, match="cannot be run on engine vLLM"):
        create_llm_model_instance(model_engine="vLLM", **launch_options)
    cache.assert_not_called()

    model = create_llm_model_instance(model_engine="llama.cpp", **launch_options)
    assert type(model) is XllamaCppModel
    cache.assert_called_once()


@pytest.mark.parametrize(
    "enable_virtual_env, cache_only", [(False, False), (False, True), (True, False)]
)
def test_hy_mt3_vllm_launch_checks_requested_runtime(
    monkeypatch, tmp_path, enable_virtual_env, cache_only
):
    from packaging.version import Version

    from .... import constants
    from .. import generate_engine_config_by_model_family
    from ..cache_manager import LLMCacheManager
    from ..core import create_llm_model_instance
    from ..llm_family import LLM_ENGINES
    from ..vllm import core as vllm_core

    name = "Hy-MT3-30B-A3B"
    family = match_llm(name, "pytorch", 30, "none")
    assert family is not None
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", True)
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version("0.19.1"))
    monkeypatch.setattr(vllm_core, "VLLM_SUPPORTED_CHAT_MODELS", [])
    monkeypatch.setattr(
        vllm_core.VLLMChatModel, "check_lib", classmethod(lambda cls: True)
    )

    # Registration uses the global virtualenv default and can therefore list
    # vLLM even when the host package cannot run the model on its own.
    engines = {}
    generate_engine_config_by_model_family(family, engines)
    assert engines[name]["vLLM"][0]["llm_class"] is vllm_core.VLLMChatModel
    monkeypatch.setitem(LLM_ENGINES, name, engines[name])
    cache = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(LLMCacheManager, "cache", cache)

    launch_options = dict(
        model_uid="hy-mt3-vllm-runtime-test",
        model_name=name,
        model_engine="vLLM",
        model_format="pytorch",
        model_size_in_billions=30,
        quantization="none",
        enable_virtual_env=enable_virtual_env,
        _xinference_cache_only=cache_only,
    )
    if not enable_virtual_env and not cache_only:
        with pytest.raises(ValueError, match=r"requires vLLM >= 0\.21\.0"):
            create_llm_model_instance(**launch_options)
        cache.assert_not_called()
    else:
        assert (
            type(create_llm_model_instance(**launch_options)) is vllm_core.VLLMChatModel
        )
        cache.assert_called_once()


@pytest.mark.parametrize(
    "launch_options, expected", [({}, 8192), ({"context_length": 2048}, 2048)]
)
def test_hy_mt3_default_context_is_overridable(tmp_path, launch_options, expected):
    from ..core import create_llm_model_instance

    model = create_llm_model_instance(
        "hy-mt3-test",
        "Hy-MT3-1.8B",
        "Transformers",
        model_size_in_billions=2,
        model_format="pytorch",
        quantization="none",
        model_path=str(tmp_path),
        enable_virtual_env=True,
        **launch_options,
    )
    assert model._pytorch_model_config["context_length"] == expected


def test_hy_mt3_moe_gguf_requires_compatible_runtime(monkeypatch):
    from .... import constants
    from ..llama_cpp.core import XllamaCppModel

    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == "Hy-MT3-30B-A3B")
    monkeypatch.setattr(constants, "XINFERENCE_ENABLE_VIRTUAL_ENV", False)
    monkeypatch.setitem(
        sys.modules, "xllamacpp", SimpleNamespace(__version__="2026.7.10068")
    )
    with pytest.raises(ValueError, match="Hy-MT3-30B-A3B requires xllamacpp"):
        XllamaCppModel.validate_launch(family)
