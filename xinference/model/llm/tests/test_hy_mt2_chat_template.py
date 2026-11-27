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
def test_hy_mt3_engine_formats(monkeypatch, size):
    from packaging.version import Version

    from ..sglang import core as sglang_core
    from ..transformers.hy_mt2 import HyMT2PytorchModel
    from ..vllm import core as vllm_core

    name = f"Hy-MT3-{size}"
    family = next(f for f in BUILTIN_LLM_FAMILIES if f.model_name == name)
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version("0.29.0"))
    monkeypatch.setattr(
        vllm_core, "VLLM_SUPPORTED_CHAT_MODELS", list(family.architectures)
    )
    monkeypatch.setattr(sglang_core, "SGLANG_INSTALLED", True)

    for spec in family.model_specs:
        model_format = spec.model_format
        expected_transformers = model_format == "pytorch" or (
            model_format == "fp8" and size != "30B-A3B"
        )
        match_args = (family, spec, spec.quantization)
        assert (HyMT2PytorchModel.match_json(*match_args) is True) == (
            expected_transformers
        )
        for engine in (vllm_core.VLLMChatModel, sglang_core.SGLANGChatModel):
            assert (engine.match_json(*match_args) is True) == (
                model_format != "ggufv2"
            )


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
