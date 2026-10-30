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

import pytest
from packaging.version import Version

from ..llm_family import match_llm
from ..sglang import core as sglang_core
from ..transformers.core import PytorchChatModel
from ..transformers.multimodal.qwen2_vl import Qwen2VLChatModel
from ..vllm import core as vllm_core

VARIANTS = [
    ("qwen3.5", 397, "Qwen3.5-397B-A17B", 17),
    ("qwen3.5", 122, "Qwen3.5-122B-A10B", 10),
    ("qwen3.6", 27, "Qwen3.6-27B", None),
    ("qwen3.6", 35, "Qwen3.6-35B-A3B", 3),
    ("qwen3.8", 27, "Qwen3.8-27B", None),
    ("qwen3.8-max", 2400, "Qwen3.8-2.4T-A95B", 95),
    ("qwen3.8-Flash-Next", 125, "Qwen3.8-Flash-Next", 6),
]


@pytest.mark.parametrize("model_name,size,checkpoint,activated", VARIANTS)
@pytest.mark.parametrize(
    "hub,owner", [("huggingface", "nvidia"), ("modelscope", "nv-community")]
)
def test_qwen_nvfp4_checkpoints(model_name, size, checkpoint, activated, hub, owner):
    family = match_llm(model_name, "fp4", size, "FP4", hub)
    spec = family.model_specs[0]
    assert spec.model_id == f"{owner}/{checkpoint}-NVFP4"
    assert spec.model_hub == hub
    assert spec.activated_size_in_billions == activated
    assert family.context_length == 262144
    assert ("vision" in family.model_ability) == (model_name != "qwen3.8-max")
    metadata = spec.model_metadata
    assert metadata is not None
    assert f"/{spec.model_id}/resolve/" in metadata.config_source
    assert len(metadata.config_sha256) == 64
    assert metadata.architecture_type == "hybrid_attention"


@pytest.mark.parametrize("model_name,size,checkpoint,activated", VARIANTS)
def test_qwen_nvfp4_engine_matching(
    monkeypatch, model_name, size, checkpoint, activated
):
    family = match_llm(model_name, "fp4", size, "FP4", "huggingface")
    spec = family.model_specs[0]
    monkeypatch.setattr(vllm_core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version("0.30.0"))
    monkeypatch.setattr(sglang_core, "SGLANG_INSTALLED", True)
    for model_cls in (vllm_core.VLLMMultiModel, sglang_core.SGLANGVisionModel):
        monkeypatch.setattr(model_cls, "_has_cuda_device", lambda: True)
        monkeypatch.setattr(model_cls, "_is_linux", lambda: True)
    is_text_only = "vision" not in family.model_ability
    vllm_cls = vllm_core.VLLMChatModel if is_text_only else vllm_core.VLLMMultiModel
    sglang_cls = (
        sglang_core.SGLANGChatModel if is_text_only else sglang_core.SGLANGVisionModel
    )
    assert vllm_cls.match_json(family, spec, "FP4") is True
    # Qwen4Exp remains vLLM-only until SGLang supports that architecture.
    result = sglang_cls.match_json(family, spec, "FP4")
    assert (result is True) == (model_name != "qwen3.8-Flash-Next")
    assert Qwen2VLChatModel.match_json(family, spec, "FP4")[0] is False
    assert PytorchChatModel.match_json(family, spec, "FP4")[0] is False


@pytest.mark.parametrize("model_name,size,checkpoint,activated", VARIANTS)
def test_qwen_nvfp4_uses_checkpoint_quantizer(
    monkeypatch, model_name, size, checkpoint, activated
):
    family = match_llm(model_name, "fp4", size, "FP4", "huggingface")
    spec = family.model_specs[0]
    vllm = object.__new__(vllm_core.VLLMModel)
    vllm.model_family = family
    vllm.model_spec = spec
    vllm._device_count = vllm._n_worker = 1
    monkeypatch.setattr(vllm_core, "VLLM_VERSION", Version("0.30.0"))
    config = vllm._sanitize_model_config({})
    assert config["quantization"] is None
    sglang = object.__new__(sglang_core.SGLANGModel)
    sglang.model_spec = spec
    config = {"quantization": "FP4"}
    sglang._apply_fp4_config(config)
    assert "quantization" not in config
    assert config["dtype"] == "bfloat16"


@pytest.mark.parametrize(
    "model_cls,architecture,abilities",
    [
        (sglang_core.SGLANGChatModel, "LlamaForCausalLM", ["chat"]),
        (
            sglang_core.SGLANGVisionModel,
            "Qwen2VLForConditionalGeneration",
            ["chat", "vision"],
        ),
    ],
)
def test_sglang_fp4_keeps_other_architectures_gated(
    monkeypatch, model_cls, architecture, abilities
):
    family = match_llm("qwen3.8", "fp4", 27, "FP4", "huggingface").copy(deep=True)
    family.architectures = [architecture]
    family.model_ability = abilities
    monkeypatch.setattr(model_cls, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(model_cls, "_is_linux", lambda: True)
    result = model_cls.match_json(family, family.model_specs[0], "FP4")
    assert result[0] is False
    assert "formats only" in result[1]


@pytest.mark.parametrize("has_gpu,is_linux", [(False, True), (True, False)])
def test_qwen_nvfp4_keeps_sglang_platform_guards(monkeypatch, has_gpu, is_linux):
    family = match_llm("qwen3.8", "fp4", 27, "FP4", "huggingface")
    model_cls = sglang_core.SGLANGVisionModel
    monkeypatch.setattr(model_cls, "_has_cuda_device", lambda: has_gpu)
    monkeypatch.setattr(model_cls, "_is_linux", lambda: is_linux)
    assert model_cls.match_json(family, family.model_specs[0], "FP4")[0] is False


@pytest.mark.parametrize(
    "model_name,size,model_cls",
    [("qwen3.8", 27, Qwen2VLChatModel), ("qwen3.8-max", 2400, PytorchChatModel)],
)
def test_transformers_preserves_custom_fpquant_models(model_name, size, model_cls):
    family = match_llm(model_name, "fp4", size, "FP4", "huggingface").copy(deep=True)
    family.is_builtin = False
    family.model_specs[0].model_id = "example/custom-fpquant"
    assert model_cls.match_json(family, family.model_specs[0], "FP4") is True
