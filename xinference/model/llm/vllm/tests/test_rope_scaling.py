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

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging import version

CHECKPOINT_THETA = 500000
QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "rope_theta": CHECKPOINT_THETA,
    "rope_scaling": None,
}
SCALING = {
    "rope_type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
}
BASE_FIELDS = {"rope_theta": CHECKPOINT_THETA, "partial_rotary_factor": 0.5}
UNRELATED_OVERRIDE = {"max_position_embeddings": 65536}


def _write_checkpoint(path: Path, config: dict) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def _model(path: Path, *, multimodal: bool = False):
    from ..core import LLMFamilyV2, VLLMModel, VLLMMultiModel

    model = object.__new__(VLLMMultiModel if multimodal else VLLMModel)
    model.model_uid = "test-model-0"
    model.model_family = LLMFamilyV2(
        version=2,
        model_name="qwen3.5" if multimodal else "qwen3",
        model_lang=["en"],
        model_ability=["chat"],
        model_specs=[],
        architectures=[
            "Qwen3_5ForConditionalGeneration" if multimodal else "Qwen3ForCausalLM"
        ],
    )
    model.model_spec = SimpleNamespace(
        model_size_in_billions="0_6", model_format="pytorch"
    )
    model.model_path = str(path)
    model._device_count, model._n_worker = 1, 1  # type: ignore[assignment]
    model._address, model._shard = None, 0
    return model


@pytest.fixture
def versions(monkeypatch):
    from .. import core

    def set_versions(transformers="5.22.2", *, patch_rope=False):
        monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.34.0"))
        monkeypatch.setattr(
            core, "_get_transformers_version", lambda: version.parse(transformers)
        )
        if transformers.startswith("4."):
            module = ModuleType("vllm.transformers_utils.config")
            if patch_rope:
                module.patch_rope_parameters = object()
            monkeypatch.setitem(sys.modules, module.__name__, module)

    return set_versions


@pytest.mark.parametrize(
    ("transformers", "patch_rope", "as_json"),
    [
        ("5.22.2", False, False),
        ("5.22.2", False, True),
        ("4.57.6", True, False),
        ("4.57.6", True, True),
        ("4.57.6", False, False),
    ],
    ids=[
        "native-dict",
        "native-json-no-theta",
        "normalized-dict-precedence",
        "normalized-json",
        "legacy-only",
    ],
)
def test_rope_scaling_merges_checkpoint_caller_and_explicit_fields(
    tmp_path, versions, transformers, patch_rope, as_json
):
    versions(transformers, patch_rope=patch_rope)
    legacy = transformers.startswith("4.")
    checkpoint = {**QWEN3_CONFIG, **BASE_FIELDS}
    overrides = dict(UNRELATED_OVERRIDE)
    scaling = dict(SCALING)
    expected_base = dict(BASE_FIELDS)
    expected_scaling = dict(SCALING)
    if as_json and not legacy:
        checkpoint.pop("rope_theta")
        expected_base.pop("rope_theta")
    conflicting_inputs = legacy and patch_rope and not as_json
    if conflicting_inputs:
        checkpoint["rope_scaling"] = {"type": "linear", "factor": 2.0, "beta_slow": 2.0}
        checkpoint["rope_parameters"] = {
            "rope_type": "default",
            "factor": 3.0,
            "beta_slow": 1.0,
        }
        overrides.update(
            rope_theta=550000,
            rope_scaling={
                "type": "linear",
                "factor": 2.0,
                "rope_theta": 600000,
                "beta_fast": 8.0,
            },
            rope_parameters={"rope_type": "default", "factor": 3.0, "beta_fast": 16.0},
        )
        scaling["type"] = scaling.pop("rope_type")
        expected_base["rope_theta"] = 600000
        expected_scaling.update(beta_fast=16.0, beta_slow=1.0)
    original = deepcopy((overrides, scaling))
    launch = {"hf_overrides": overrides, "rope_scaling": scaling}
    if as_json:
        launch = {key: json.dumps(value) for key, value in launch.items()}

    result = _model(_write_checkpoint(tmp_path, checkpoint))._sanitize_model_config(
        launch
    )

    expected = dict(UNRELATED_OVERRIDE)
    if legacy:
        legacy_scaling = dict(expected_scaling)
        if conflicting_inputs:
            legacy_scaling["type"] = legacy_scaling.pop("rope_type")
        expected.update(expected_base, rope_scaling=legacy_scaling)
    if not legacy or patch_rope:
        expected["rope_parameters"] = {**expected_base, **expected_scaling}
    assert "rope_scaling" not in result
    assert result["hf_overrides"] == expected
    assert (overrides, scaling) == original


def test_rope_scaling_resolves_separate_checkpoint_text_config(tmp_path, versions):
    versions()
    weights = _write_checkpoint(
        tmp_path / "weights", dict(QWEN3_CONFIG, rope_theta=700000)
    )
    mrope = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    }
    config_path = str(
        _write_checkpoint(
            tmp_path / "metadata",
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "rope_parameters": dict(mrope, rope_type="default"),
                },
                "vision_config": {},
            },
        )
    )
    overrides = {
        "vision_config": {"out_hidden_size": 64},
        "text_config": dict(UNRELATED_OVERRIDE),
    }

    result = _model(weights, multimodal=True)._sanitize_model_config(
        dict(hf_config_path=config_path, rope_scaling=SCALING, hf_overrides=overrides)
    )

    assert result["hf_config_path"] == config_path
    assert result["hf_overrides"] == {
        "vision_config": {"out_hidden_size": 64},
        "text_config": {
            **UNRELATED_OVERRIDE,
            "rope_parameters": {**mrope, **SCALING},
        },
    }


def test_rope_scaling_leaves_per_layer_maps_to_vllm(tmp_path, versions):
    versions()
    checkpoint = {
        "model_type": "gemma3_text",
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 500000},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000},
        },
    }
    result = _model(_write_checkpoint(tmp_path, checkpoint))._sanitize_model_config(
        {"hf_overrides": UNRELATED_OVERRIDE, "rope_scaling": SCALING}
    )
    assert result["hf_overrides"] == {**UNRELATED_OVERRIDE, "rope_scaling": SCALING}


def test_empty_rope_scaling_keeps_hf_overrides(tmp_path, versions):
    versions()
    overrides = json.dumps(UNRELATED_OVERRIDE)
    result = _model(tmp_path)._sanitize_model_config(
        {"hf_overrides": overrides, "rope_scaling": "{}"}
    )
    assert "rope_scaling" not in result
    assert result["hf_overrides"] == overrides


def test_rope_scaling_builds_yarn_with_checkpoint_base(tmp_path, monkeypatch):
    vllm = pytest.importorskip("vllm")
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
        YaRNScalingRotaryEmbedding,
    )
    from vllm.transformers_utils.config import set_default_rope_theta

    from .. import core

    monkeypatch.setattr(core, "VLLM_VERSION", version.parse(vllm.__version__))
    model = _model(_write_checkpoint(tmp_path, QWEN3_CONFIG))
    scaling = dict(SCALING, original_max_position_embeddings=128)
    overrides = model._sanitize_model_config({"rope_scaling": scaling})["hf_overrides"]
    model_config = AsyncEngineArgs(
        model=model.model_path, hf_overrides=overrides
    ).create_model_config()
    config = model_config.hf_config
    set_default_rope_theta(config, default_theta=1000000)  # Qwen3DecoderLayer
    with set_current_vllm_config(
        VllmConfig(model_config=model_config, device_config=DeviceConfig(device="cpu"))
    ):
        rotary = get_rope(
            config.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

    assert isinstance(rotary, YaRNScalingRotaryEmbedding)
    assert rotary.base == CHECKPOINT_THETA
