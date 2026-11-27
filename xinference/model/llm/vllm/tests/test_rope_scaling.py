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
from inspect import signature
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from packaging import version

CHECKPOINT_THETA = 500000
QWEN3_VL_THETA = 5000000
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
AXES = {"mrope_section": [24, 20, 20], "mrope_interleaved": True}
QWEN3_VL_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "model_type": "qwen3_vl",
    "vision_config": {},
    "text_config": {
        **QWEN3_CONFIG,
        "model_type": "qwen3_vl_text",
        "hidden_size": 512,
        "head_dim": 128,
        "rope_theta": QWEN3_VL_THETA,
        "rope_scaling": dict(AXES, rope_type="default"),
    },
}
SCALING_FIELDS = {"factor": 4.0, "original_max_position_embeddings": 32768}
SCALING = {"rope_type": "yarn", **SCALING_FIELDS}
BASE_FIELDS = {"rope_theta": CHECKPOINT_THETA, "partial_rotary_factor": 0.5}
UNRELATED_OVERRIDE = {"max_position_embeddings": 65536}
YARN_ERROR = r"YaRN.*mrope_section.*>=0\.12\.0,<0\.17\.0.*0\.11\.2 or >=0\.17\.0"


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


def _rope_overrides(path: Path, checkpoint: dict, scaling: dict) -> dict:
    _write_checkpoint(path, checkpoint)
    model = _model(path, multimodal="text_config" in checkpoint)
    return model._sanitize_model_config({"rope_scaling": scaling})["hf_overrides"]


@pytest.fixture
def versions(monkeypatch):
    from .. import core

    def set_versions(transformers="5.22.2", *, patch_rope=False, vllm="0.34.0"):
        monkeypatch.setattr(core, "VLLM_VERSION", version.parse(vllm))
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
        pytest.param("5.22.2", False, False, id="native-dict"),
        pytest.param("5.22.2", False, True, id="native-json-no-theta"),
        pytest.param("4.57.6", True, False, id="normalized-dict-precedence"),
        pytest.param("4.57.6", True, True, id="normalized-json"),
        pytest.param("4.57.6", False, False, id="legacy-only"),
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
        checkpoint.update(
            rope_scaling={"type": "linear", "factor": 2.0, "beta_slow": 2.0},
            rope_parameters={"rope_type": "default", "factor": 3.0, "beta_slow": 1.0},
        )
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
        expected.update(expected_base, rope_scaling=expected_scaling)
    if not legacy or patch_rope:
        expected["rope_parameters"] = {**expected_base, **expected_scaling}
    assert "rope_scaling" not in result
    assert result["hf_overrides"] == expected
    assert (overrides, scaling) == original


@pytest.mark.parametrize(
    ("vllm", "transformers"), [("0.34.0", "5.22.2"), ("0.10.2", "4.57.6")]
)
def test_rope_scaling_resolves_separate_checkpoint_text_config(
    tmp_path, versions, vllm, transformers
):
    versions(transformers, vllm=vllm)
    weights = _write_checkpoint(tmp_path / "weights", QWEN3_CONFIG)
    mrope = dict(AXES, rope_theta=10000000, partial_rotary_factor=0.25)
    metadata = {"text_config": {"rope_parameters": dict(mrope, rope_type="default")}}
    config_path = str(_write_checkpoint(tmp_path / "metadata", metadata))
    overrides = {
        "vision_config": {"out_hidden_size": 64},
        "text_config": dict(UNRELATED_OVERRIDE),
    }
    if vllm == "0.10.2":
        overrides = dict(UNRELATED_OVERRIDE)

    result = _model(weights, multimodal=True)._sanitize_model_config(
        dict(hf_config_path=config_path, rope_scaling=SCALING, hf_overrides=overrides)
    )

    assert result["hf_config_path"] == config_path
    if vllm == "0.10.2":
        assert result["hf_overrides"] == {**overrides, "rope_scaling": SCALING}
        return
    assert result["hf_overrides"] == {
        "vision_config": {"out_hidden_size": 64},
        "text_config": {
            **UNRELATED_OVERRIDE,
            "rope_parameters": {**mrope, **SCALING},
        },
    }


@pytest.mark.parametrize("key", ["rope_type", "type"])
@pytest.mark.parametrize(
    ("vllm", "mode", "requested", "error"),
    [
        ("0.11.2", "mrope", "yarn", None),
        pytest.param("0.12.0", "mrope", "yarn", YARN_ERROR, id="0.12.0-mrope-yarn"),
        pytest.param("0.16.0", "mrope", "yarn", YARN_ERROR, id="0.16.0-mrope-yarn"),
        ("0.17.0", "mrope", "yarn", None),
        ("0.13.0", "mrope", None, None),
        ("0.13.0", "text", "yarn", None),
        pytest.param("0.13.0", "caller", "yarn", YARN_ERROR, id="0.13.0-caller-yarn"),
        ("0.34.0", "mrope", "linear", "linear.*mrope_section"),
        ("0.34.0", "text", "mrope", "mrope.*mrope_section"),
    ],
)
def test_mrope_compatibility(tmp_path, versions, key, vllm, mode, requested, error):
    versions("4.57.6", patch_rope=vllm != "0.11.2", vllm=vllm)
    nested = mode != "text"
    checkpoint = QWEN3_VL_CONFIG if nested else QWEN3_CONFIG
    model = _model(_write_checkpoint(tmp_path, checkpoint), multimodal=nested)
    target = dict(UNRELATED_OVERRIDE)
    overrides = {"text_config": target} if nested else target
    scaling = SCALING_FIELDS | ({key: requested} if requested else {})
    if mode == "caller":
        target["rope_scaling"] = scaling
        scaling = {"factor": SCALING["factor"]}
    launch = dict(hf_overrides=overrides, rope_scaling=scaling)
    if error:
        original = deepcopy(overrides)
        with pytest.raises(ValueError, match=error):
            model._sanitize_model_config(launch)
        assert launch["hf_overrides"] is overrides and overrides == original
        return
    result = model._sanitize_model_config(launch)["hf_overrides"]
    actual = result["text_config"] if nested else result
    expected = dict(SCALING, rope_type=requested or "default")
    assert actual["rope_scaling"] == ({**AXES, **expected} if nested else expected)
    assert {k: actual[k] for k in UNRELATED_OVERRIDE} == UNRELATED_OVERRIDE
    assert ("rope_parameters" in actual) == (vllm != "0.11.2")


@pytest.mark.parametrize("saved", ["openpangu", "mrope", "yarn", None])
def test_rope_scaling_factor_only_keeps_saved_type(tmp_path, versions, saved):
    versions("4.57.6", patch_rope=True, vllm="0.12.0")
    params = dict(AXES, rope_type=saved) if saved else {}
    if saved == "yarn":
        params.update(SCALING_FIELDS)
        with pytest.raises(ValueError, match=YARN_ERROR):
            _rope_overrides(tmp_path, {"rope_scaling": params}, {"factor": 2.0})
        versions("4.57.6", patch_rope=True, vllm="0.17.0")
    overrides = _rope_overrides(tmp_path, {"rope_scaling": params}, {"factor": 2.0})
    expected = dict(params, factor=2.0)
    if saved == "mrope":
        expected["rope_type"] = "default"
    assert overrides == {"rope_scaling": expected, "rope_parameters": expected}


@pytest.mark.parametrize(("vllm", "patch_rope"), [("0.11.2", False), ("0.12.0", True)])
@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_rope_scaling_syncs_retained_type(tmp_path, versions, vllm, patch_rope, nested):
    versions("4.57.6", vllm=vllm, patch_rope=patch_rope)
    saved = {"rope_scaling": {"type": "default"}}
    checkpoint = {"text_config": saved} if nested else saved
    overrides = _rope_overrides(tmp_path, checkpoint, dict(SCALING, type="linear"))
    target = overrides["text_config"] if nested else overrides
    assert target["rope_scaling"] == (dict(SCALING, type="yarn") if nested else SCALING)
    assert target.get("rope_parameters") == (SCALING if patch_rope else None)


@pytest.mark.parametrize("key", ["type", "rope_type"])
def test_rope_scaling_preserves_historical_root_spelling(tmp_path, versions, key):
    versions("4.40.2", vllm="0.4.3")
    scaling = {key: "linear", "factor": 2.0}
    checkpoint = {"model_type": "llama", "rope_theta": CHECKPOINT_THETA}
    assert _rope_overrides(tmp_path, checkpoint, scaling) == dict(
        rope_scaling=scaling, rope_theta=CHECKPOINT_THETA
    )


def test_rope_scaling_preserves_mllama_text_config_on_legacy_vllm(tmp_path, versions):
    from transformers import MllamaConfig, PretrainedConfig

    versions("4.57.6", vllm="0.10.2")
    config = MllamaConfig(
        text_config={"num_hidden_layers": 2, "cross_attention_layers": [1]}
    )
    scaling = {"rope_type": "linear", "factor": 2.0}
    overrides = _rope_overrides(tmp_path, config.to_dict(), scaling)

    config.update(overrides)
    assert isinstance(config.text_config, PretrainedConfig)
    config.update({"text_config": {"rope_scaling": scaling}})
    assert isinstance(config.text_config, dict)


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


@pytest.mark.parametrize("key", ["type", "rope_type"])
@pytest.mark.parametrize("requested", ["linear", "yarn", "mrope"])
def test_rope_scaling_legacy_rotary(tmp_path, key, requested):
    installed = version.parse(pytest.importorskip("vllm").__version__)
    if not version.parse("0.10.2") <= installed < version.parse("0.12.0"):
        pytest.skip("Requires the legacy get_rope API in vLLM 0.10.2–0.11.2")
    from vllm import config as vllm_config
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.transformers_utils.config import get_config, get_hf_text_config

    nested = requested != "linear"
    recursive = hasattr(vllm_config.ModelConfig, "_apply_dict_overrides")
    if nested and not recursive:
        pytest.skip("vLLM lacks recursive text_config dict overrides")
    root_checkpoint = dict(QWEN3_CONFIG, rope_scaling={"type": "default"})
    checkpoint = QWEN3_VL_CONFIG if nested else root_checkpoint
    scaling = {key: requested}
    if requested != "mrope":
        scaling["factor"] = 2.0
    if requested == "yarn":
        scaling["original_max_position_embeddings"] = 32768
    overrides = _rope_overrides(tmp_path, checkpoint, scaling)
    scalar_overrides = {k: v for k, v in overrides.items() if not isinstance(v, dict)}
    kwargs = scalar_overrides if recursive else overrides
    config = get_config(str(tmp_path), trust_remote_code=False, hf_overrides_kw=kwargs)
    if recursive:
        dict_overrides = {k: v for k, v in overrides.items() if isinstance(v, dict)}
        object.__new__(vllm_config.ModelConfig)._apply_dict_overrides(
            config, dict_overrides
        )
    text = get_hf_text_config(config)
    dim, length, base = text.head_dim, text.max_position_embeddings, text.rope_theta
    params = text.rope_scaling
    device = vllm_config.DeviceConfig(device="cpu")
    with vllm_config.set_current_vllm_config(
        vllm_config.VllmConfig(device_config=device)
    ):
        rotary = get_rope(dim, dim, length, base, True, params)
    assert rotary.base == (QWEN3_VL_THETA if nested else CHECKPOINT_THETA)
    if nested:
        assert type(rotary).__name__ == "MRotaryEmbedding"
        assert params["rope_type"] == ("default" if requested == "mrope" else "yarn")
        assert params.get("type", params["rope_type"]) == params["rope_type"]
        assert {key: params[key] for key in AXES} == AXES
        assert list(rotary.mrope_section) == AXES["mrope_section"]
        assert rotary.scaling_factor == (2.0 if requested == "yarn" else None)
    else:
        assert type(rotary).__name__ == "LinearScalingRotaryEmbedding"
        assert not recursive or "type" not in params


@pytest.mark.parametrize(
    ("mode", "rotary_class"),
    [
        ("text", "YaRNScalingRotaryEmbedding"),
        ("mrope", "MRotaryEmbedding"),
        ("openpangu", "MRotaryEmbeddingInterleaved"),
    ],
)
def test_rope_scaling_builds_rotary_with_checkpoint_base(tmp_path, mode, rotary_class):
    release = version.parse(pytest.importorskip("vllm").__version__)
    nested = mode != "text"
    checkpoint = deepcopy(QWEN3_VL_CONFIG) if nested else QWEN3_CONFIG
    scaling = dict(SCALING, original_max_position_embeddings=128)
    if mode == "openpangu":
        checkpoint["text_config"]["rope_scaling"]["rope_type"] = "openpangu"
        scaling = {"factor": 2.0}
    if mode == "mrope" and version.parse("0.12.0") <= release < version.parse("0.17.0"):
        with pytest.raises(ValueError, match=YARN_ERROR):
            _rope_overrides(tmp_path, checkpoint, scaling)
        return

    from vllm.transformers_utils import config as config_utils

    if not hasattr(config_utils, "set_default_rope_theta"):
        pytest.skip("vLLM lacks the rope_parameters API")
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.model_executor.layers.rotary_embedding import get_rope

    overrides = _rope_overrides(tmp_path, checkpoint, scaling)
    model_config = AsyncEngineArgs(
        model=str(tmp_path), hf_overrides=overrides
    ).create_model_config()
    config = model_config.hf_text_config
    config_utils.set_default_rope_theta(config, default_theta=1000000)
    params = config.rope_parameters
    kwargs = dict(max_position=config.max_position_embeddings, rope_parameters=params)
    if "rotary_dim" in signature(get_rope).parameters:
        kwargs["rotary_dim"] = config.head_dim
    with set_current_vllm_config(
        VllmConfig(model_config=model_config, device_config=DeviceConfig(device="cpu"))
    ):
        try:
            rotary = get_rope(config.head_dim, **kwargs)
        except ValueError as exc:
            if mode == "openpangu" and str(exc) == f"Unknown RoPE scaling type {mode}":
                pytest.skip(f"vLLM {release} does not support openpangu ({exc})")
            raise

    assert type(rotary).__name__ == rotary_class
    assert rotary.base == (QWEN3_VL_THETA if nested else CHECKPOINT_THETA)
    expected_type = "openpangu" if mode == "openpangu" else "yarn"
    assert params["rope_type"] == expected_type
    if nested:
        assert {key: params[key] for key in AXES} == AXES
    if mode != "openpangu":
        assert rotary.scaling_factor == SCALING["factor"]
