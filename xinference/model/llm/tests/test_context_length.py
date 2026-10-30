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


from types import SimpleNamespace

import pytest

from ..context_length import (
    MAX_CONTEXT_LENGTH,
    map_context_length,
    validate_context_length,
    validate_context_length_launch,
)


@pytest.mark.parametrize(
    "engine,key",
    [
        ("vLLM", "max_model_len"),
        ("SGLang", "context_length"),
        ("Transformers", "context_length"),
        ("MLX", "context_length"),
        ("TRT-LLM", "max_seq_len"),
        ("llama.cpp", "n_ctx"),
    ],
)
def test_factory_passes_unified_context_to_each_engine(monkeypatch, engine, key):
    from .. import llm_family
    from ..core import create_llm_model_instance

    spec = SimpleNamespace(
        model_format="pytorch", model_size_in_billions=4, quantization="none"
    )
    family = SimpleNamespace(model_name="test", model_specs=[spec])

    class DummyModel:
        def __init__(self, uid, family, path, config):
            self.config = config

    monkeypatch.setattr(llm_family, "match_llm", lambda *args: family)
    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: DummyModel
    )
    kwargs = {"context_length": 32768, "enable_virtual_env": False}
    if engine == "llama.cpp":
        kwargs["n_parallel"] = 3
    model = create_llm_model_instance(
        "test-0", "test", engine, model_path="/already-cached", **kwargs
    )
    assert model.config[key] == 32768 * (3 if engine == "llama.cpp" else 1)
    assert ("context_length" in model.config) == (key == "context_length")
    assert kwargs["context_length"] == 32768


@pytest.mark.parametrize(
    "engine,key",
    [("vLLM", "max_model_len"), ("llama.cpp", "n_ctx")],
)
def test_native_context_defaults_and_explicit_aliases_are_preserved(engine, key):
    config = {key: 8192, "temperature": 0.5}
    assert map_context_length(engine, config) == config
    assert map_context_length(engine, {"context_length": None, **config}) == config
    matching = {"context_length": 8192, **config}
    if engine == "llama.cpp":
        matching["n_parallel"] = 1
    assert map_context_length(engine, matching)[key] == 8192
    with pytest.raises(ValueError, match="conflicts"):
        map_context_length(engine, {**matching, "context_length": 4096})


@pytest.mark.parametrize("cpus,slots", [(None, 1), (2, 2), (32, 8)])
def test_llama_cpp_maps_per_request_context_to_slot_capacity(monkeypatch, cpus, slots):
    monkeypatch.setattr("os.cpu_count", lambda: cpus)
    config = map_context_length("llama.cpp", {"context_length": 32768})
    assert config == {"n_ctx": 32768 * slots, "n_parallel": slots}
    with pytest.raises(ValueError, match="exceeds"):
        map_context_length(
            "llama.cpp", {"context_length": MAX_CONTEXT_LENGTH, "n_parallel": 2}
        )


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, "32768", MAX_CONTEXT_LENGTH + 1]
)
def test_invalid_context_is_rejected(value):
    with pytest.raises(ValueError, match="context_length"):
        validate_context_length(value)


def test_invalid_context_and_aliases_fail_before_discovery_or_download(monkeypatch):
    from .. import llm_family
    from ..core import create_llm_model_instance

    def no_discovery(*args):
        raise AssertionError("invalid context must fail before model discovery")

    monkeypatch.setattr(llm_family, "match_llm", no_discovery)
    with pytest.raises(ValueError, match="conflicts"):
        create_llm_model_instance(
            "test", "test", "vLLM", context_length=8192, max_model_len=4096
        )
    with pytest.raises(ValueError, match="context_length"):
        create_llm_model_instance("test", "test", "vLLM", context_length=0)
    with pytest.raises(ValueError, match="positive integer"):
        create_llm_model_instance(
            "test", "test", "llama.cpp", context_length=8192, n_parallel=False
        )


def test_context_only_applies_to_supported_llm_engines():
    with pytest.raises(ValueError, match="only supported for LLM"):
        validate_context_length_launch(
            "embedding", "sentence_transformers", {"context_length": 32768}
        )
    with pytest.raises(ValueError, match="not supported"):
        map_context_length("custom", {"context_length": 32768})
    assert map_context_length("custom", {"native_option": 1}) == {"native_option": 1}


def test_mlx_runtime_override_wins_over_nested_architecture_context():
    from ..utils import get_context_length_from_config

    config = {
        "max_position_embeddings": 262144,
        "text_config": {"max_position_embeddings": 262144},
    }
    assert get_context_length_from_config(config) == 262144
    assert get_context_length_from_config(config, context_length=32768) == 32768
    assert config["text_config"]["max_position_embeddings"] == 262144
    with pytest.raises(ValueError):
        get_context_length_from_config(config, context_length=True)


@pytest.mark.parametrize(
    "name,size",
    [
        ("qwen3.5", 9),
        ("qwen3.6", 35),
        ("qwen3.8", 27),
        ("qwen3.8-max", 2400),
        ("qwen3.8-Flash-Next", 125),
    ],
)
@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
@pytest.mark.parametrize(
    "context,factor",
    [
        (None, None),
        (131072, None),
        (262144, None),
        (262145, 2.0),
        (524288, 2.0),
        (524289, 3.0),
        (1000000, 4.0),
    ],
)
def test_auto_yarn_follows_declared_recipe(name, size, engine, context, factor):
    import json

    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm(name, "pytorch", size, "none", "huggingface")
    original = {"context_length": context}
    config, flag = configure_auto_yarn(family, engine, original)
    unsupported = name == "qwen3.8-Flash-Next" and engine == "SGLang"
    if factor is None or unsupported:
        assert config == original
        assert flag is None
    else:
        key = "hf_overrides" if engine == "vLLM" else "json_model_override_args"
        overrides = config[key] if engine == "vLLM" else json.loads(config[key])
        for part in family.context_length_extension.rope_parameters_path:
            overrides = overrides[part]
        assert overrides["rope_type"] == "yarn"
        assert overrides["factor"] == factor
        assert overrides["original_max_position_embeddings"] == 262144
        assert flag == (
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
            if engine == "vLLM"
            else "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
        )
    assert original == {"context_length": context}


@pytest.mark.parametrize(
    "engine,key", [("vLLM", "hf_overrides"), ("SGLang", "json_model_override_args")]
)
@pytest.mark.parametrize("rope", ["rope_parameters", "rope_scaling"])
@pytest.mark.parametrize("nested", [False, True])
def test_auto_yarn_preserves_manual_rope_settings(engine, key, rope, nested):
    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none", "huggingface")
    overrides = {rope: {"rope_type": "yarn", "factor": 8.0}}
    if nested:
        overrides = {"text_config": overrides}
    # Explicit recipes can intentionally go beyond the automatic limit.
    config = {"context_length": 2000000, key: overrides}
    assert configure_auto_yarn(family, engine, config) == (config, None)


def test_auto_yarn_merges_overrides_without_mutating_them():
    from copy import deepcopy

    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none", "huggingface")
    config = {
        "max_model_len": 1000000,
        "hf_overrides": {"text_config": {"attention_dropout": 0}, "custom": 123},
    }
    original = deepcopy(config)
    result, _ = configure_auto_yarn(family, "vLLM", config)
    assert result["hf_overrides"]["custom"] == 123
    assert result["hf_overrides"]["text_config"]["attention_dropout"] == 0
    assert config == original


def test_auto_yarn_does_not_override_a_scaled_checkpoint(tmp_path):
    import json

    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none", "huggingface")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 8.0}}}
        )
    )
    config = {"context_length": 1000000}
    assert configure_auto_yarn(family, "vLLM", config, str(tmp_path)) == (config, None)


@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
def test_auto_yarn_limits_only_verified_models_and_engines(engine):
    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none", "huggingface")
    with pytest.raises(ValueError, match="up to 1000000"):
        configure_auto_yarn(family, engine, {"context_length": 1000001})
    config = {"context_length": 1000000}
    assert configure_auto_yarn(family, "MLX", config) == (config, None)
    family.context_length_extension = None
    assert configure_auto_yarn(family, engine, config) == (config, None)


@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
def test_factory_applies_auto_yarn(monkeypatch, engine):
    import json

    from .. import llm_family
    from ..core import create_llm_model_instance

    class DummyModel:
        def __init__(self, uid, family, path, config):
            self.config = config

    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *args: DummyModel
    )
    model = create_llm_model_instance(
        "test-0",
        "qwen3.8",
        engine,
        model_format="pytorch",
        model_size_in_billions=27,
        quantization="none",
        model_path="/already-cached",
        enable_virtual_env=False,
        context_length=1000000,
    )
    key = "hf_overrides" if engine == "vLLM" else "json_model_override_args"
    overrides = model.config[key] if engine == "vLLM" else json.loads(model.config[key])
    assert overrides["text_config"]["rope_parameters"]["factor"] == 4.0
    assert model._auto_yarn_env == (
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
        if engine == "vLLM"
        else "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"
    )


@pytest.mark.parametrize(
    "engine,flag",
    [
        ("vLLM", "VLLM_ALLOW_LONG_MAX_MODEL_LEN"),
        ("SGLang", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"),
    ],
)
@pytest.mark.parametrize("context", [262144, 1000000])
@pytest.mark.parametrize("explicit_flag", [None, "0"])
def test_worker_persists_resolved_yarn_flag_without_changing_parent_environment(
    engine, flag, context, explicit_flag
):
    import os

    from xinference.core.worker import _inject_auto_yarn_env

    original_env = {"KEEP": "value"}
    if explicit_flag is not None:
        original_env[flag] = explicit_flag
    parent_flag = os.environ.get(flag)

    launch_args = {"context_length": context, "envs": original_env}
    model = SimpleNamespace(_auto_yarn_env=flag if context > 262144 else None)
    envs = _inject_auto_yarn_env(model, original_env, launch_args)
    assert envs.get(flag) == (
        explicit_flag
        if explicit_flag is not None
        else ("1" if context > 262144 else None)
    )
    assert envs["KEEP"] == "value"
    assert launch_args == {"context_length": context, "envs": envs}
    assert os.environ.get(flag) == parent_flag
    assert "1" not in original_env.values()


@pytest.mark.parametrize("engine", ["vLLM", "SGLang"])
@pytest.mark.parametrize("scaled", [False, True])
def test_downloaded_checkpoint_controls_yarn_overrides_and_environment(
    monkeypatch, tmp_path, engine, scaled
):
    import json

    from xinference.core.worker import _inject_auto_yarn_env

    from .. import llm_family
    from ..cache_manager import LLMCacheManager
    from ..core import create_llm_model_instance

    class DummyModel:
        def __init__(self, uid, family, path, config):
            self.config = config

    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *a: DummyModel
    )
    monkeypatch.setattr(LLMCacheManager, "cache", lambda self: str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "rope_parameters": {"rope_type": "yarn" if scaled else "default"}
                }
            }
        )
    )
    args = dict(
        model_format="pytorch",
        model_size_in_billions=27,
        quantization="none",
        enable_virtual_env=False,
        context_length=1000001 if scaled else 1000000,
    )
    model = create_llm_model_instance("test-0", "qwen3.8", engine, **args)
    envs = _inject_auto_yarn_env(model, {}, {})
    override = "hf_overrides" if engine == "vLLM" else "json_model_override_args"
    assert (override in model.config) is not scaled
    assert bool(envs) is not scaled
    if not scaled:
        with pytest.raises(ValueError, match="up to 1000000"):
            create_llm_model_instance(
                "test-0", "qwen3.8", engine, **{**args, "context_length": 1000001}
            )


@pytest.mark.parametrize(
    "rope",
    [
        {"type": "mrope", "mrope_section": [11, 11, 10]},
        {"rope_type": "default", "mrope_interleaved": True},
        {"rope_type": "unknown", "factor": 1},
    ],
)
def test_unscaled_checkpoint_still_receives_yarn(tmp_path, rope):
    import json

    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none")
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"rope_parameters": rope}})
    )
    config, flag = configure_auto_yarn(
        family, "vllm", {"context_length": 1000000}, str(tmp_path)
    )
    assert flag == "VLLM_ALLOW_LONG_MAX_MODEL_LEN"
    assert config["hf_overrides"]["text_config"]["rope_parameters"]["factor"] == 4


@pytest.mark.parametrize(
    "engine,key", [("vllm", "hf_overrides"), ("sglang", "json_model_override_args")]
)
def test_json_overrides_merge_and_invalid_values_fail(engine, key):
    import json

    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none")
    original = {"context_length": 1000000, key: '{"text_config": {"keep": 1}}'}
    result, _ = configure_auto_yarn(family, engine, original)
    overrides = result[key] if engine == "vllm" else json.loads(result[key])
    assert overrides["text_config"]["keep"] == 1
    assert overrides["text_config"]["rope_parameters"]["factor"] == 4
    for invalid in ("not-json", "[]", "null", "1", [1]):
        with pytest.raises(ValueError, match="JSON object"):
            configure_auto_yarn(family, engine, {**original, key: invalid})
    with pytest.raises(ValueError, match="configuration path"):
        configure_auto_yarn(family, engine, {**original, key: {"text_config": 1}})
    for manual in (
        {key: lambda config: config},
        {"rope_scaling": {"type": "linear", "factor": 4}},
    ):
        config = {"context_length": 2000000, **manual}
        assert configure_auto_yarn(family, engine, config) == (config, None)


@pytest.mark.parametrize("original,factor", [(131072, 8), (None, 4)])
def test_yarn_factor_uses_recipe_original_window(original, factor):
    from ..context_length import configure_auto_yarn
    from ..llm_family import match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none").copy(deep=True)
    parameters = family.context_length_extension.rope_parameters
    if original is None:
        parameters.pop("original_max_position_embeddings")
    else:
        parameters["original_max_position_embeddings"] = original
    result, _ = configure_auto_yarn(family, "vllm", {"context_length": 1000000})
    rope = result["hf_overrides"]["text_config"]["rope_parameters"]
    assert rope["factor"] == factor
    assert rope["original_max_position_embeddings"] * factor >= 1000000


@pytest.mark.parametrize(
    "patch",
    [
        {"engines": []},
        {"engines": ["vllm", "vllm"]},
        {"rope_parameters_path": []},
        {"rope_parameters_path": [""]},
        {"rope_parameters": {"original_max_position_embeddings": 0}},
        {"rope_parameters": {"original_max_position_embeddings": "262144"}},
        {"max_context_length": 262144},
        {"max_context_length": 131072},
    ],
)
def test_custom_registration_validates_context_extension(patch):
    from ..llm_family import LLMFamilyV2, match_llm

    family = match_llm("qwen3.8", "pytorch", 27, "none").dict()
    family["context_length_extension"].update(patch)
    with pytest.raises(ValueError):
        LLMFamilyV2.parse_obj(family)
