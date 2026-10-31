# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux

import json
from types import SimpleNamespace

import pytest
from packaging.version import Version

from ..context_length import configure_auto_yarn
from ..llm_family import BUILTIN_LLM_FAMILIES, match_llm


@pytest.mark.parametrize("slots", [1, 2, 8])
@pytest.mark.parametrize("unified", [True, False])
def test_llama_yarn_uses_per_slot_window(slots, unified):
    family = match_llm("qwen3.8", "ggufv2", 27, "Q4_K_M")
    config = {"n_parallel": slots}
    config["context_length" if unified else "n_ctx"] = (
        524289 if unified else 524289 * slots
    )
    result, flag = configure_auto_yarn(family, "llama.cpp", config)
    assert result["rope_freq_scale"] == pytest.approx(1 / 3)
    assert result["yarn_orig_ctx"] == 262144
    assert result["rope_scaling_type"] == "yarn"
    assert flag is None
    assert "rope_freq_scale" not in config


@pytest.mark.parametrize(
    "engine,fmt,quant", [("MLX", "mlx", "4bit"), ("llama.cpp", "ggufv2", "Q4_K_M")]
)
@pytest.mark.parametrize("context", [None, 262144, 1000000, 1000001])
def test_additional_engines_obey_native_default_and_extension_limit(
    engine, fmt, quant, context
):
    family = match_llm("qwen3.8", fmt, 27, quant)
    assert family is not None
    config = {"context_length": context, "n_parallel": 2}
    if context == 1000001:
        with pytest.raises(ValueError, match="up to 1000000"):
            configure_auto_yarn(family, engine, config)
    else:
        result, flag = configure_auto_yarn(family, engine, config)
        if context in (None, 262144):
            assert result == config
        elif engine == "MLX":
            assert result["_xinference_auto_yarn"]["factor"] == 4
        else:
            assert result["rope_freq_scale"] == 0.25
        assert flag is None


@pytest.mark.parametrize(
    "engine,manual",
    [
        ("mlx", {"rope_scaling": {"type": "linear", "factor": 8}}),
        (
            "mlx",
            {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 8}}},
        ),
        ("llama.cpp", {"rope_freq_scale": 0.125}),
        ("llama.cpp", {"yarn_orig_ctx": 131072}),
        ("llama.cpp", {"rope_scaling_type": 2}),
    ],
)
def test_new_engines_preserve_manual_scaling(engine, manual):
    family = match_llm("qwen3.8", "pytorch", 27, "none")
    config = {"context_length": 2000000, **manual}
    assert configure_auto_yarn(family, engine, config) == (config, None)


def test_mlx_preserves_scaled_checkpoint(tmp_path):
    family = match_llm("qwen3.8", "mlx", 27, "4bit")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 8}}}
        )
    )
    config = {"context_length": 2000000}
    assert configure_auto_yarn(family, "MLX", config, str(tmp_path)) == (config, None)


@pytest.mark.parametrize("scaling", [None, "yarn", "linear"])
def test_gguf_checkpoint_scaling_is_read_before_automatic_override(tmp_path, scaling):
    from .test_collect_memory_catalog import gguf_header

    fields = {"general.architecture": "qwen35"}
    if scaling:
        fields["qwen35.rope.scaling.type"] = scaling
        fields["qwen35.rope.scaling.factor"] = 8
    path = tmp_path / "model.gguf"
    path.write_bytes(gguf_header(fields))
    family = match_llm("qwen3.8", "ggufv2", 27, "Q4_K_M")
    config = {"context_length": 2000000 if scaling else 1000000, "n_parallel": 2}
    result, flag = configure_auto_yarn(family, "llama.cpp", config, str(path))
    assert flag is None
    if scaling:
        assert result == config
    else:
        assert result["rope_freq_scale"] == 0.25


def test_gguf_invalid_metadata_does_not_silently_receive_rope_overrides(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"not a valid checkpoint")
    family = match_llm("qwen3.8", "ggufv2", 27, "Q4_K_M")
    with pytest.raises(ValueError, match="GGUF"):
        configure_auto_yarn(family, "llama.cpp", {"context_length": 1000000}, str(path))


@pytest.mark.parametrize(
    "engine,fmt,quant", [("MLX", "mlx", "4bit"), ("llama.cpp", "ggufv2", "Q4_K_M")]
)
def test_factory_applies_extension_before_mapping_slots(
    monkeypatch, engine, fmt, quant
):
    from .. import llm_family
    from ..core import create_llm_model_instance

    class DummyModel:
        def __init__(self, uid, family, path, config):
            self.config = config

    monkeypatch.setattr(
        llm_family, "check_engine_by_spec_parameters", lambda *a: DummyModel
    )
    model = create_llm_model_instance(
        "test-0",
        "qwen3.8",
        engine,
        model_format=fmt,
        model_size_in_billions=27,
        quantization=quant,
        model_path="/not-downloaded",
        enable_virtual_env=False,
        context_length=524289,
        n_parallel=8,
    )
    if engine == "MLX":
        assert model.config["_xinference_auto_yarn"]["factor"] == 3
    else:
        assert model.config["n_ctx"] == 524289 * 8
        assert model.config["rope_freq_scale"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("fmt", ["pytorch", "fp8", "fp4"])
@pytest.mark.parametrize("runtime", ["0.5.21", "0.5.22rc1", "0.5.22", "0.5.23"])
@pytest.mark.parametrize("virtualenv", [False, True])
def test_flash_next_sglang_version_gate(monkeypatch, fmt, runtime, virtualenv):
    from ..sglang import core

    family = next(
        f for f in BUILTIN_LLM_FAMILIES if f.model_name == "qwen3.8-Flash-Next"
    )
    spec = next(s for s in family.model_specs if s.model_format == fmt)
    monkeypatch.setattr(core.SGLANGVisionModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(core.SGLANGVisionModel, "_is_linux", lambda: True)
    monkeypatch.setattr(core, "SGLANG_INSTALLED", not virtualenv)
    monkeypatch.setattr(core, "_virtual_env_allows_missing_sglang", lambda: virtualenv)
    monkeypatch.setattr(core, "_get_sglang_version", lambda: Version(runtime))
    result = core.SGLANGVisionModel.match_json(family, spec, spec.quantization)
    if virtualenv or Version(runtime) >= Version("0.5.22"):
        assert result is True
    else:
        assert result[0] is False and "requires SGLang >= 0.5.22" in result[1]


def test_flash_next_sglang_virtualenv_and_defaults(monkeypatch):
    from ..sglang import core

    family = match_llm("qwen3.8-Flash-Next", "pytorch", 125, "none")
    assert 'sglang>=0.5.22 ; #engine# == "sglang"' in family.virtualenv.packages
    model = object.__new__(core.SGLANGModel)
    model.model_family = family
    model.model_spec = family.model_specs[0]
    model._n_worker = 1
    monkeypatch.setattr(model, "_get_cuda_count", lambda: 2)
    model._apply_draft_model = lambda config: None
    config = model._sanitize_model_config({})
    assert config["page_size"] == 64
    assert config["mamba_scheduler_strategy"] == "extra_buffer"
    assert config["mamba_track_interval"] == 64
    assert model._sanitize_model_config({"page_size": 128})["page_size"] == 128


def test_gguf_path_resolution_matches_runtime_variants(tmp_path):
    from ..gguf_model_metadata import resolve_gguf_model_path

    spec = SimpleNamespace(
        model_file_name_template="model-{quantization}.gguf",
        model_file_name_split_template=None,
        quantization_parts=None,
        quantization="Q4_K_M",
    )
    assert resolve_gguf_model_path(str(tmp_path), spec) == str(
        tmp_path / "model-Q4_K_M.gguf"
    )
    legacy = tmp_path / "model.bin"
    legacy.touch()
    assert resolve_gguf_model_path(str(tmp_path), spec) == str(legacy)
    spec.model_file_name_split_template = "part-{part}-{quantization}.gguf"
    spec.quantization_parts = {"Q4_K_M": ["00001", "00002"]}
    assert resolve_gguf_model_path(str(tmp_path), spec) == str(
        tmp_path / "part-00001-Q4_K_M.gguf"
    )
    direct = tmp_path / "selected.gguf"
    direct.touch()
    assert resolve_gguf_model_path(str(direct), spec) == str(direct)


def test_llama_automatic_yarn_reaches_native_parameters(monkeypatch, tmp_path):
    xllamacpp = pytest.importorskip("xllamacpp")
    from ..context_length import map_context_length
    from ..llama_cpp.core import XllamaCppModel

    family = match_llm("qwen3.8", "ggufv2", 27, "Q4_K_M")
    path = tmp_path / "model.gguf"
    path.touch()
    config, _ = configure_auto_yarn(
        family, "llama.cpp", {"context_length": 1000000, "n_parallel": 2}
    )
    config = map_context_length("llama.cpp", config)
    config["n_gpu_layers"] = 0
    model = XllamaCppModel("yarn-native-0", family, str(path), config)
    monkeypatch.setattr(model, "prepare_parse_tool_calls", lambda: None)
    monkeypatch.setattr(model, "prepare_parse_reasoning_content", lambda *a, **kw: None)

    class Captured(Exception):
        pass

    def server(params):
        assert params.n_ctx == 2000000
        assert params.n_parallel == 2
        assert params.rope_freq_scale == 0.25
        assert params.yarn_orig_ctx == 262144
        assert (
            params.rope_scaling_type
            == xllamacpp.llama_rope_scaling_type.LLAMA_ROPE_SCALING_TYPE_YARN
        )
        raise Captured

    monkeypatch.setattr(xllamacpp, "Server", server)
    with pytest.raises(Captured):
        model.load()


def test_mlx_checkpoint_and_explicit_yarn_precedence(tmp_path):
    from ..mlx.yarn import configured_yarn_parameters

    path = tmp_path / "config.json"
    checkpoint = {
        "text_config": {
            "max_position_embeddings": 262144,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "rope_theta": 10000000,
            },
        }
    }
    original = json.dumps(checkpoint)
    path.write_text(original)
    actual = configured_yarn_parameters(str(tmp_path), {}, None)
    assert actual["factor"] == 4
    assert actual["rope_theta"] == 10000000
    assert actual["original_max_position_embeddings"] == 262144
    overrides = {
        "rope_scaling": {
            "type": "yarn",
            "factor": 8,
            "original_max_position_embeddings": 131072,
        }
    }
    actual = configured_yarn_parameters(str(tmp_path), overrides, 262144)
    assert actual["factor"] == 8
    assert actual["original_max_position_embeddings"] == 131072
    overrides["text_config"] = {"unrelated": True}
    assert configured_yarn_parameters(str(tmp_path), overrides, 262144)["factor"] == 8
    assert (
        configured_yarn_parameters(str(tmp_path), {"rope_scaling": {}}, 262144) is None
    )
    assert path.read_text() == original
