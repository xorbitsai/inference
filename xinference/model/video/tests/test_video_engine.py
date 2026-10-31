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

from unittest.mock import Mock, patch

import pytest

from ....core.utils import filter_virtualenv_packages_by_markers
from ...core import create_model_instance
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import BUILTIN_VIDEO_MODELS, VIDEO_MODEL_DESCRIPTIONS, _install
from ..cache_manager import VideoCacheManager
from ..core import (
    create_video_model_instance,
    match_diffusion,
    resolve_video_model_name_and_engine,
)
from ..engine import (
    MLX_VIDEO_MODEL_NAMES,
    DiffusersVideoEngineModel,
    MLXVideoEngineModel,
)
from ..engine_family import (
    VIDEO_ENGINES,
    check_engine_by_model_name_and_engine,
    check_engine_by_model_name_and_engine_with_virtual_env,
)
from ..native import _NativeVideoModel


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    with (
        patch.object(MLXVideoEngineModel, "_is_apple_silicon", return_value=True),
        patch("xinference.model.video.engine.sys.version_info", (3, 11)),
    ):
        _install()
        yield
    _install()


def test_builtin_video_models_register_expected_engines():
    assert set(VIDEO_ENGINES) == set(BUILTIN_VIDEO_MODELS)
    for model_name, engines in VIDEO_ENGINES.items():
        expected_engines = []
        if any(
            family.engine == "diffusers" for family in BUILTIN_VIDEO_MODELS[model_name]
        ):
            expected_engines.append("diffusers")
        if model_name in MLX_VIDEO_MODEL_NAMES:
            expected_engines.append("MLX")
        if _NativeVideoModel.check_host() is True:
            expected_engines.extend(
                engine
                for engine in ("vLLM", "SGLang")
                if any(
                    family.engine == engine
                    for family in BUILTIN_VIDEO_MODELS[model_name]
                )
            )
        assert list(engines) == expected_engines

        if "diffusers" in engines:
            assert engines["diffusers"] == [
                {
                    "model_name": model_name,
                    "model_format": "diffusers",
                    "quantization": "none",
                    "video_class": DiffusersVideoEngineModel,
                    "default_model_config": next(
                        family.default_model_config
                        for family in BUILTIN_VIDEO_MODELS[model_name]
                        if family.engine == "diffusers"
                    ),
                    **(
                        {
                            "runtime_precisions": ["bf16", "fp8", "int8", "int4"],
                            "default_runtime_precision": "int4",
                        }
                        if model_name == "MiniMax-H3"
                        else {}
                    ),
                }
            ]
        if "MLX" in engines:
            assert engines["MLX"] == [
                {
                    "model_name": model_name,
                    "model_format": "mlx",
                    "quantization": "none",
                    "video_class": MLXVideoEngineModel,
                }
            ]


def test_video_engine_lookup_is_case_insensitive():
    assert (
        check_engine_by_model_name_and_engine("DIFFUSERS", "MiniMax-H3")
        is DiffusersVideoEngineModel
    )
    assert resolve_video_model_name_and_engine(
        "MiniMax-H3", use_default_engine=True
    ) == ("MiniMax-H3", "diffusers")
    assert (
        check_engine_by_model_name_and_engine("mlx", "LTX-2-distilled")
        is MLXVideoEngineModel
    )
    assert resolve_video_model_name_and_engine(
        "Wan2.1-1.3B", "mlx", use_default_engine=True
    ) == ("Wan2.1-1.3B", "MLX")


def test_create_video_model_instance_records_default_engine():
    model = create_video_model_instance(
        "uid",
        "MiniMax-H3",
        model_path="/fake/path",
        enable_virtual_env=False,
    )

    assert isinstance(model, DiffusersVideoEngineModel)
    assert model.model_family.model_engine == "diffusers"
    assert model.model_family.to_description()["model_engine"] == "diffusers"


def test_create_mlx_video_model_instance():
    model = create_video_model_instance(
        "uid",
        "LTX-2-distilled",
        model_path="/fake/path",
        model_engine="mlx",
        model_format="mlx",
        quantization="none",
        enable_virtual_env=False,
    )

    assert isinstance(model, MLXVideoEngineModel)
    assert model.model_family.model_engine == "MLX"
    assert model.model_family.cache_name == "LTX-2-distilled-mlx"


def test_invalid_video_engine_is_rejected_before_download():
    with patch.object(VideoCacheManager, "cache") as cache:
        with pytest.raises(ValueError, match="cannot be run on engine"):
            create_video_model_instance(
                "uid",
                "MiniMax-H3",
                model_engine="not-an-engine",
                enable_virtual_env=False,
            )
    cache.assert_not_called()


@pytest.mark.parametrize(
    ("model_name", "message"),
    [
        pytest.param(
            "Wan2.1-1.3B",
            "Model Wan2.1-1.3B does not support lightning acceleration",
            id="model-without-lightning",
        ),
        pytest.param(
            "MiniMax-H3",
            "Cannot support lightning version not-a-version, available lightning "
            "versions: ['4step_v0.1', '8step_v1.0_bf16', '4step_v1.0_768p_bf16']",
            id="unknown-version",
        ),
    ],
)
def test_unsupported_video_lightning_is_rejected_before_download(model_name, message):
    cache = Mock(side_effect=AssertionError("Invalid lightning reached caching"))
    with (
        patch.object(VideoCacheManager, "cache", cache),
        patch.object(VideoCacheManager, "cache_lightning", cache),
    ):
        with pytest.raises(ValueError) as exc_info:
            create_video_model_instance(
                "uid",
                model_name,
                lightning_version="not-a-version",
                enable_virtual_env=False,
            )
    assert str(exc_info.value) == message
    cache.assert_not_called()


@pytest.mark.parametrize("lightning_model_path", [None, "/mine/lightning.safetensors"])
def test_video_still_caches_the_model_then_the_lightning_file(lightning_model_path):
    calls = []

    def fake_cache(cache_manager):
        calls.append("model")
        return "/cache/model"

    def fake_cache_lightning(cache_manager, lightning_version):
        calls.append("lightning " + lightning_version)
        return "/cache/lightning.safetensors"

    with (
        patch.object(VideoCacheManager, "cache", fake_cache),
        patch.object(VideoCacheManager, "cache_lightning", fake_cache_lightning),
    ):
        model = create_video_model_instance(
            "uid",
            "MiniMax-H3",
            lightning_version="not-a-version" if lightning_model_path else "4step_v0.1",
            lightning_model_path=lightning_model_path,
            enable_virtual_env=False,
        )

    assert model._model_path == "/cache/model"
    if lightning_model_path:
        # An explicit path is used as given: the version is not consulted.
        assert calls == ["model"]
        assert model._lightning_model_path == lightning_model_path
    else:
        assert calls == ["model", "lightning 4step_v0.1"]
        assert model._lightning_model_path == "/cache/lightning.safetensors"


@pytest.mark.parametrize("download_hub", ["modelscope", "auto", None])
def test_hf_only_mlx_video_rejects_modelscope_before_download(
    monkeypatch, download_hub
):
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")
    with patch.object(VideoCacheManager, "cache") as cache:
        with pytest.raises(ValueError, match="does not provide a modelscope source"):
            create_video_model_instance(
                "uid",
                "Wan2.1-1.3B",
                model_engine="MLX",
                download_hub=download_hub,
                enable_virtual_env=True,
            )
    cache.assert_not_called()


@pytest.mark.parametrize(
    ("model_format", "quantization"),
    [("diffusers", "none"), ("mlx", "q8_0")],
)
def test_virtualenv_fallback_preserves_explicit_video_tuple(model_format, quantization):
    model_spec = match_diffusion(
        "Wan2.1-1.3B", download_hub="huggingface", model_engine="MLX"
    )

    with pytest.raises(ValueError, match="with format"):
        check_engine_by_model_name_and_engine_with_virtual_env(
            "MLX",
            model_spec.model_name,
            model_format,
            quantization,
            model_family=model_spec,
        )


def test_generic_factory_forwards_video_engine():
    with patch(
        "xinference.model.video.core.create_video_model_instance"
    ) as create_video:
        create_model_instance(
            "uid",
            "video",
            "MiniMax-H3",
            "diffusers",
            model_format="diffusers",
            quantization="none",
            download_hub="huggingface",
        )

    create_video.assert_called_once_with(
        "uid",
        "MiniMax-H3",
        "huggingface",
        None,
        model_engine="diffusers",
        model_format="diffusers",
        quantization="none",
    )


def test_video_engine_api_returns_diffusers_format():
    with patch.object(DiffusersVideoEngineModel, "check_lib", return_value=True):
        params = get_engine_params_by_name(
            "video", "MiniMax-H3", enable_virtual_env=False
        )

    assert set(params) == {"diffusers", "vLLM", "SGLang"}
    assert params["diffusers"] == [
        {
            "model_name": "MiniMax-H3",
            "model_format": "diffusers",
            "quantization": "none",
            "runtime_precisions": ["bf16", "fp8", "int8", "int4"],
            "default_runtime_precision": "int4",
            "default_model_config": {
                "quantization": "int4",
                "group_offload": True,
                "torch_dtype": "bfloat16",
            },
        }
    ]


def test_video_engine_uses_virtualenv_when_dependency_is_missing():
    with patch.object(
        DiffusersVideoEngineModel,
        "check_lib",
        return_value=(False, "diffusers is not installed"),
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "video", "MiniMax-H3", enable_virtual_env=True
        )

    assert params["diffusers"][0]["virtualenv_required"] is True
    assert params["diffusers"][0]["runtime_precisions"] == [
        "bf16",
        "fp8",
        "int8",
        "int4",
    ]


def test_mlx_video_engine_uses_virtualenv_when_dependency_is_missing():
    with (
        patch("xinference.model.utils.sys.platform", "darwin"),
        patch.object(
            MLXVideoEngineModel,
            "check_lib",
            return_value=(False, "Blaizzy/mlx-video is not installed"),
        ),
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "video", "LTX-2-distilled", enable_virtual_env=True
        )

    assert params["MLX"][0]["virtualenv_required"] is True


def test_mlx_video_engine_old_python_is_not_virtualenv_repairable():
    with (
        patch("xinference.model.utils.sys.platform", "darwin"),
        patch.object(MLXVideoEngineModel, "_is_apple_silicon", return_value=True),
        patch("xinference.model.video.engine.sys.version_info", (3, 10)),
    ):
        params = get_engine_params_by_name_with_virtual_env(
            "video", "LTX-2-distilled", enable_virtual_env=True
        )

    assert params["MLX"] == "Blaizzy/mlx-video requires Python 3.11 or newer"


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_mlx_video_engine_incompatible_host_is_rejected_before_download(
    enable_virtual_env,
):
    with (
        patch.object(MLXVideoEngineModel, "_is_apple_silicon", return_value=False),
        patch.object(VideoCacheManager, "cache") as cache,
        pytest.raises(ValueError, match="requires Apple Silicon"),
    ):
        create_video_model_instance(
            "uid",
            "LTX-2-distilled",
            model_engine="MLX",
            enable_virtual_env=enable_virtual_env,
        )

    cache.assert_not_called()


def test_video_specs_scope_diffusers_dependency_to_engine():
    for families in BUILTIN_VIDEO_MODELS.values():
        for family in families:
            if family.engine != "diffusers":
                continue
            assert family.engine == "diffusers"
            assert family.model_format == "diffusers"
            assert family.virtualenv is not None
            diffusers_packages = [
                package
                for package in family.virtualenv.packages
                if "diffusers" in package.lower()
            ]
            assert diffusers_packages
            assert all(
                '#engine# == "diffusers"' in package for package in diffusers_packages
            )
            assert filter_virtualenv_packages_by_markers(
                family.virtualenv.packages, "other", cuda_version=None
            ) == [
                package
                for package in family.virtualenv.packages
                if "diffusers" not in package.lower()
            ]


def test_mlx_video_specs_are_pinned_and_isolated():
    mlx_specs = [
        family
        for families in BUILTIN_VIDEO_MODELS.values()
        for family in families
        if family.engine == "MLX"
    ]

    assert {family.model_name for family in mlx_specs} == MLX_VIDEO_MODEL_NAMES
    assert len({family.cache_name for family in mlx_specs}) == len(
        MLX_VIDEO_MODEL_NAMES
    )
    for family in mlx_specs:
        assert family.model_format == "mlx"
        if family.model_hub == "huggingface":
            assert family.model_revision not in (None, "main")
        else:
            assert family.model_hub == "modelscope"
            assert family.model_revision == "master"
        assert family.cache_name.endswith("-mlx")
        assert family.virtualenv is not None
        mlx_packages = [
            package
            for package in family.virtualenv.packages
            if package.startswith("mlx-video @")
        ]
        assert len(mlx_packages) == 1
        assert "Blaizzy/mlx-video.git@87db56a" in mlx_packages[0]
        assert '#engine# == "MLX"' in mlx_packages[0]

    wan21_specs = [
        family for family in mlx_specs if family.model_name.startswith("Wan2.1-")
    ]
    assert all(
        '#system_torch# ; #engine# == "MLX"' in family.virtualenv.packages
        for family in wan21_specs
    )
    ltx23_specs = [
        family
        for family in mlx_specs
        if family.model_family == "LTX-2.3" and family.model_hub == "huggingface"
    ]
    assert all(
        family.text_encoder_model_id == "prince-canuma/LTX-2-distilled"
        for family in ltx23_specs
    )
    modelscope_ltx23_specs = [
        family
        for family in mlx_specs
        if family.model_family == "LTX-2.3" and family.model_hub == "modelscope"
    ]
    assert all(
        family.text_encoder_model_id == "Xorbits/LTX-2-distilled"
        for family in modelscope_ltx23_specs
    )


@pytest.mark.parametrize(
    ("model_name", "model_id"),
    [
        ("Wan2.2-A14B", "Xorbits/wan2.2-t2v-a14b-mlx"),
        ("Wan2.2-i2v-A14B", "Xorbits/wan2.2-i2v-a14b-mlx"),
        ("Wan2.2-ti2v-5B", "Xorbits/wan2.2-ti2v-5b-mlx"),
        ("LTX-2-distilled", "Xorbits/LTX-2-distilled"),
        ("LTX-2-dev", "Xorbits/LTX-2-dev"),
        ("LTX-2.3-distilled", "Xorbits/LTX-2.3-distilled"),
        ("LTX-2.3-dev", "Xorbits/LTX-2.3-dev"),
    ],
)
def test_mirrored_mlx_models_default_to_modelscope(monkeypatch, model_name, model_id):
    monkeypatch.setenv("XINFERENCE_MODEL_SRC", "modelscope")

    model_spec = match_diffusion(model_name, model_engine="MLX")

    assert model_spec.model_hub == "modelscope"
    assert model_spec.model_id == model_id
    assert model_spec.model_revision == "master"


@pytest.mark.asyncio
async def test_video_catalog_groups_hub_variants():
    from ....core.worker import WorkerActor

    worker = WorkerActor.__new__(WorkerActor)
    registrations = await worker.list_model_registrations("video", detailed=True)
    entries = [item for item in registrations if item["model_name"] == "MiniMax-H3"]

    assert len(entries) == 1
    specs = entries[0]["model_specs"]
    assert {spec["model_engine"] for spec in specs} == {"diffusers", "vLLM", "SGLang"}
    assert {spec["model_format"] for spec in specs} == {"diffusers"}
    assert {spec["model_hub"] for spec in specs} == {"huggingface", "modelscope"}

    wan_entries = [
        item for item in registrations if item["model_name"] == "Wan2.2-A14B"
    ]
    assert len(wan_entries) == 1
    wan_specs = wan_entries[0]["model_specs"]
    assert {spec["model_engine"] for spec in wan_specs} == {
        "diffusers",
        "MLX",
        "vLLM",
        "SGLang",
    }
    assert {spec["model_format"] for spec in wan_specs} == {"diffusers", "mlx"}
    assert {spec["model_hub"] for spec in wan_specs} == {"huggingface", "modelscope"}


def test_video_registry_reload_failure_preserves_live_registries(monkeypatch):
    from ... import utils as model_utils

    model_snapshot = {name: list(specs) for name, specs in BUILTIN_VIDEO_MODELS.items()}
    description_snapshot = {
        name: list(descriptions)
        for name, descriptions in VIDEO_MODEL_DESCRIPTIONS.items()
    }
    engine_snapshot = {
        name: {engine: list(params) for engine, params in engines.items()}
        for name, engines in VIDEO_ENGINES.items()
    }

    def fail_after_partial_build(target, *_args, **_kwargs):
        target["partial"] = []
        raise RuntimeError("reload failed")

    monkeypatch.setattr(
        model_utils, "install_models_with_merge", fail_after_partial_build
    )

    with pytest.raises(RuntimeError, match="reload failed"):
        _install()

    assert BUILTIN_VIDEO_MODELS == model_snapshot
    assert VIDEO_MODEL_DESCRIPTIONS == description_snapshot
    assert VIDEO_ENGINES == engine_snapshot


@pytest.mark.parametrize("engine", ["diffusers", "vLLM", "SGLang"])
@pytest.mark.parametrize("precision", ["bf16", "fp8", "int8", "int4"])
def test_h3_runtime_precision_maps_before_engine_construction(engine, precision):
    from ..precision import apply_runtime_precision

    spec = match_diffusion("MiniMax-H3", "huggingface", model_engine=engine)
    # Launch histories can carry precision settings from another engine.
    legacy = {
        "quantization_config": "int4",
        "torch_dtype": "float16",
        "dtype": "float16",
        "component_quantizations": {"transformer": "int8"},
        "component_precisions": {"text_encoder": "fp16"},
        "runtime_precision": precision,
        "lightning_version": "4step_v1.0_768p_bf16",
        "cpu_offload": True,
    }
    original = legacy.copy()
    if precision not in spec.runtime_precisions:
        with pytest.raises(ValueError, match="not supported"):
            apply_runtime_precision(spec, engine, legacy)
        return
    config = apply_runtime_precision(spec, engine, legacy)
    assert legacy == original
    assert "runtime_precision" not in config
    assert config["lightning_version"] == legacy["lightning_version"]
    assert config["cpu_offload"] is True
    if engine == "diffusers":
        assert config["quantization_config"] == precision
        assert config["torch_dtype"] == "bfloat16"
        assert "dtype" not in config
        assert "component_quantizations" not in config
    elif engine == "vLLM":
        assert config["quantization_config"] == ("fp8" if precision == "fp8" else None)
        assert config["dtype"] == "bfloat16"
        assert "torch_dtype" not in config
        assert "component_quantizations" not in config
    else:
        assert config["component_quantizations"] == (
            {"transformer": "fp8", "text_encoder": "fp8"} if precision == "fp8" else {}
        )
        assert config["pipeline_config"] == {
            "dit_precision": "bf16",
            "text_encoder_precisions": ["bf16"],
        }
        assert "component_precisions" not in config
        assert "quantization_config" not in config
        assert "torch_dtype" not in config


@pytest.mark.parametrize("precision", [None, "auto"])
@pytest.mark.parametrize("engine", ["diffusers", "vLLM", "SGLang"])
def test_h3_default_precision_preserves_legacy_config(engine, precision):
    from ..precision import apply_runtime_precision

    spec = match_diffusion("MiniMax-H3", "huggingface", model_engine=engine)
    legacy = {"quantization_config": {"custom": True}, "torch_dtype": "float16"}
    config = apply_runtime_precision(
        spec, engine, dict(legacy, runtime_precision=precision)
    )
    assert config == legacy


@pytest.mark.parametrize(
    "engine,precision", [("vLLM", "int4"), ("SGLang", "int8"), ("diffusers", "fp4")]
)
def test_invalid_runtime_precision_does_not_download(engine, precision):
    with patch.object(VideoCacheManager, "cache") as cache:
        with pytest.raises(ValueError, match="runtime_precision.*not supported"):
            create_video_model_instance(
                "uid",
                "MiniMax-H3",
                model_engine=engine,
                download_hub="huggingface",
                runtime_precision=precision,
                enable_virtual_env=True,
            )
    cache.assert_not_called()


def test_factory_consumes_runtime_precision():
    model = create_video_model_instance(
        "uid",
        "MiniMax-H3",
        model_path="/fake/path",
        download_hub="huggingface",
        model_engine="diffusers",
        runtime_precision="bf16",
        enable_virtual_env=False,
    )
    assert model._kwargs["quantization_config"] == "bf16"
    assert model._kwargs["torch_dtype"] == "bfloat16"
    assert "runtime_precision" not in model._kwargs


def test_runtime_precision_requires_model_support():
    with patch.object(VideoCacheManager, "cache") as cache:
        with pytest.raises(ValueError, match="runtime_precision.*not supported"):
            create_video_model_instance(
                "uid",
                "CogVideoX-2b",
                model_engine="diffusers",
                download_hub="huggingface",
                runtime_precision="fp8",
                enable_virtual_env=False,
            )
    cache.assert_not_called()


@pytest.mark.parametrize("engine", ["diffusers", "vLLM", "SGLang"])
def test_h3_engine_api_reports_runtime_precisions(engine):
    try:
        with patch.object(_NativeVideoModel, "check_host", return_value=True):
            _install()
            params = get_engine_params_by_name_with_virtual_env(
                "video", "MiniMax-H3", enable_virtual_env=True
            )
        assert params[engine][0]["runtime_precisions"] == (
            ["bf16", "fp8", "int8", "int4"]
            if engine == "diffusers"
            else ["bf16", "fp8"]
        )
        assert params[engine][0]["default_runtime_precision"] == (
            "int4" if engine == "diffusers" else "bf16"
        )
    finally:
        _install()


def test_h3_sglang_precision_preserves_other_pipeline_options():
    from ..precision import apply_runtime_precision

    spec = match_diffusion("MiniMax-H3", "huggingface", model_engine="SGLang")
    pipeline = {"vae_tiling": True, "dit_precision": "fp16"}
    config = apply_runtime_precision(
        spec,
        "SGLang",
        {
            "runtime_precision": "bf16",
            "pipeline_config": pipeline,
        },
    )
    assert pipeline == {"vae_tiling": True, "dit_precision": "fp16"}
    assert config["pipeline_config"] == {
        "vae_tiling": True,
        "dit_precision": "bf16",
        "text_encoder_precisions": ["bf16"],
    }


def test_h3_explicit_precision_rejects_opaque_pipeline_config():
    from ..precision import apply_runtime_precision

    spec = match_diffusion("MiniMax-H3", "huggingface", model_engine="SGLang")
    with pytest.raises(ValueError, match="dictionary pipeline_config"):
        apply_runtime_precision(
            spec,
            "SGLang",
            {
                "runtime_precision": "bf16",
                "pipeline_config": "/custom/config.json",
            },
        )
