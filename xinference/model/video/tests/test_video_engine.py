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

from unittest.mock import patch

import pytest

from ....core.utils import filter_virtualenv_packages_by_markers
from ...core import create_model_instance
from ...utils import (
    get_engine_params_by_name,
    get_engine_params_by_name_with_virtual_env,
)
from .. import BUILTIN_VIDEO_MODELS, _install
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
from ..engine_family import VIDEO_ENGINES, check_engine_by_model_name_and_engine


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    with patch.object(MLXVideoEngineModel, "_is_apple_silicon", return_value=True):
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
        assert list(engines) == expected_engines

        if "diffusers" in engines:
            assert engines["diffusers"] == [
                {
                    "model_name": model_name,
                    "model_format": "diffusers",
                    "quantization": "none",
                    "video_class": DiffusersVideoEngineModel,
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

    assert params == {
        "diffusers": [
            {
                "model_name": "MiniMax-H3",
                "model_format": "diffusers",
                "quantization": "none",
            }
        ]
    }


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
    assert {spec["model_engine"] for spec in specs} == {"diffusers"}
    assert {spec["model_format"] for spec in specs} == {"diffusers"}
    assert {spec["model_hub"] for spec in specs} == {"huggingface", "modelscope"}

    wan_entries = [
        item for item in registrations if item["model_name"] == "Wan2.2-A14B"
    ]
    assert len(wan_entries) == 1
    wan_specs = wan_entries[0]["model_specs"]
    assert {spec["model_engine"] for spec in wan_specs} == {"diffusers", "MLX"}
    assert {spec["model_format"] for spec in wan_specs} == {"diffusers", "mlx"}
    assert {spec["model_hub"] for spec in wan_specs} == {"huggingface", "modelscope"}
