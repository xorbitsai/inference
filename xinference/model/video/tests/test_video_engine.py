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
from ..core import create_video_model_instance, resolve_video_model_name_and_engine
from ..engine import DiffusersVideoEngineModel
from ..engine_family import VIDEO_ENGINES, check_engine_by_model_name_and_engine


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    _install()


def test_builtin_video_models_register_diffusers_engine():
    assert set(VIDEO_ENGINES) == set(BUILTIN_VIDEO_MODELS)
    for model_name, engines in VIDEO_ENGINES.items():
        assert list(engines) == ["diffusers"]
        params = engines["diffusers"]
        assert params == [
            {
                "model_name": model_name,
                "model_format": "diffusers",
                "quantization": "none",
                "video_class": DiffusersVideoEngineModel,
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


def test_video_specs_scope_diffusers_dependency_to_engine():
    for families in BUILTIN_VIDEO_MODELS.values():
        for family in families:
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
