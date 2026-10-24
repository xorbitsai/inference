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

import json
import os
from typing import List, Union
from unittest.mock import Mock

import pytest

from xinference._model_catalog import load_model_catalog

from ..core import BUILTIN_IMAGE_MODELS, IMAGE_MODEL_DESCRIPTIONS
from ..engine_family import IMAGE_ENGINES
from ..ocr.ocr_family import OCR_ENGINES


def test_register_builtin_model_prunes_stale_derived_entries_on_catalog_removal(
    tmp_path, monkeypatch
):
    # Downloaded-only models still in IMAGE_ENGINES/OCR_ENGINES or
    # IMAGE_MODEL_DESCRIPTIONS after they drop out of a later catalog refresh
    # keep advertising a launch config or description, even though
    # BUILTIN_IMAGE_MODELS (the table they derive from) no longer has them.
    import xinference.model.image as image_module

    from .... import constants
    from .. import register_builtin_model

    # has_downloaded_models()/load_downloaded_models() read the module-level
    # XINFERENCE_MODEL_DIR bound at import time, not a fresh lookup, so both
    # bindings need patching (same shape as the rerank regression test).
    monkeypatch.setattr(image_module, "XINFERENCE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(constants, "XINFERENCE_MODEL_DIR", str(tmp_path))

    spec_path = os.path.join(os.path.dirname(__file__), "..", "models")
    raw_entry = load_model_catalog(spec_path)[0]
    downloaded_only = dict(raw_entry)
    downloaded_only["model_name"] = "downloaded-only-catalog-removal-test"

    raw_ocr_entry = next(
        entry
        for entry in load_model_catalog(spec_path)
        if entry["model_name"] == "PaddleOCR-VL"
    )
    downloaded_only_ocr = dict(raw_ocr_entry)
    downloaded_only_ocr["model_name"] = "PaddleOCR-VL-catalog-removal-test"

    builtin_dir = os.path.join(str(tmp_path), "v2", "builtin", "image")
    os.makedirs(builtin_dir, exist_ok=True)
    catalog_path = os.path.join(builtin_dir, "image_models.json")
    with open(catalog_path, "w") as f:
        json.dump([downloaded_only, downloaded_only_ocr], f)

    register_builtin_model()
    assert "downloaded-only-catalog-removal-test" in BUILTIN_IMAGE_MODELS
    assert "downloaded-only-catalog-removal-test" in IMAGE_ENGINES
    assert "downloaded-only-catalog-removal-test" in IMAGE_MODEL_DESCRIPTIONS
    assert "PaddleOCR-VL-catalog-removal-test" in BUILTIN_IMAGE_MODELS
    assert "PaddleOCR-VL-catalog-removal-test" in OCR_ENGINES
    assert "PaddleOCR-VL-catalog-removal-test" in IMAGE_MODEL_DESCRIPTIONS

    baseline_image_entries = sum(
        len(entries)
        for entries in IMAGE_ENGINES["downloaded-only-catalog-removal-test"].values()
    )
    baseline_ocr_entries = sum(
        len(entries)
        for entries in OCR_ENGINES["PaddleOCR-VL-catalog-removal-test"].values()
    )
    register_builtin_model()
    assert (
        sum(
            len(entries)
            for entries in IMAGE_ENGINES[
                "downloaded-only-catalog-removal-test"
            ].values()
        )
        == baseline_image_entries
    )
    assert (
        sum(
            len(entries)
            for entries in OCR_ENGINES["PaddleOCR-VL-catalog-removal-test"].values()
        )
        == baseline_ocr_entries
    )

    # A later refresh's catalog no longer lists the model (removed upstream).
    with open(catalog_path, "w") as f:
        json.dump([], f)

    register_builtin_model()
    assert "downloaded-only-catalog-removal-test" not in BUILTIN_IMAGE_MODELS
    assert "downloaded-only-catalog-removal-test" not in IMAGE_ENGINES
    assert "downloaded-only-catalog-removal-test" not in IMAGE_MODEL_DESCRIPTIONS
    assert "PaddleOCR-VL-catalog-removal-test" not in BUILTIN_IMAGE_MODELS
    assert "PaddleOCR-VL-catalog-removal-test" not in OCR_ENGINES
    assert "PaddleOCR-VL-catalog-removal-test" not in IMAGE_MODEL_DESCRIPTIONS


@pytest.mark.parametrize(
    "controlnet,expected_names",
    [
        pytest.param("canny", ["canny"], id="scalar"),
        pytest.param(["canny"], ["canny"], id="singleton-list"),
        pytest.param(["canny", "mlsd"], ["canny", "mlsd"], id="multiple-list"),
    ],
)
def test_create_image_model_instance_accepts_controlnet(
    monkeypatch: pytest.MonkeyPatch,
    controlnet: Union[str, List[str]],
    expected_names: List[str],
) -> None:
    from ..cache_manager import ImageCacheManager
    from ..core import create_image_model_instance

    model_name = "stable-diffusion-v1.5"
    cached_names: List[str] = []

    class CacheReached(Exception):
        pass

    def fake_cache(cache_manager: ImageCacheManager) -> str:
        name = cache_manager._model_family.model_name
        cached_names.append(name)
        if name == model_name:
            raise CacheReached
        return f"/cache/{name}"

    monkeypatch.setattr(ImageCacheManager, "cache", fake_cache)
    with pytest.raises(CacheReached):
        create_image_model_instance(
            "test", model_name, controlnet=controlnet, enable_virtual_env=False
        )
    assert cached_names == expected_names + [model_name]


@pytest.mark.parametrize(
    "controlnet,message",
    [
        pytest.param(
            ["canny", "canny"],
            "controlnet should be a list of unique str.",
            id="duplicates",
        ),
        pytest.param(
            ["unknown"],
            "controlnet `unknown` is not supported for model `stable-diffusion-v1.5`.",
            id="unknown-list",
        ),
        pytest.param(
            "unknown",
            "controlnet `unknown` is not supported for model `stable-diffusion-v1.5`.",
            id="unknown-scalar",
        ),
    ],
)
def test_create_image_model_instance_rejects_invalid_controlnet(
    monkeypatch: pytest.MonkeyPatch,
    controlnet: Union[str, List[str]],
    message: str,
) -> None:
    from ..cache_manager import ImageCacheManager
    from ..core import create_image_model_instance

    cache = Mock(side_effect=AssertionError("Invalid controlnet reached caching"))
    monkeypatch.setattr(ImageCacheManager, "cache", cache)
    with pytest.raises(ValueError) as exc_info:
        create_image_model_instance(
            "test",
            "stable-diffusion-v1.5",
            controlnet=controlnet,
            enable_virtual_env=False,
        )
    assert str(exc_info.value) == message
    cache.assert_not_called()
