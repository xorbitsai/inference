# Copyright 2022-2026 XProbe Inc.
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

import dataclasses
import sys
import types
from typing import Any, Optional
from unittest.mock import patch

import numpy as np
import pytest

from .. import BUILTIN_IMAGE_MODELS, _install
from ..engine import SGLangImageModel
from ..engine import platform as engine_platform
from ..sglang.core import SGLANG_SUPPORTED_IMAGE_MODELS, SGLangDiffusionModel


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    _install()


def _get_spec(model_name: str):
    return BUILTIN_IMAGE_MODELS[model_name][0]


def test_match_on_linux_with_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[SGLangImageModel.__module__], "has_cuda_device", return_value=True
    ):
        for model_name in SGLANG_SUPPORTED_IMAGE_MODELS:
            assert SGLangImageModel.match(_get_spec(model_name)) is True
        assert SGLangImageModel.match(_get_spec("sd3-medium")) is False


def test_match_rejects_non_linux():
    with patch.object(engine_platform, "system", return_value="Darwin"), patch.object(
        sys.modules[SGLangImageModel.__module__], "has_cuda_device", return_value=True
    ):
        assert SGLangImageModel.match(_get_spec("Qwen-Image")) is False


def test_match_rejects_without_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[SGLangImageModel.__module__], "has_cuda_device", return_value=False
    ):
        assert SGLangImageModel.match(_get_spec("Qwen-Image")) is False


def test_abilities_restricted_to_text2image():
    spec = _get_spec("Qwen-Image")
    original_abilities = list(spec.model_ability)
    assert "image2image" in original_abilities

    model = SGLangDiffusionModel("uid", "/path", model_spec=spec)
    assert model.model_ability == ["text2image"]
    assert model.model_family.model_ability == ["text2image"]
    # the shared builtin spec must stay untouched
    assert spec.model_ability == original_abilities


def test_constructor_rejects_unsupported_features():
    spec = _get_spec("Qwen-Image")
    with pytest.raises(ValueError, match="GGUF"):
        SGLangDiffusionModel(
            "uid", "/path", model_spec=spec, gguf_model_path="/gguf/path"
        )
    with pytest.raises(ValueError, match="Lightning"):
        SGLangDiffusionModel(
            "uid", "/path", model_spec=spec, lightning_model_path="/lightning/path"
        )
    with pytest.raises(ValueError, match="LoRA"):
        SGLangDiffusionModel("uid", "/path", model_spec=spec, lora_model=[object()])
    with pytest.raises(ValueError, match="Controlnet"):
        SGLangDiffusionModel("uid", "/path", model_spec=spec, controlnet=("canny",))


@pytest.fixture
def fake_sglang_sampling_params():
    @dataclasses.dataclass
    class FakeSamplingParams:
        prompt: Optional[str] = None
        negative_prompt: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        num_inference_steps: Optional[int] = None
        guidance_scale: float = 1.0
        seed: int = 42
        num_outputs_per_prompt: int = 1
        save_output: bool = True
        return_frames: bool = False
        request_id: Optional[str] = None

    mod = types.ModuleType("sglang.multimodal_gen.configs.sample.sampling_params")
    mod.SamplingParams = FakeSamplingParams
    fake_modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.multimodal_gen": types.ModuleType("sglang.multimodal_gen"),
        "sglang.multimodal_gen.configs": types.ModuleType(
            "sglang.multimodal_gen.configs"
        ),
        "sglang.multimodal_gen.configs.sample": types.ModuleType(
            "sglang.multimodal_gen.configs.sample"
        ),
        "sglang.multimodal_gen.configs.sample.sampling_params": mod,
    }
    to_restore = {name: sys.modules.get(name) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        yield FakeSamplingParams
    finally:
        for name, orig in to_restore.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def test_build_sampling_params(fake_sglang_sampling_params):
    spec = _get_spec("Qwen-Image")
    model = SGLangDiffusionModel("uid", "/path", model_spec=spec)

    params = model._build_sampling_params(
        "a cat",
        2,
        512,
        768,
        {
            "guidance_scale": 1,
            "true_cfg_scale": 1,  # diffusers-only, should be dropped
            "num_inference_steps": 20,
            "seed": 123,
        },
    )
    assert params["prompt"] == "a cat"
    assert params["width"] == 512
    assert params["height"] == 768
    assert params["num_outputs_per_prompt"] == 2
    assert params["seed"] == 123
    assert params["num_inference_steps"] == 20
    assert params["save_output"] is False
    assert params["return_frames"] is True
    assert "true_cfg_scale" not in params


def test_build_sampling_params_stringifies_uuid(fake_sglang_sampling_params):
    import uuid

    spec = _get_spec("Qwen-Image")
    model = SGLangDiffusionModel("uid", "/path", model_spec=spec)
    request_id = uuid.uuid4()
    params = model._build_sampling_params(
        "a cat", 1, 512, 512, {"request_id": request_id}
    )
    assert params["request_id"] == str(request_id)


@pytest.mark.parametrize("seed", [None, -1])
def test_build_sampling_params_random_seed(fake_sglang_sampling_params, seed):
    spec = _get_spec("Qwen-Image")
    model = SGLangDiffusionModel("uid", "/path", model_spec=spec)
    params = model._build_sampling_params("a cat", 1, 512, 512, {"seed": seed})
    assert isinstance(params["seed"], int)
    assert params["seed"] >= 0


def test_extract_images():
    @dataclasses.dataclass
    class FakeResult:
        frames: Any = None

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    images = SGLangDiffusionModel._extract_images(FakeResult(frames=[frame]))
    assert len(images) == 1
    assert images[0].size == (8, 8)

    images = SGLangDiffusionModel._extract_images(
        [FakeResult(frames=[frame]), FakeResult(frames=[frame])]
    )
    assert len(images) == 2

    with pytest.raises(RuntimeError, match="failed"):
        SGLangDiffusionModel._extract_images(None)

    with pytest.raises(RuntimeError, match="no images"):
        SGLangDiffusionModel._extract_images(FakeResult(frames=[]))


def test_builtin_specs_have_sglang_virtualenv_marker():
    for model_name in SGLANG_SUPPORTED_IMAGE_MODELS:
        for spec in BUILTIN_IMAGE_MODELS[model_name]:
            packages = spec.virtualenv.packages if spec.virtualenv else []
            assert any(
                "sglang" in pkg and '#engine# == "SGLang"' in pkg for pkg in packages
            ), f"{model_name} misses sglang virtualenv marker"
