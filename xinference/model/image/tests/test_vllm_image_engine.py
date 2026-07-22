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

import pytest
from PIL import Image

from .. import BUILTIN_IMAGE_MODELS, _install
from ..engine import VLLMImageModel
from ..engine import platform as engine_platform
from ..vllm.core import VLLM_SUPPORTED_IMAGE_MODELS, VLLMDiffusionModel


@pytest.fixture(scope="module", autouse=True)
def setup_builtin_models():
    _install()


def _get_spec(model_name: str):
    return BUILTIN_IMAGE_MODELS[model_name][0]


def test_match_on_linux_with_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
    ):
        for model_name in VLLM_SUPPORTED_IMAGE_MODELS:
            assert VLLMImageModel.match(_get_spec(model_name)) is True
        assert VLLMImageModel.match(_get_spec("sd3-medium")) is False


def test_match_rejects_non_linux():
    with patch.object(engine_platform, "system", return_value="Darwin"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=True
    ):
        assert VLLMImageModel.match(_get_spec("Z-Image")) is False


def test_match_rejects_without_cuda():
    with patch.object(engine_platform, "system", return_value="Linux"), patch.object(
        sys.modules[VLLMImageModel.__module__], "has_cuda_device", return_value=False
    ):
        assert VLLMImageModel.match(_get_spec("Z-Image")) is False


def test_constructor_rejects_unsupported_features():
    spec = _get_spec("Z-Image")
    with pytest.raises(ValueError, match="GGUF"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, gguf_model_path="/gguf")
    with pytest.raises(ValueError, match="Lightning"):
        VLLMDiffusionModel(
            "uid", "/path", model_spec=spec, lightning_model_path="/lightning"
        )
    with pytest.raises(ValueError, match="LoRA"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, lora_model=[object()])
    with pytest.raises(ValueError, match="Controlnet"):
        VLLMDiffusionModel("uid", "/path", model_spec=spec, controlnet=("canny",))


@pytest.fixture
def fake_vllm_omni_sampling_params():
    @dataclasses.dataclass
    class FakeSamplingParams:
        negative_prompt: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        num_inference_steps: Optional[int] = None
        guidance_scale: float = 1.0
        true_cfg_scale: float = 1.0
        seed: Optional[int] = None
        num_outputs_per_prompt: int = 1

    data_mod = types.ModuleType("vllm_omni.inputs.data")
    data_mod.OmniDiffusionSamplingParams = FakeSamplingParams
    fake_modules = {
        "vllm_omni": types.ModuleType("vllm_omni"),
        "vllm_omni.inputs": types.ModuleType("vllm_omni.inputs"),
        "vllm_omni.inputs.data": data_mod,
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


def test_build_sampling_params(fake_vllm_omni_sampling_params):
    spec = _get_spec("Z-Image")
    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)

    params = model._build_sampling_params(
        2,
        512,
        768,
        {
            "true_cfg_scale": 4.0,
            "num_inference_steps": 20,
            "seed": 123,
            "save_output": False,  # sglang-only, should be dropped
        },
    )
    assert isinstance(params, fake_vllm_omni_sampling_params)
    assert params.width == 512
    assert params.height == 768
    assert params.num_outputs_per_prompt == 2
    assert params.seed == 123
    assert params.num_inference_steps == 20
    assert params.true_cfg_scale == 4.0
    assert not hasattr(params, "save_output")


@pytest.mark.parametrize("seed", [None, -1])
def test_build_sampling_params_random_seed(fake_vllm_omni_sampling_params, seed):
    spec = _get_spec("Z-Image")
    model = VLLMDiffusionModel("uid", "/path", model_spec=spec)
    params = model._build_sampling_params(1, 512, 512, {"seed": seed})
    assert isinstance(params.seed, int)
    assert params.seed >= 0


def test_extract_images():
    @dataclasses.dataclass
    class FakeOutput:
        images: Any = None
        request_output: Any = None

    image = Image.new("RGB", (8, 8))
    images = VLLMDiffusionModel._extract_images([FakeOutput(images=[image])])
    assert len(images) == 1
    assert images[0].size == (8, 8)

    # images nested inside request_output
    images = VLLMDiffusionModel._extract_images(
        [FakeOutput(request_output=FakeOutput(images=[image, image]))]
    )
    assert len(images) == 2

    with pytest.raises(RuntimeError, match="failed"):
        VLLMDiffusionModel._extract_images([])

    with pytest.raises(RuntimeError, match="no images"):
        VLLMDiffusionModel._extract_images([FakeOutput(images=[])])

    with pytest.raises(RuntimeError, match="Unexpected image type"):
        VLLMDiffusionModel._extract_images([FakeOutput(images=[object()])])


def test_builtin_specs_have_vllm_virtualenv_marker():
    for model_name in VLLM_SUPPORTED_IMAGE_MODELS:
        for spec in BUILTIN_IMAGE_MODELS[model_name]:
            packages = spec.virtualenv.packages if spec.virtualenv else []
            assert any(
                "vllm-omni" in pkg and '#engine# == "vLLM"' in pkg for pkg in packages
            ), f"{model_name} misses vllm-omni virtualenv marker"
