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

from pathlib import Path

import pytest

from .. import load_model_family_from_json


@pytest.fixture
def krea_families():
    families = {}
    spec_path = Path(__file__).parents[1] / "model_spec.json"
    load_model_family_from_json(str(spec_path), families)
    return families


def test_krea_2_model_metadata(krea_families):
    assert set(krea_families) >= {"Krea-2-Raw", "Krea-2-Turbo"}

    for model_name in ("Krea-2-Raw", "Krea-2-Turbo"):
        specs = {spec.model_hub: spec for spec in krea_families[model_name]}
        assert specs["huggingface"].model_id == f"krea/{model_name}"
        assert specs["huggingface"].model_revision == "main"
        assert specs["modelscope"].model_id == f"krea/{model_name}"
        assert specs["modelscope"].model_revision == "master"
        for spec in specs.values():
            assert getattr(spec, "lora_models", None) is None
            assert spec.virtualenv.no_build_isolation is False
            assert (
                'diffusers>=0.39.0 ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )
            assert all(
                not package.startswith(
                    "git+https://github.com/huggingface/diffusers"
                )
                for package in spec.virtualenv.packages
            )
            assert (
                'huggingface-hub>=1.23.0,<2.0 ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )
            assert (
                '#system_torchvision# ; #engine# == "diffusers"'
                in spec.virtualenv.packages
            )
            assert any(
                "sglang[diffusion]>=0.5.15" in package
                and '#engine# == "SGLang"' in package
                for package in spec.virtualenv.packages
            )

    raw_spec = krea_families["Krea-2-Raw"][0]
    assert raw_spec.default_generate_config == {
        "num_inference_steps": 52,
        "guidance_scale": 3.5,
    }

    turbo_spec = krea_families["Krea-2-Turbo"][0]
    assert turbo_spec.default_generate_config == {
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
    }
