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

import pytest

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# The runwayml repos were removed from Hugging Face. Their names now redirect to
# the stable-diffusion-v1-5 organization, where the old commits do not exist, so
# the specs pin these repos and commits instead.
HF_SOURCES = {
    "stable-diffusion-v1.5": (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    ),
    "stable-diffusion-inpainting": (
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "8a4288a76071f7280aedbdb3253bdb9e9d5d84bb",
    ),
}


@pytest.mark.parametrize("model_name", sorted(HF_SOURCES))
def test_sd15_hf_source_is_resolvable(model_name):
    with open(os.path.join(MODELS_DIR, f"{model_name}.json"), encoding="utf-8") as f:
        (spec,) = json.load(f)
    hf = spec["model_src"]["huggingface"]
    assert (hf["model_id"], hf["model_revision"]) == HF_SOURCES[model_name]
