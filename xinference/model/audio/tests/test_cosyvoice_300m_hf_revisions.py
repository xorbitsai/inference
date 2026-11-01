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

# The FunAudioLLM/CosyVoice-300M* repos no longer contain the commits these
# specs used to pin. These are commits that exist and contain cosyvoice.yaml,
# llm.pt, flow.pt, hift.pt, campplus.onnx and speech_tokenizer_v1.onnx.
HF_REVISIONS = {
    "CosyVoice-300M": "24c40509c3c5ea6fe06b5f8790ff99e3714a6bee",
    "CosyVoice-300M-Instruct": "706bee1915e9fd1f1214929e2a0509c874cff433",
    "CosyVoice-300M-SFT": "fbb71de2afe387ed854eebd80b9f3d078c6b9869",
}


@pytest.mark.parametrize("model_name", sorted(HF_REVISIONS))
def test_cosyvoice_300m_hf_revision_exists(model_name):
    with open(os.path.join(MODELS_DIR, f"{model_name}.json"), encoding="utf-8") as f:
        spec = json.load(f)
    spec = spec[0] if isinstance(spec, list) else spec
    hf = spec["model_src"]["huggingface"]
    assert hf["model_id"] == f"FunAudioLLM/{model_name}"
    assert hf["model_revision"] == HF_REVISIONS[model_name]
