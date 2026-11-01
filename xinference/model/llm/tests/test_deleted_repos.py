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
import glob
import json
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Repos that no longer exist on Hugging Face. The last three are still on
# ModelScope, so their specs keep their ModelScope source.
DELETED = {
    "cpatonn/DeepSeek-V3.1-GPTQ-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen-1_8B-Chat",
    "Skywork/Skywork-OR1-32B-Preview",
    "Skywork/Skywork-OR1-7B-Preview",
}


def test_no_spec_points_at_a_deleted_repo():
    found = []
    for path in glob.glob(os.path.join(MODEL_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            for family in json.load(f):
                for spec in family["model_specs"]:
                    src = spec["model_src"].get("huggingface") or {}
                    if src.get("model_id") in DELETED:
                        found.append((family["model_name"], src["model_id"]))
    assert found == []


def test_llama_3_1_instruct_mlx_8b_unquantized_uses_the_bf16_repo():
    # Same naming as the 70B spec: mlx-community/Meta-Llama-3.1-70B-Instruct-bf16.
    from ..llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("llama-3.1-instruct")
    ids = {
        s.model_id
        for s in family.model_specs
        if s.model_format == "mlx"
        and s.model_size_in_billions == 8
        and s.quantization == "none"
    }
    assert ids == {"mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"}


def test_specs_only_on_modelscope_keep_their_modelscope_source():
    ms_ids = set()
    for name in ("qwen-chat", "skywork-or1-preview"):
        with open(os.path.join(MODEL_DIR, f"{name}.json"), encoding="utf-8") as f:
            for family in json.load(f):
                for spec in family["model_specs"]:
                    src = spec["model_src"].get("modelscope") or {}
                    ms_ids.add(src.get("model_id"))
    assert {
        "qwen/Qwen-1_8B-Chat",
        "Skywork/Skywork-OR1-32B-Preview",
        "Skywork/Skywork-OR1-7B-Preview",
    } <= ms_ids
