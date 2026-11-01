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

# MLX repos that exist on neither hub (okwinds' bf16 repo was only ever
# missing on ModelScope).
MISSING = {
    "mlx-community/ERNIE-4.5-300B-47B-PT-4bit",
    "mlx-community/Qwen3-VL-235B-A22B-Instruct-8bit",
    "mlx-community/Qwen3-VL-235B-A22B-Thinking-4bit",
    "mlx-community/Qwen3-VL-235B-A22B-Thinking-8bit",
    "okwinds/DeepSeek-R1-Distill-Llama-8B-MLX-bf16",
}


def _repos():
    for path in glob.glob(os.path.join(MODEL_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            for family in json.load(f):
                for spec in family["model_specs"]:
                    for hub, src in spec["model_src"].items():
                        for q in src["quantizations"]:
                            yield family["model_name"], spec, hub, src[
                                "model_id"
                            ].replace("{quantization}", q)


def test_no_spec_points_at_a_missing_mlx_repo():
    found = [(name, hub, repo) for name, _, hub, repo in _repos() if repo in MISSING]
    assert found == []


def test_ernie_4_5_300b_mlx_uses_the_a47b_repo():
    repos = {
        (hub, repo)
        for name, spec, hub, repo in _repos()
        if name == "Ernie4.5"
        and spec["model_format"] == "mlx"
        and spec["model_size_in_billions"] == 300
    }
    assert repos == {
        ("huggingface", "mlx-community/ERNIE-4.5-300B-A47B-PT-4bit"),
        ("modelscope", "mlx-community/ERNIE-4.5-300B-A47B-PT-4bit"),
    }


def test_deepseek_r1_distill_llama_8b_mlx_uses_the_same_repos_on_both_hubs():
    repos = {}
    for name, spec, hub, repo in _repos():
        if (
            name == "deepseek-r1-distill-llama"
            and spec["model_format"] == "mlx"
            and spec["model_size_in_billions"] == 8
        ):
            repos.setdefault(hub, set()).add(repo)
    assert repos["modelscope"] == repos["huggingface"]
    assert "mlx-community/DeepSeek-R1-Distill-Llama-8B-bf16" in repos["modelscope"]
