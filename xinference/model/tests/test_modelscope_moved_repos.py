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

MODEL_ROOT = os.path.join(os.path.dirname(__file__), "..")

# ModelScope repos that now return 404, and the repo holding the same files
# today (every LFS file has the same SHA-256 as the Hugging Face source).
MOVED = {
    "AIDC-AI/Marco-o1": "ATH-MaaS/Marco-o1",
    "AIDC-AI/Ovis2-1B": "ATH-MaaS/Ovis2-1B",
    "AIDC-AI/Ovis2-2B": "ATH-MaaS/Ovis2-2B",
    "AIDC-AI/Ovis2-4B": "ATH-MaaS/Ovis2-4B",
    "AIDC-AI/Ovis2-8B": "ATH-MaaS/Ovis2-8B",
    "AIDC-AI/Ovis2-16B": "ATH-MaaS/Ovis2-16B",
    "AIDC-AI/Ovis2-34B": "ATH-MaaS/Ovis2-34B",
    "AIDC-AI/Ovis2-2B-GPTQ-{quantization}": "ATH-MaaS/Ovis2-2B-GPTQ-{quantization}",
    "AIDC-AI/Ovis2-4B-GPTQ-{quantization}": "ATH-MaaS/Ovis2-4B-GPTQ-{quantization}",
    "AIDC-AI/Ovis2-8B-GPTQ-{quantization}": "ATH-MaaS/Ovis2-8B-GPTQ-{quantization}",
    "AIDC-AI/Ovis2-16B-GPTQ-{quantization}": "ATH-MaaS/Ovis2-16B-GPTQ-{quantization}",
    "AIDC-AI/Ovis2-34B-GPTQ-{quantization}": "ATH-MaaS/Ovis2-34B-GPTQ-{quantization}",
    "ByteDance-Seed/Seed-OSS-36B-Instruct": "bytedance-community/Seed-OSS-36B-Instruct",
    "ByteDance/MegaTTS3": "bytedance-community/MegaTTS3",
    "DianJin/DianJin-R1-7B": "tongyi_dianjin/DianJin-R1-7B",
    "DianJin/DianJin-R1-32B": "tongyi_dianjin/DianJin-R1-32B",
    "SeaLLMs/SeaLLMs-v3-1.5B-Chat": "QwenCollection/SeaLLMs-v3-1.5B-Chat",
    "SeaLLMs/SeaLLMs-v3-7B-Chat": "QwenCollection/SeaLLMs-v3-7B-Chat",
    "Tencent-Hunyuan/WeMM-Embedding-2B": "tencent-community/WeMM-Embedding-2B",
    "Tencent-Hunyuan/WeMM-Embedding-4B": "tencent-community/WeMM-Embedding-4B",
    "Tencent-Hunyuan/WeMM-Embedding-9B": "tencent-community/WeMM-Embedding-9B",
    "rednote-hilab/dots.ocr": "dots-studio/dots.ocr",
}


def _modelscope_ids():
    ids = set()

    def walk(node):
        if isinstance(node, dict):
            src = (node.get("model_src") or {}).get("modelscope") or {}
            if src.get("model_id"):
                ids.add(src["model_id"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in glob.glob(os.path.join(MODEL_ROOT, "*", "models", "*.json")):
        with open(path, encoding="utf-8") as f:
            walk(json.load(f))
    return ids


def test_no_modelscope_source_points_at_a_moved_repo():
    assert sorted(_modelscope_ids() & MOVED.keys()) == []


def test_moved_specs_point_at_the_new_modelscope_repo():
    assert sorted(set(MOVED.values()) - _modelscope_ids()) == []


def test_llama_3_1_8b_gguf_uses_the_repo_with_the_files_on_modelscope():
    # LLM-Research/Meta-Llama-3.1-8B-Instruct-GGUF only has lower-case q2_k
    # and hf-q3_k_m files; ModelScope mirrors the Hugging Face source repo.
    path = os.path.join(MODEL_ROOT, "llm", "models", "llama-3.1-instruct.json")
    with open(path, encoding="utf-8") as f:
        families = json.load(f)
    ids = {
        hub: src["model_id"]
        for family in families
        for spec in family["model_specs"]
        if spec["model_format"] == "ggufv2" and spec["model_size_in_billions"] == 8
        for hub, src in spec["model_src"].items()
    }
    assert ids == {
        "huggingface": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
        "modelscope": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    }
