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

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# GGUF files in city96/Qwen-Image-gguf (Hugging Face and the ModelScope mirror).
QWEN_IMAGE_GGUF_FILES = {
    f"qwen-image-{q}.gguf"
    for q in (
        "BF16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S",
        "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
    )
}  # fmt: skip


def test_qwen_image_gguf_files_exist():
    with open(os.path.join(MODELS_DIR, "Qwen-Image.json"), encoding="utf-8") as f:
        (spec,) = json.load(f)
    for source in spec["model_src"].values():
        template = source["gguf_model_file_name_template"]
        names = {template.format(quantization=q) for q in source["gguf_quantizations"]}
        assert names <= QWEN_IMAGE_GGUF_FILES
