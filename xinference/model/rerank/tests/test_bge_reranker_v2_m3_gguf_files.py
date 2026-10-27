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
from ..rerank_family import BUILTIN_RERANK_MODELS

# GGUF files in gpustack/bge-reranker-v2-m3-GGUF at its latest (and second)
# commit 3093af03b1a635e67b084b1d8c03c5f5e020fd05 on Hugging Face; the
# ModelScope mirror has the same files.
GPUSTACK_GGUF_FILES = {
    "bge-reranker-v2-m3-FP16.gguf",
    "bge-reranker-v2-m3-Q2_K.gguf",
    "bge-reranker-v2-m3-Q3_K_M.gguf",
    "bge-reranker-v2-m3-Q4_0.gguf",
    "bge-reranker-v2-m3-Q4_K_M.gguf",
    "bge-reranker-v2-m3-Q5_0.gguf",
    "bge-reranker-v2-m3-Q5_K_M.gguf",
    "bge-reranker-v2-m3-Q6_K.gguf",
    "bge-reranker-v2-m3-Q8_0.gguf",
}


def test_bge_reranker_v2_m3_gguf_quantizations_exist():
    from .. import _install

    _install()
    specs = [
        spec
        for family in BUILTIN_RERANK_MODELS["bge-reranker-v2-m3"]
        for spec in family.model_specs
        if spec.model_format == "ggufv2"
    ]
    assert specs
    missing = [
        (spec.model_hub, spec.quantization)
        for spec in specs
        if spec.model_file_name_template.format(quantization=spec.quantization)
        not in GPUSTACK_GGUF_FILES
    ]
    assert missing == []


def test_bge_reranker_v2_m3_default_gguf_quantization_exists():
    # Launching with model_format ggufv2 and no quantization takes the first
    # listed one, so it must name a file the repo has.
    from .. import _install
    from ..rerank_family import match_rerank

    _install()
    for hub in ("huggingface", "modelscope"):
        spec = match_rerank(
            "bge-reranker-v2-m3", model_format="ggufv2", download_hub=hub
        ).model_specs[0]
        file_name = spec.model_file_name_template.format(quantization=spec.quantization)
        assert file_name in GPUSTACK_GGUF_FILES
