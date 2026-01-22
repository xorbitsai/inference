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

import shutil

from ...cache_manager import RerankCacheManager as CacheManager
from ...core import (
    LlamaCppRerankSpecV1,
    RerankModelFamilyV2,
    create_rerank_model_instance,
)

TEST_MODEL_SPEC = RerankModelFamilyV2(
    version=2,
    model_name="bge-reranker-v2-m3",
    language=["en"],
    model_specs=[
        LlamaCppRerankSpecV1(
            model_format="ggufv2",
            model_id="gpustack/bge-reranker-v2-m3-GGUF",
            model_file_name_template="bge-reranker-v2-m3-{quantization}.gguf",
            quantization="Q4_K_M",
            model_hub="modelscope",
        )
    ],
)


def test_rerank_model_with_xllamacpp():
    model_path = None
    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_rerank_model_instance(
            "mock",
            "bge-reranker-v2-m3",
            "llama.cpp",
            model_format="ggufv2",
            quantization="Q4_K_M",
            model_path=model_path,
        )
        model.load()

        query = "A man is eating pasta."

        corpus = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin.",
            "Two men pushed carts through the woods.",
            "A man is riding a white horse on an enclosed ground.",
            "A monkey is playing drums.",
            "A cheetah is running behind its prey.",
        ]

        scores = model.rerank(corpus, query, None, None, True, True)
        assert scores["results"][0]["index"] == 0

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
