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

from ...cache_manager import EmbeddingCacheManager as CacheManager
from ...core import (
    EmbeddingModelFamilyV2,
    LlamaCppEmbeddingSpecV1,
    create_embedding_model_instance,
)

TEST_MODEL_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="Qwen3-Embedding-0.6B",
    dimensions=1024,
    max_tokens=32768,
    language=["en"],
    model_specs=[
        LlamaCppEmbeddingSpecV1(
            model_format="ggufv2",
            model_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
            model_file_name_template="Qwen3-Embedding-0.6B-{quantization}.gguf",
            quantization="Q8_0",
            model_hub="huggingface",
        )
    ],
)


async def test_embedding_model_with_xllamacpp():
    model_path = None
    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_embedding_model_instance(
            "mock",
            "Qwen3-Embedding-0.6B",
            "llama.cpp",
            model_format="ggufv2",
            quantization="Q8_0",
            model_path=model_path,
        )
        model.load()

        # input is a lit
        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ]
        # test sparse and dense
        r = await model.create_embedding(input_texts)
        assert len(r["data"]) == 4

        r = await model.create_embedding(input_texts)
        for d in r["data"]:
            assert len(d["embedding"]) == 1024
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
