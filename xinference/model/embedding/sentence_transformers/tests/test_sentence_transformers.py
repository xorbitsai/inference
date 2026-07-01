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
    TransformersEmbeddingSpecV1,
    create_embedding_model_instance,
)

TEST_MODEL_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="bge-small-en-v1.5",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="BAAI/bge-small-en-v1.5",
            quantization="none",
            # use huggingface in ci
            # model_hub="modelscope",
        )
    ],
)


async def test_embedding_model_with_sentence_transformer():
    model_path = None

    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_embedding_model_instance(
            "mock",
            "bge-small-en-v1.5",
            "sentence_transformers",
            model_path=model_path,
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = await model.create_embedding(input_text)
        assert len(r["data"]) == 1
        assert len(r["data"][0]["embedding"]) == 384

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
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


async def test_embedding_model_with_sentence_transformer_truncate_dim():
    model_path = None

    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        truncate_dim = 128
        model = create_embedding_model_instance(
            "mock",
            "bge-small-en-v1.5",
            "sentence_transformers",
            model_path=model_path,
            dimensions=truncate_dim,
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = await model.create_embedding(input_text)
        assert len(r["data"]) == 1
        assert len(r["data"][0]["embedding"]) == truncate_dim

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
        for d in r["data"]:
            assert len(d["embedding"]) == truncate_dim
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
