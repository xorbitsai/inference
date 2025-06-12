# Copyright 2022-2025 XProbe Inc.
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

import pytest

from ...core import EmbeddingModelSpec, cache, create_embedding_model_instance
from ..core import VLLMEmbeddingModel

TEST_MODEL_SPEC = EmbeddingModelSpec(
    model_name="bge-small-en-v1.5",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_id="BAAI/bge-small-en-v1.5",
    model_hub="modelscope",
)


@pytest.mark.skipif(not VLLMEmbeddingModel.check_lib(), reason="vllm not installed")
def test_embedding_model_with_vllm():
    model_path = None

    try:
        model_path = cache(TEST_MODEL_SPEC)

        model, _ = create_embedding_model_instance(
            "mook",
            None,
            "mock",
            "bge-small-en-v1.5",
            "vllm",
            model_path,
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = model.create_embedding(input_text)
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
        r = model.create_embedding(input_texts)
        assert len(r["data"]) == 4
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
