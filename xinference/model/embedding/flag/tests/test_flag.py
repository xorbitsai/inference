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

from ...core import EmbeddingModelSpec, cache, create_embedding_model_instance

TEST_MODEL_SPEC = EmbeddingModelSpec(
    model_name="bge-small-en-v1.5",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_id="BAAI/bge-small-en-v1.5",
    model_hub="modelscope",
)


# todo 参考sentence_transformer的返回格式进行返回
def test_embedding_model_with_flag():
    model_path = None
    try:
        model_path = cache(TEST_MODEL_SPEC)

        model, _ = create_embedding_model_instance(
            "mook", "cuda", "mock", "bge-small-en-v1.5", "flag", model_path
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = model.create_embedding(input_text, **{"return_sparse": True})
        assert len(r["data"]) == 1

        r = model.create_embedding(input_text)
        assert len(r["data"][0]["embedding"]) == 384

        # input is a lit
        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ]
        # test sparse and dense
        r = model.create_embedding(input_texts, **{"return_sparse": True})
        assert len(r["data"]) == 4

        r = model.create_embedding(input_texts)
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
