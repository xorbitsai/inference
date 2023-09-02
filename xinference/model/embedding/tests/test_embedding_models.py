# Copyright 2022-2023 XProbe Inc.
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

from ..core import EmbeddingModel, EmbeddingModelSpec, cache

TEST_MODEL_SPEC = EmbeddingModelSpec(
    model_name="gte-small",
    dimensions=384,
    max_tokens=512,
    language="en",
    model_id="thenlper/gte-small",
    model_revision="d8e2604cadbeeda029847d19759d219e0ce2e6d8",
)


def test_model():
    model_path = cache(TEST_MODEL_SPEC)
    model = EmbeddingModel("mock", model_path)
    # input is a string
    input_text = "what is the capital of China?"
    model.load()
    r = model.create_embedding(input_text)
    assert len(r["data"]) == 1
    for d in r["data"]:
        assert len(d["embedding"]) == 384

    # input is a lit
    input_texts = [
        "what is the capital of China?",
        "how to implement quick sort in python?",
        "Beijing",
        "sorting algorithms",
    ]
    model.load()
    r = model.create_embedding(input_texts)
    assert len(r["data"]) == 4
    for d in r["data"]:
        assert len(d["embedding"]) == 384
