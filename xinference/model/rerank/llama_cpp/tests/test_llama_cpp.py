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

import shutil

import pytest

from ...cache_manager import RerankCacheManager as CacheManager
from ...core import (
    LlamaCppRerankSpecV1,
    RerankModelFamilyV2,
    create_rerank_model_instance,
)
from ..core import XllamaCppRerankModel

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


class _FakeLlamaCppServer:
    def __init__(self, response):
        self.response = response

    def handle_rerank(self, data):
        return self.response


@pytest.mark.parametrize("nested", [False, True])
def test_rerank_model_raises_xllamacpp_error(nested):
    message = "Field 'documents': documents must not be empty"
    error = {
        "code": 400,
        "message": message,
        "type": "invalid_request_error",
    }
    model = XllamaCppRerankModel.__new__(XllamaCppRerankModel)
    model._llm = _FakeLlamaCppServer({"error": error} if nested else error)

    with pytest.raises(Exception, match="documents must not be empty") as exc_info:
        model.rerank([], "query", None, None, False, False)

    assert str(exc_info.value) == message


def test_gguf_rerank_defaults_engine_and_quantization(tmp_path, monkeypatch):
    # A GGUF-only custom rerank model launched without ``--model-engine`` or
    # ``--quantization`` must pick llama.cpp and the spec's quantization
    # instead of failing on sentence_transformers / rendering ``None`` into
    # the GGUF file name.
    from ...custom import CustomRerankModelFamilyV2, register_rerank, unregister_rerank

    model_name = "custom_test_rerank_gguf"
    model_family = CustomRerankModelFamilyV2(
        model_name=model_name,
        type="normal",
        language=["en"],
        max_tokens=512,
        model_specs=[
            LlamaCppRerankSpecV1(
                model_format="ggufv2",
                model_id=None,
                model_revision=None,
                model_uri=str(tmp_path),
                model_file_name_template="bge-reranker-v2-m3-{quantization}.gguf",
                quantization="Q4_K_M",
                model_file_name_split_template=None,
                quantization_parts=None,
            )
        ],
    )
    # Engine registration checks that xllamacpp is importable.
    monkeypatch.setattr(XllamaCppRerankModel, "check_lib", classmethod(lambda _: True))

    register_rerank(model_family, False)
    try:
        model = create_rerank_model_instance(
            "mock", model_name, None, model_path=str(tmp_path)
        )
    finally:
        unregister_rerank(model_name)

    assert isinstance(model, XllamaCppRerankModel)
    assert model.model_family.model_engine == "llama.cpp"
    assert model._quantization == "Q4_K_M"


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
