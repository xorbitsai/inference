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
import torch

from ...cache_manager import EmbeddingCacheManager as CacheManager
from ...core import (
    EmbeddingModelFamilyV2,
    TransformersEmbeddingSpecV1,
    create_embedding_model_instance,
)
from ..core import _normalize_transformers_dtype_kwargs, _transformers_dtype_compat

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
            model_hub="modelscope",
        )
    ],
)


@pytest.mark.parametrize(
    ("version", "kwargs", "expected"),
    [
        (
            "4.57.6",
            {"dtype": torch.float16},
            {"torch_dtype": torch.float16},
        ),
        (
            "5.0.0",
            {"torch_dtype": torch.bfloat16},
            {"dtype": torch.bfloat16},
        ),
        (
            "5.0.0",
            {"dtype": torch.float32},
            {"dtype": torch.float32},
        ),
        (
            "5.0.0",
            {"dtype": torch.float32, "torch_dtype": torch.float16},
            {"dtype": torch.float32},
        ),
    ],
)
def test_normalize_transformers_dtype_kwargs(version, kwargs, expected):
    original = dict(kwargs)
    assert _normalize_transformers_dtype_kwargs(kwargs, version) == expected
    assert kwargs == original


def test_transformers_dtype_compat_restores_automodel(monkeypatch):
    import transformers

    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "model"

    original_descriptor = FakeAutoModel.__dict__["from_pretrained"]
    monkeypatch.setattr(transformers, "AutoModel", FakeAutoModel)
    monkeypatch.setattr(transformers, "__version__", "4.57.6")

    with _transformers_dtype_compat():
        assert FakeAutoModel.from_pretrained("model", dtype=torch.float16) == "model"

    assert calls == [(("model",), {"torch_dtype": torch.float16})]
    assert FakeAutoModel.__dict__["from_pretrained"] is original_descriptor


# todo Refer to the return format of sentence_transformer
async def test_embedding_model_with_flag():
    model_path = None
    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_embedding_model_instance(
            "mook", "bge-small-en-v1.5", "flag", model_path=model_path
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = await model.create_embedding(input_text, **{"return_sparse": True})
        assert len(r["data"]) == 1

        r = await model.create_embedding(input_text)
        assert len(r["data"][0]["embedding"]) == 384

        # input is a lit
        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ]
        # test sparse and dense
        r = await model.create_embedding(input_texts, **{"return_sparse": True})
        assert len(r["data"]) == 4

        r = await model.create_embedding(input_texts)
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
