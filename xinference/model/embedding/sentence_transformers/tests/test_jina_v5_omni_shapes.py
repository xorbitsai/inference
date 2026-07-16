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

"""Regression tests for jina-embeddings-v5-omni input/output shape handling.

These tests exercise the dict/str dispatch path and the tail re-wrap
interaction without spinning up a real SentenceTransformer model.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def omni_model():
    """Build a minimal SentenceTransformerEmbeddingModel-like object.

    Only the bits exercised by ``_create_embedding`` are populated; everything
    else stays as ``MagicMock`` so unrelated attribute access does not blow up.
    """
    from ..core import SentenceTransformerEmbeddingModel

    inst = SentenceTransformerEmbeddingModel.__new__(SentenceTransformerEmbeddingModel)
    inst.model_family = SimpleNamespace(model_name="jina-embeddings-v5-omni-nano")
    inst._model_name = "jina-embeddings-v5-omni-nano"
    inst._model_uid = "uid"
    inst._embedder = None
    inst._kwargs = {}
    # mock the underlying SentenceTransformer.encode
    inst._model = MagicMock()
    # default behavior: return (1, 4) ndarray to simulate single-item batch
    inst._model.encode = MagicMock(
        side_effect=lambda objs, **kw: np.ones((len(objs), 4), dtype=np.float32)
    )
    inst._fix_langchain_openai_inputs = lambda x: x
    inst._clean_cache_if_needed = lambda *a, **kw: None
    inst._text_length = lambda s: 1
    return inst


class TestOmniSingleString:
    """Single ``str`` input must produce a flat 1-D vector, not nested."""

    def test_single_string_returns_flat_vector(self, omni_model):
        out = omni_model._create_embedding("hello")
        assert len(out["data"]) == 1
        emb = out["data"][0]["embedding"]
        assert isinstance(emb, list)
        # must be flat list of floats, not nested
        assert all(isinstance(v, float) for v in emb)
        assert len(emb) == 4


class TestOmniSingleDict:
    """Single ``dict`` input (e.g. {"image": "..."}) must not iterate keys."""

    def test_single_image_dict_returns_flat_vector(self, omni_model):
        out = omni_model._create_embedding({"image": "/tmp/x.jpg"})
        assert len(out["data"]) == 1
        emb = out["data"][0]["embedding"]
        assert all(isinstance(v, float) for v in emb)
        # the underlying encode received the URL, not the dict key
        called_with = omni_model._model.encode.call_args.args[0]
        assert called_with == ["/tmp/x.jpg"]


class TestOmniList:
    """List inputs preserve cardinality and ordering."""

    def test_list_of_strings(self, omni_model):
        out = omni_model._create_embedding(["a", "b", "c"])
        assert len(out["data"]) == 3
        for d in out["data"]:
            assert all(isinstance(v, float) for v in d["embedding"])

    def test_mixed_list_text_and_image_dict(self, omni_model):
        out = omni_model._create_embedding([{"text": "hello"}, {"image": "/tmp/y.jpg"}])
        assert len(out["data"]) == 2
        called_with = omni_model._model.encode.call_args.args[0]
        assert called_with == ["hello", "/tmp/y.jpg"]
