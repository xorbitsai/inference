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

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from .....core.exceptions import InvalidModelRequestError
from ....batch import BatchMixin
from .. import core as core_module
from ..core import SentenceTransformerEmbeddingModel


def _make_model(monkeypatch):
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = object
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(core_module, "ensure_wemm_video_reader", lambda: None)

    model = SentenceTransformerEmbeddingModel.__new__(SentenceTransformerEmbeddingModel)
    model.model_family = SimpleNamespace(model_name="WeMM-Embedding-2B")
    model._model_name = "WeMM-Embedding-2B"
    model._model_uid = "wemm"
    model._embedder = None
    model._native_embedding_dimensions = 2048
    model._kwargs = {}
    model._model = MagicMock()
    model._model.__getitem__.return_value.auto_model.config = SimpleNamespace(
        matryoshka_dimensions=[64, 128, 256, 512, 1024, 2048]
    )
    model._model.encode.side_effect = lambda inputs, **kwargs: np.ones(
        (len(inputs), kwargs.get("truncate_dim", 2048)), dtype=np.float32
    )
    model._fix_langchain_openai_inputs = lambda value: value
    model._clean_cache_if_needed = lambda *args, **kwargs: None
    return model


def test_wemm_sentence_transformers_keeps_interleaved_dict(monkeypatch):
    model = _make_model(monkeypatch)
    input_value = {
        "role": "user",
        "content": [
            {"type": "image", "image": "x.jpg"},
            {"type": "text", "text": "caption"},
        ],
    }

    result = model._create_embedding(input_value, dimensions=256)

    assert len(result["data"]) == 1
    assert len(result["data"][0]["embedding"]) == 256
    args, kwargs = model._model.encode.call_args
    assert args[0] == [[input_value]]
    assert kwargs["truncate_dim"] == 256
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 1


def test_wemm_sentence_transformers_preserves_batch_cardinality(monkeypatch):
    model = _make_model(monkeypatch)
    result = model._create_embedding(["text", {"image": "x.jpg"}, {"video": "x.mp4"}])
    assert len(result["data"]) == 3


async def test_unsupported_wemm_dimensions_do_not_fail_concurrent_requests(monkeypatch):
    model = _make_model(monkeypatch)
    BatchMixin.__init__(
        model, model.create_embedding, batch_size=8, batch_interval=0.01
    )
    try:
        invalid, valid = await asyncio.gather(
            model.create_embedding("first", dimensions=96),
            model.create_embedding("second", dimensions=128),
            return_exceptions=True,
        )
    finally:
        if model._process_batch_task is not None:
            model._process_batch_task.cancel()
            await asyncio.gather(model._process_batch_task, return_exceptions=True)

    assert isinstance(invalid, InvalidModelRequestError)
    assert "WeMM-Embedding dimensions" in invalid.client_message
    assert len(valid["data"][0]["embedding"]) == 128
