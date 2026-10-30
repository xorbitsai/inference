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

from typing import Optional
from unittest.mock import Mock, patch

import pytest

from ..cache_manager import EmbeddingCacheManager
from ..core import create_embedding_model_instance
from ..embed_family import EMBEDDING_ENGINES
from ..sentence_transformers.core import SentenceTransformerEmbeddingModel
from ..vllm.core import VLLMEmbeddingModel

MODEL = "bge-small-en-v1.5"


@pytest.mark.parametrize("enable_virtual_env", [False, True])
def test_invalid_embedding_engine_is_rejected_before_download(enable_virtual_env):
    cache = Mock(side_effect=AssertionError("An invalid engine reached caching"))
    with patch.object(EmbeddingCacheManager, "cache", cache):
        with pytest.raises(ValueError) as exc_info:
            create_embedding_model_instance(
                "uid", MODEL, "not-an-engine", enable_virtual_env=enable_virtual_env
            )
    assert (
        str(exc_info.value) == f"Model {MODEL} cannot be run on engine not-an-engine."
    )
    cache.assert_not_called()


@pytest.mark.parametrize(
    "model_engine,model_path",
    [
        ("sentence_transformers", None),
        (None, None),
        ("sentence_transformers", "/models/mine"),
    ],
)
def test_valid_embedding_engine_still_caches_the_model(
    model_engine: Optional[str], model_path: Optional[str]
):
    with patch.object(
        EmbeddingCacheManager, "cache", return_value="/cache/model"
    ) as cache:
        model = create_embedding_model_instance(
            "uid",
            MODEL,
            model_engine,
            model_path=model_path,
            enable_virtual_env=False,
        )
    # An explicit model_path is used as given and never cached.
    assert cache.call_count == (0 if model_path else 1)
    assert isinstance(model, SentenceTransformerEmbeddingModel)
    assert model._model_path == (model_path or "/cache/model")
    assert model.model_family.model_engine == "sentence_transformers"


def test_virtualenv_marker_engine_is_still_accepted_and_cached():
    # vLLM is a declared virtualenv engine of this model but not a registered one
    # here: only the virtualenv-aware lookup accepts it.
    model_name = "Qwen3-VL-Embedding-2B"
    engines = dict(EMBEDDING_ENGINES.get(model_name, {}))
    engines.pop("vllm", None)
    with (
        patch.dict(EMBEDDING_ENGINES, {model_name: engines}),
        patch.object(
            EmbeddingCacheManager, "cache", return_value="/cache/model"
        ) as cache,
    ):
        model = create_embedding_model_instance(
            "uid", model_name, "vllm", enable_virtual_env=True
        )
    cache.assert_called_once_with()
    assert isinstance(model, VLLMEmbeddingModel)
