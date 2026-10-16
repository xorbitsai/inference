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

import base64
import json
import struct
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xinference.api import restful_api


async def _create_embedding(
    embeddings: list, encoding_format: Optional[str]
) -> tuple[Dict[str, Any], MagicMock]:
    api = MagicMock()
    api._report_error_event = AsyncMock()
    model = MagicMock(uid="replica-uid")
    model.create_embedding = AsyncMock(
        return_value=json.dumps(
            {
                "object": "list",
                "model": "embed-model",
                "model_replica": "replica-uid",
                "data": [
                    {"index": i, "object": "embedding", "embedding": embedding}
                    for i, embedding in enumerate(embeddings)
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            }
        ).encode()
    )
    payload: Dict[str, Any] = {"model": "embed-model", "input": ["a", "b"]}
    if encoding_format is not None:
        payload["encoding_format"] = encoding_format
    request = MagicMock()
    request.json = AsyncMock(return_value=payload)

    with patch.object(restful_api, "require_model", AsyncMock(return_value=model)):
        response = await restful_api.RESTfulAPI.create_embedding(api, request)

    assert response.media_type == "application/json"
    return json.loads(response.body), model


@pytest.mark.asyncio
async def test_create_embedding_base64_encodes_float32_vectors():
    vectors = [[0.5, -1.0, 2.0], [0.1, 0.2, 0.3]]
    body, model = await _create_embedding(vectors, "base64")

    for item, vector in zip(body["data"], vectors):
        assert isinstance(item["embedding"], str)
        raw = base64.b64decode(item["embedding"])
        decoded = struct.unpack(f"<{len(vector)}f", raw)
        assert decoded == pytest.approx(vector)
    assert body["usage"] == {"prompt_tokens": 2, "total_tokens": 2}
    # encoding_format is applied by the API layer, not forwarded to the model.
    assert "encoding_format" not in model.create_embedding.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding_format", [None, "float"])
async def test_create_embedding_float_keeps_lists(encoding_format):
    vectors = [[0.5, -1.0, 2.0], [0.1, 0.2, 0.3]]
    body, _ = await _create_embedding(vectors, encoding_format)

    assert [item["embedding"] for item in body["data"]] == vectors


@pytest.mark.asyncio
async def test_create_embedding_base64_keeps_sparse_embeddings():
    sparse = {"hello": 0.25, "world": 0.5}
    body, _ = await _create_embedding([sparse], "base64")

    assert body["data"][0]["embedding"] == sparse
