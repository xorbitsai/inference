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

import pytest
import torch

from ...embed_family import BUILTIN_EMBEDDING_MODELS
from ..core import SentenceTransformerEmbeddingModel

# From gte-Qwen2-7B-instruct's config_sentence_transformers.json.
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer "
    "the query\nQuery: "
)


class _GteQwen2Encoder:
    device = torch.device("cpu")
    prompts = {"query": QUERY_PROMPT}
    default_prompt_name = None
    truncate_dim = None

    def __init__(self):
        self.tokenized = []

    def eval(self):
        pass

    def to(self, *args, **kwargs):
        pass

    def tokenize(self, sentences):
        self.tokenized.append(list(sentences))
        return {
            "input_ids": torch.ones((len(sentences), 4), dtype=torch.long),
            "attention_mask": torch.ones((len(sentences), 4), dtype=torch.long),
        }

    def forward(self, features, **kwargs):
        rows = features["attention_mask"].shape[0]
        return {"sentence_embedding": torch.ones((rows, 3))}


@pytest.mark.parametrize(
    "kwargs,prompt",
    [
        ({}, QUERY_PROMPT),
        # Clients that serialize an unset prompt as null keep the default.
        ({"prompt": None}, QUERY_PROMPT),
        ({"prompt_name": "query"}, QUERY_PROMPT),
        # The model card encodes documents without a prompt.
        ({"prompt_name": None}, ""),
        ({"prompt": "Instruct: find code\nQuery: "}, "Instruct: find code\nQuery: "),
    ],
    ids=["default", "null-prompt", "query-prompt-name", "no-prompt", "custom-prompt"],
)
async def test_request_prompt_overrides_the_default_query_prompt(kwargs, prompt):
    family = BUILTIN_EMBEDDING_MODELS["gte-Qwen2"][0]
    model = SentenceTransformerEmbeddingModel("test", "/unused", family)
    model._model = encoder = _GteQwen2Encoder()
    try:
        result = await model.create_embedding(
            ["summit define"], model_uid="gte", **kwargs
        )
    finally:
        model._process_batch_task.cancel()
        await asyncio.gather(model._process_batch_task, return_exceptions=True)

    assert len(result["data"]) == 1
    # The last tokenize call is the embedded batch; an earlier one measures
    # the prompt length.
    assert encoder.tokenized[-1] == [prompt + "summit define"]
