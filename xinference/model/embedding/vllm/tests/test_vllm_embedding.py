# Copyright 2022-2026 XProbe Inc.
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

from .....client import Client
from ...cache_manager import EmbeddingCacheManager as CacheManager
from ...core import (
    EmbeddingModelFamilyV2,
    TransformersEmbeddingSpecV1,
    create_embedding_model_instance,
)
from ..core import VLLMEmbeddingModel

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


@pytest.mark.skipif(VLLMEmbeddingModel.check_lib() != True, reason="vllm not installed")
async def test_embedding_model_with_vllm():
    model_path = None

    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_embedding_model_instance(
            "mook",
            "bge-small-en-v1.5",
            "vllm",
            model_path=model_path,
        )
        model.load()

        # input is a string
        input_text = "what is the capital of China?"

        # test sparse and dense
        r = await model.create_embedding(input_text)
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
        r = await model.create_embedding(input_texts)
        assert len(r["data"]) == 4
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.skipif(VLLMEmbeddingModel.check_lib() != True, reason="vllm not installed")
async def test_embedding_model_with_vllm_long_text():
    """Test embedding model with text that exceeds 512 tokens."""
    model_path = None

    try:
        model_path = CacheManager(TEST_MODEL_SPEC).cache()

        model = create_embedding_model_instance(
            "bge-small-en-v1.5",
            "bge-small-en-v1.5",
            "vllm",
            "pytorch",
            "none",
            model_path=model_path,
        )
        model.load()
        model.wait_for_load()

        # Create a long text that exceeds 512 tokens
        # Each word is roughly 1-2 tokens, so we create a text with ~600 words to ensure >512 tokens
        base_text = "This is a very long text that is designed to exceed the maximum token limit of 512 tokens for the embedding model. "
        long_text = base_text * 20  # Repeat to create a very long text

        # Add more content to ensure we definitely exceed 512 tokens
        additional_content = (
            "We are testing the behavior of the VLLM embedding model when processing text that contains "
            "significantly more tokens than the specified maximum limit. This test is important because "
            "it helps us understand how the model handles token truncation or other processing strategies "
            "when dealing with extremely long input sequences. The model should either truncate the input "
            "or handle it gracefully without crashing. We expect the embedding dimension to remain consistent "
            "at 384 dimensions regardless of the input length, as the model architecture should maintain "
            "the same output dimensionality. This comprehensive test ensures robustness and reliability "
            "of the embedding generation process under edge case conditions with very long text inputs."
        )

        long_text += additional_content

        # Verify the text is indeed long (should be well over 512 tokens)
        print(f"Long text length in characters: {len(long_text)}")
        print(f"Approximate word count: {len(long_text.split())}")

        # Test single long text
        r = await model.create_embedding(long_text)
        assert len(r["data"]) == 1
        assert len(r["data"][0]["embedding"]) == 384

        # Verify that token count is reported in usage
        assert "usage" in r
        assert "prompt_tokens" in r["usage"]
        print(f"Reported token count: {r['usage']['prompt_tokens']}")

        # Test multiple long texts
        long_texts = [
            long_text,
            long_text + " Additional content for the second long text.",
            "Another very long text: " + long_text[: len(long_text) // 2],
        ]

        r_multiple = await model.create_embedding(long_texts)
        assert len(r_multiple["data"]) == 3
        for d in r_multiple["data"]:
            assert len(d["embedding"]) == 384

        # Verify total token usage for multiple texts
        assert r_multiple["usage"]["prompt_tokens"] > r["usage"]["prompt_tokens"]
        print(
            f"Total token count for multiple texts: {r_multiple['usage']['prompt_tokens']}"
        )

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.skipif(VLLMEmbeddingModel.check_lib() != True, reason="vllm not installed")
def test_change_dim(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    model_uid = client.launch_model(
        model_name="Qwen3-Embedding-0.6B",
        model_type="embedding",
        model_engine="vllm",
    )

    model = client.get_model(model_uid)

    content = (
        "We are testing the behavior of the VLLM embedding model when processing text that contains "
        "significantly more tokens than the specified maximum limit. This test is important because "
        "it helps us understand how the model handles token truncation or other processing strategies "
        "when dealing with extremely long input sequences. The model should either truncate the input "
        "or handle it gracefully without crashing. We expect the embedding dimension to remain consistent "
        "at 384 dimensions regardless of the input length, as the model architecture should maintain "
        "the same output dimensionality. This comprehensive test ensures robustness and reliability "
        "of the embedding generation process under edge case conditions with very long text inputs."
    )

    embeds = model.create_embedding(content, dimensions=500)
    assert len(embeds["data"][0]["embedding"]) == 500

    embeds = model.create_embedding(content)
    assert len(embeds["data"][0]["embedding"]) == 1024
