import shutil

import pytest

from .....client import Client

# Ensure rerank engines are properly initialized
# This addresses potential CI environment initialization issues
from ... import BUILTIN_RERANK_MODELS
from ...cache_manager import RerankCacheManager
from ...core import RerankModelFamilyV2, RerankSpecV1
from ..core import VLLMRerankModel

# Force import of the entire rerank module to ensure initialization



if "bge-reranker-base" in BUILTIN_RERANK_MODELS:
    # Force regeneration of engine configuration
    from ... import generate_engine_config_by_model_name

    generate_engine_config_by_model_name(BUILTIN_RERANK_MODELS["bge-reranker-base"])

TEST_MODEL_SPEC = RerankModelFamilyV2(
    version=2,
    model_name="bge-reranker-base",
    type="normal",
    max_tokens=512,
    language=["en", "zh"],
    model_specs=[
        RerankSpecV1(
            model_id="BAAI/bge-reranker-base",
            model_revision="465b4b7ddf2be0a020c8ad6e525b9bb1dbb708ae",
            model_format="pytorch",
        )
    ],
)


@pytest.mark.skipif(not VLLMRerankModel.check_lib(), reason="vllm not installed")
def test_model():
    model_path = None
    try:
        model_path = RerankCacheManager(TEST_MODEL_SPEC).cache()
        model = VLLMRerankModel("mock", model_path, TEST_MODEL_SPEC, "none")

        query = "A man is eating pasta."
        # With all sentences in the corpus
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
        model.load()
        scores = model.rerank(corpus, query, None, None, True, True)
        assert scores["results"][0]["index"] == 0
        assert scores["results"][0]["document"]["text"] == corpus[0]

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.skipif(not VLLMRerankModel.check_lib(), reason="vllm not installed")
def test_qwen3_vllm(setup):
    endpoint, _ = setup
    client = Client(endpoint)
    model_uid = client.launch_model(
        model_name="Qwen3-Reranker-0.6B",
        model_type="rerank",
        model_engine="vllm",
        max_num_batched_tokens=81920,  # Allow larger batch size for Qwen3
    )

    model = client.get_model(model_uid)
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
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

    scores = model.rerank(corpus, query, return_documents=True)
    assert scores["results"][0]["index"] == 1
