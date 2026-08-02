import shutil
from unittest.mock import MagicMock

import pytest

from ...cache_manager import RerankCacheManager
from ...core import RerankModelFamilyV2, TransformersRerankSpecV1
from ..core import SentenceTransformerRerankModel

TEST_MODEL_SPEC = RerankModelFamilyV2(
    version=2,
    model_name="bge-reranker-base",
    type="normal",
    max_tokens=512,
    language=["en", "zh"],
    model_specs=[
        TransformersRerankSpecV1(
            model_id="BAAI/bge-reranker-base",
            model_revision="465b4b7ddf2be0a020c8ad6e525b9bb1dbb708ae",
            model_format="pytorch",
        )
    ],
)


async def test_model():
    model_path = None
    try:
        model_path = RerankCacheManager(TEST_MODEL_SPEC).cache()
        model = SentenceTransformerRerankModel(
            "mock", model_path, TEST_MODEL_SPEC, "none"
        )

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
        scores = await model.rerank(corpus, query, None, None, True, True)
        assert scores["results"][0]["index"] == 0
        assert scores["results"][0]["document"]["text"] == corpus[0]

        n_tokens = scores["meta"]["tokens"]["input_tokens"]
        tokenizer = model._model.tokenizer
        expect_n_tokens = sum(len(tokenizer.tokenize([query, d])) for d in corpus)
        assert n_tokens >= expect_n_tokens

    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


def test_jina_reranker_v35_batch_isolation():
    """Regression test: batched requests with the same query but different
    document lists must each be passed to model.rerank() separately.

    Jina v3.5 is genuinely listwise – its native implementation computes
    query embeddings and block weights from the full candidate set, so
    mixing documents from independent requests would change scores even
    when the query text is identical.
    """
    model_family = RerankModelFamilyV2(
        version=2,
        model_name="jina-reranker-v3.5",
        type="normal",
        max_tokens=8192,
        language=["en"],
        model_specs=[
            TransformersRerankSpecV1(
                model_id="jinaai/jina-reranker-v3.5",
                model_format="pytorch",
            )
        ],
    )

    # Create instance without calling __init__ (no model download needed)
    model = SentenceTransformerRerankModel.__new__(SentenceTransformerRerankModel)
    model.model_family = model_family
    model._vl_reranker = None

    # Track which (query, documents) pairs are sent to the native reranker
    rerank_calls: list = []

    def mock_rerank(query, documents):
        rerank_calls.append((query, list(documents)))
        # Return one result per document, sorted by index
        return [
            {"index": i, "relevance_score": 0.9 - i * 0.1}
            for i in range(len(documents))
        ]

    mock_model = MagicMock()
    mock_model.rerank = mock_rerank
    model._model = mock_model

    # Simulate the batch handler coalescing two requests with the SAME
    # query but different document lists:
    #   Request 1: query="Q", docs=["A", "B"]       (offset 0, size 2)
    #   Request 2: query="Q", docs=["C", "D", "E"]  (offset 2, size 3)
    documents = ["A", "B", "C", "D", "E"]
    query = ["Q"] * 5
    batch_offsets = [(0, 2), (2, 3)]

    scores = model._rerank(
        documents,
        query,
        None,  # top_n
        None,  # max_chunks_per_doc
        True,  # return_documents
        False,  # return_len
        _batch_offsets=batch_offsets,
    )

    # The native reranker must be called once per request, not once total
    assert len(rerank_calls) == 2, f"Expected 2 rerank() calls, got {len(rerank_calls)}"
    # First call: only documents from request 1
    assert rerank_calls[0][0] == "Q"
    assert rerank_calls[0][1] == ["A", "B"]
    # Second call: only documents from request 2
    assert rerank_calls[1][0] == "Q"
    assert rerank_calls[1][1] == ["C", "D", "E"]
    # Scores should be mapped back to the correct positions
    assert len(scores) == 5
    assert scores[0] == pytest.approx(0.9)  # A
    assert scores[1] == pytest.approx(0.8)  # B
    assert scores[2] == pytest.approx(0.9)  # C
    assert scores[3] == pytest.approx(0.8)  # D
    assert scores[4] == pytest.approx(0.7)  # E
