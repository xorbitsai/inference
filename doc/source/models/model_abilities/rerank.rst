.. _rerank:

=====================
Rerank
=====================

Learn how to use rerank models in Xinference.


Introduction
================

Given a query and a list of documents, Rerank indexes the documents from most to least semantically relevant to the query.
Rerank models in Xinference can be invoked through the Rerank endpoint to rank a list of documents. 


Quickstart
================

We can try Rerank API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/rerank' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "query": "A man is eating pasta.",
        "documents": [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin."
        ]
      }'

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model(<MODEL_UID>)

    query = "A man is eating pasta."
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin."
    ]
    print(model.rerank(corpus, query))

  .. code-tab:: json output

    {
        "id": "480dca92-8910-11ee-b76a-c2c8e4cad3f5",
        "results": [{
            "index": 0,
            "relevance_score": 0.9999247789382935,
            "document": "A man is eating food."
        }, {
            "index": 1,
            "relevance_score": 0.2564932405948639,
            "document": "A man is eating a piece of bread."
        }, {
            "index": 3,
            "relevance_score": 0.00003955026841140352,
            "document": "A man is riding a horse."
        }, {
            "index": 2,
            "relevance_score": 0.00003742107219295576,
            "document": "The girl is carrying a baby."
        }, {
            "index": 4,
            "relevance_score": 0.00003739788007806055,
            "document": "A woman is playing violin."
        }]
    }


Multimodal reranking with jina-reranker-m0
===========================================

``jina-reranker-m0`` supports text and image reranking with a maximum context
length of 10,240 tokens. Text documents use the standard rerank API shown
above. To rerank image URLs or local image paths, pass ``doc_type="image"``::

    model.rerank(
        [
            "https://example.com/document-1.png",
            "https://example.com/document-2.png",
        ],
        "Which document describes a small language model?",
        doc_type="image",
    )

For an image query, also pass ``query_type="image"``.


Multimodal reranking with Nemotron-3.5-Rerank-8B
================================================

``Nemotron-3.5-Rerank-8B`` scores text queries against text documents, single
images, or combined image-text documents. Its context limit is 32,768 tokens,
including image tokens. Scores are raw relevance logits; higher is better.
Image queries, video, and multiple images in one document are unsupported.

Launch with one of ``sentence_transformers``, ``transformers``, or ``vllm``::

    xinference launch --model-type rerank --model-name Nemotron-3.5-Rerank-8B \
        --model-engine sentence_transformers --model-format pytorch

The model virtualenv settings pin the validated backend versions:
``sentence-transformers[image]==6.1.0`` with ``transformers==5.20.0`` for
Sentence Transformers, ``transformers==5.20.0`` for native Transformers, and
``vllm==0.32.0`` with ``transformers==5.19.0`` for vLLM. Use separate virtual
environments for these backends. Model virtualenv settings install their
respective dependencies automatically.

Hugging Face uses ``nvidia/Nemotron-3.5-Rerank-8B-BF16`` at revision ``main``;
ModelScope uses ``nv-community/Nemotron-3.5-Rerank-8B-BF16`` at revision ``master``.
Select the download hub with ``--download_hub huggingface`` or
``--download_hub modelscope`` when launching.

Use the returned model UID in the standard rerank endpoint::

    curl http://localhost:9997/v1/rerank \
        -H 'Content-Type: application/json' \
        -d '{
          "model": "<MODEL_UID>",
          "query": "Which document describes NVIDIA NeMo?",
          "documents": [
            "NeMo is a framework for developing AI models.",
            {"image": "https://example.com/page.png"},
            {"text": "NeMo architecture", "image": "https://example.com/page.png"}
          ],
          "top_n": 2,
          "return_documents": true
        }'

Images can be HTTP(S) URLs or base64 data URLs. OpenAI-style content dictionaries
with ``text`` and ``image_url`` blocks are also accepted. Returned documents
retain their original input structure. The vLLM adapter renders the checkpoint's
query/document template and uses classification pooling, preserving its image
placeholders and raw logits.

See the `official model card
<https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16>`_ for runtime examples.
