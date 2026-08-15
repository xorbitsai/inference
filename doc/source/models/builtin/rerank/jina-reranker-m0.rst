.. _models_builtin_jina-reranker-m0:

================
jina-reranker-m0
================

- **Model Name:** jina-reranker-m0
- **Languages:** en, zh, multilingual
- **Abilities:** text and image rerank

Specifications
^^^^^^^^^^^^^^

- **Model ID:** jinaai/jina-reranker-m0
- **Maximum Context Length:** 10,240 tokens

Execute the following command to launch the model::

   xinference launch --model-name jina-reranker-m0 --model-type rerank

Text documents use the standard rerank API. To rerank image URLs or local image
paths, pass ``doc_type="image"``::

   model.rerank(
       [
           "https://example.com/document-1.png",
           "https://example.com/document-2.png",
       ],
       "Which document describes a small language model?",
       doc_type="image",
   )

For an image query, also pass ``query_type="image"``.
