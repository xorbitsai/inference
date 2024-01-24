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

We can try it out either cURL, OpenAI Client, or via Xinference's python client:

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

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_HOST>")
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
