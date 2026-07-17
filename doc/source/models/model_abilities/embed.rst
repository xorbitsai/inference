.. _embed:

=====================
Embeddings
=====================


Learn how to create text embeddings in Xinference.


Introduction
============

Text embeddings are used to quantify how related different pieces of text are. They can be used in various applications including
search, clustering, recommendations, anomaly detection, diversity measurement, and classification.

An embedding is a vector of floating point numbers. The proximity between two vectors serves as an indicator of their similarity. 
Less distance implies a higher correlation, while a larger distance indicates reduced correlation.

Embedding models in Xinference can be invoked through the Embeddings API to create embeddings. 
The Embeddings API mimics OpenAI's `create embeddings API <https://platform.openai.com/docs/api-reference/embeddings/create>`_.

.. list-table:: 
   :widths: 25 50
   :header-rows: 1

   * - API ENDPOINT
     - OpenAI-compatible ENDPOINT

   * - Embeddings API
     - /v1/embeddings


Supported models
-------------------

You can examine all the :ref:`builtin embedding models in Xinference <models_embedding_index>`.

Embedding engines
-------------------

When launching an embedding model, you can pick the serving engine with the
``model_engine`` parameter (``--model-engine`` on the command line):

* ``sentence_transformers``: the default engine, available for all embedding
  models.
* ``vllm``: high-throughput serving for supported model families — currently
  models whose names start with ``bge``, ``gte``, ``text2vec``, ``m3e``,
  ``Qwen3``, or ``bce`` (e.g. ``bce-embedding-base_v1``).
* ``flag``: FlagEmbedding-based engine; also supports hybrid (sparse+dense)
  output, see the FAQ below.
* ``llama.cpp``: serve GGUF-format embedding models.

Truncating input
-------------------

The Embeddings API accepts an optional ``truncate_prompt_tokens`` parameter
to cap the token length of each input before encoding:

* unset / ``null``: no truncation.
* a positive integer ``N``: truncate each input to at most ``N`` tokens.
* ``-1``: truncate to the model's own maximum input length.

.. code-block:: bash

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/embeddings' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "input": "A very long document ...",
        "truncate_prompt_tokens": 512
      }'


Quickstart
============

We can try Embeddings API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/embeddings' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "input": "What is the capital of China?"
      }'

  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
      api_key="cannot be empty", 
      base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.embeddings.create(
      model=model_uid, 
      input=["What is the capital of China?"]
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    input = "What is the capital of China?"
    model.create_embedding(input)

  .. code-tab:: json output

    {
      "object": "list",
      "model": "<MODEL_UID>",
      "data": [{
        "index": 0,
        "object": "embedding",
        "embedding": [
          -0.014207549393177032, 
          -0.01832585781812668, 
          ...
          -0.03009396605193615,
          0.05420297756791115]
      }],
      "usage": {
        "prompt_tokens": 37,
        "total_tokens": 37
      }
    }


FAQ
========

Does the LLM in Xinference support Embeddings API?
------------------------------------------------------------

No. Xinference doesn't provide embed API for LLMs due to considerations of performance.


Does Embeddings API provides integration method for LangChain?
-----------------------------------------------------------------------------------

Yes, you can refer to the related sections in LangChain's respective official Xinference documentation.
Here is the link: `Text Embedding Models: Xinference <https://python.langchain.com/docs/integrations/text_embedding/xinference>`_ 


Does Embeddings API support hrbrid model?
-----------------------------------------------------------------------------------

Yes, you can use ``flag`` as the engine to deploy the model and call Embeddings API by setting the extra parameter ``return_parse=True`` which will return sparse vectors.