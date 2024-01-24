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


Quickstart
============

We can try it out either cURL, OpenAI Client, or via Xinference's python client:

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


You can find more examples of ``embed`` ability in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: LangChain Streamlit Doc Chat
      :link: https://github.com/xorbitsai/inference/blob/main/examples/LangChain_Streamlit_Doc_Chat.py
      
      Learn from an example demonstrating how to use embed API via LangChain


FAQ
========

Does the LLM in Xinference support Embeddings API?
------------------------------------------------------------

No. Xinference doesn't provide embed API for LLMs due to considerations of performance.


Does Embeddings API provides integration method for LangChain?
-----------------------------------------------------------------------------------

Yes, you can refer to the related sections in LangChain's respective official Xinference documentation.
Here is the link: `Text Embedding Models: Xinference <https://python.langchain.com/docs/integrations/text_embedding/xinference>`_ 