.. _accessing_models:

================
Accessing Models
================


.. note:: Suppose you have started the Xinference server endpoint at ``http://0.0.0.0:8001``. 

Please refer to :ref:`Launching Models <launching_models>` guide for get your model running.

Using Clients
=============


LLM Model
---------
When the Ability of the LLM model includes "chat", we can converse with it using the model's chat interface:

.. code-block:: python

   from xinference.client import RESTfulClient
   client = RESTfulClient("http://0.0.0.0:8001")
   model = client.get_model(model_uid)

   chat_history = []
   prompt = "What is the largest animal?"
   model.chat(
       prompt,
       chat_history,
       generate_config={"max_tokens": 1024}
   )

The response will look like:

.. code-block:: json

   {
     "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
     "model": "56f69622-1e73-11ee-a3bd-9af9f16816c6",
     "object": "chat.completion",
     "created": 1688919187,
     "choices": [
       {
         "index": 0,
         "message": {
           "role": "assistant",
           "content": "The largest animal that has been scientifically measured is the blue whale..."
         },
         "finish_reason": "None"
       }
     ],
     "usage": {
       "prompt_tokens": -1,
       "completion_tokens": -1,
       "total_tokens": -1
     }
   }


Embedding Model
---------------

To interact with Xinference's embedding model, i.e., inputting a text and getting an embedding from ``RESTfulClient``:

.. code-block:: python

   from xinference.client import RESTfulClient
   client = RESTfulClient("http://0.0.0.0:8001")
   model = client.get_model(model_uid)
   model.create_embedding("write a poem.")

The response will be:

.. code-block:: json

  {
    "object": "list",
    "model": "3ef99480-496f-11ee-9009-c2c8e4cad3f6",
    "data": [
        {
          "index": 0,
          "object": "embedding",
          "embedding": [-0.003699747147038579]
        }
    ],
    "usage": {
        "prompt_tokens": 37,
        "total_tokens": 37
    }
  }

Using OpenAI Python SDK
=======================

Xinference provides an OpenAI-compatible RESTful interface. Thus, you can also use the OpenAI Python SDK to
chat with the model via the service's endpoint:

.. code-block:: python

   import openai
   import sys

   openai.api_base = "http://0.0.0.0:8001/v1"
   openai.api_key = ""

   for resp in openai.Completion.create(model=model_uid, prompt=prompt, max_tokens=512, stream=True):
       sys.stdout.write(resp.choices[0].text)
       sys.stdout.flush()
