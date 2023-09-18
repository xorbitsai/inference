.. _accessing_models:

================
Accessing Models
================


.. note:: Suppose you have started the Xinference server endpoint at ``http://127.0.0.1:9997``. 

Please refer to :ref:`Launching Models <launching_models>` guide to get your model running.

Using Xinference Client
=============


Large Language Model
-----------------------
When the abilities of the LLM include "chat," we can converse with it using the model's chat interface:

.. code-block:: python

   from xinference.client import RESTfulClient
   client = RESTfulClient("http://127.0.0.1:9997")
   model = client.get_model("f543d078-55f1-11ee-8fc0-0a28cc89f433")  # The model UID of the model you just launched.

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
   "id":"chat3626defa-55f2-11ee-8f6c-0a28cc89f433",
   "object":"chat.completion",
   "created":1695020931,
   "model":"f543d078-55f1-11ee-8fc0-0a28cc89f433",
   "choices":[
      {
         "index":0,
         "message":{
            "role":"assistant",
            "content":"The largest animal in the world is the blue whale. It can grow up to 100 feet long and weigh up to 200 tons. It is the largest mammal on Earth and the largest living thing that ever existed. Blue whales are found in all of the world's oceans and are the only animals that can communicate with each other through a series of clicks and whistles. They are also the fastest animals in the world, able to swim at speeds of up to 35 miles per hour."
         },
         "finish_reason":"stop"
      }
   ],
   "usage":{
      "prompt_tokens":48,
      "completion_tokens":117,
      "total_tokens":165
   }
}

Embedding Model
---------------

To interact with Xinference's embedding model, i.e., inputting a text and getting an embedding from ``RESTfulClient``:

.. code-block:: python

   from xinference.client import RESTfulClient
   client = RESTfulClient("http://127.0.0.1:9997")
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
          "embedding": [-0.003699747147038579, ...]
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

   openai.api_base = "http://127.0.0.1:9997/v1"
   openai.api_key = ""

   for resp in openai.Completion.create(model=model_uid, prompt=prompt, max_tokens=512, stream=True):
       sys.stdout.write(resp.choices[0].text)
       sys.stdout.flush()
