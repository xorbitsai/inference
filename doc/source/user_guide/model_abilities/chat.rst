.. _chat:

=====================
Chat & Generate
=====================

Learn how to chat with LLMs in Xinference.

Introduction
============

Models equipped with ``chat`` or ``generate`` abilities are frequently referred to as large language models (LLM) or text generation models.
These models are designed to respond with text outputs to the inputs they receive, commonly known as "prompts".
Typically, one can direct these models using specific instructions or by providing concrete examples illustrating
how to accomplish a task.

Models with ``generate`` capacities are typically pre-trained large language models. On the other hand, models equipped with ``chat``
capabilities are finely-tuned and aligned LLMs, optimized for dialogues use case. In most cases, models ending with "chat" 
(e.g. ``llama-2-chat``, ``qwen-chat``, etc) are identified as having ``chat`` capabilities. 


The Chat API and Generate API offer two distinct approaches for interacting with LLMs:

* The Chat API (like OpenAI's `Chat Completion API <https://platform.openai.com/docs/api-reference/chat/create>`__)
  can conduct multi-turn conversations.

* The Generate API (like OpenAI's legacy `Completions API <https://platform.openai.com/docs/api-reference/completions/create>`__)
  allows you to generate text based on a text prompt.

.. list-table:: 
   :widths: 25 25 50
   :header-rows: 1

   * - MODEL ABILITY
     - API ENDPOINT
     - OpenAI-compatible ENDPOINT

   * - chat
     - Chat API
     - /v1/chat/completions

   * - generate
     - Generate API
     - /v1/completions


Supported models
-------------------

You can examine the abilities of all the :ref:`builtin LLM models in Xinference <models_llm_index>`.

Quickstart
===================

Chat API 
------------

The Chat API mimics OpenAI's `Chat Completion API <https://platform.openai.com/docs/api-reference/chat/create>`__. 
We can try it out either cURL, OpenAI Client, or via Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/chat/completions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is the largest animal?"
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7        
      }'

  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty", 
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.chat.completions.create(
        model=("<MODEL_UID>",
        messages=[
            {
                "content": "What is the largest animal?",
                "role": "user",
            }
        ],
        max_tokens: 512,
        temperature: 0.7        
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import RESTfulClient

    client = RESTfulClient("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<MODEL_UID>")
    print(model.chat(
        prompt="What is the largest animal?",
        system_prompt="You are a helpful assistant.",
        chat_history=[],
        generate_config={
          "max_tokens": 512,
          "temperature": 0.7
        }        
    ))

  .. code-tab:: json output

    {
      "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
      "model": "<MODEL_UID>",
      "object": "chat.completion",
      "created": 1688919187,
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life."
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


You can find more examples of Chat API in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Gradio Chat
      :link: https://github.com/xorbitsai/inference/blob/main/examples/gradio_chatinterface.py

      Learn from an example of utilizing the Chat API with the Xinference Python client.


Generate API 
----------------

The Generate API mirrors OpenAI's legacy `Completions API <https://platform.openai.com/docs/api-reference/completions/create>`__.

The difference between the Generate API and the Chat API lies primarily in the form of input. Opposite to the Chat API that takes
a list of messages as input, the Generate API accepts a freeform text string named "prompt".

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/completions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "prompt": "What is the largest animal?",
        "max_tokens": 512,
        "temperature": 0.7
      }'

  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(api_key="cannot be empty", base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1")
    client.chat.completions.create(
        model=("<MODEL_UID>",
        prompt="What is the largest animal?"
        max_tokens=512,
        temperature=0.7
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import RESTfulClient

    client = RESTfulClient("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<MODEL_UID>")
    print(model.generate(
        prompt="What is the largest animal?",
        generate_config={
          "max_tokens": 512,
          "temperature": 0.7
        }
    ))

  .. code-tab:: json output

    {
      "id": "cmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
      "model": "<MODEL_UID>",
      "object": "text_completion",
      "created": 1688919187,
      "choices": [
        {
          "index": 0,
          "text": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life.",
          "finish_reason": "None"
        }
      ],
      "usage": {
        "prompt_tokens": -1,
        "completion_tokens": -1,
        "total_tokens": -1
      }
    }




FAQ
========

Does Xinference's LLM provide integration methods for LangChain or LlamaIndex?
-----------------------------------------------------------------------------------

Yes, you can refer to the related sections in their respective official Xinference documentation. Here are the links:

* `LangChain LLMs: Xinference <https://python.langchain.com/docs/integrations/llms/xinference>`__

* `LlamaIndex LLM integrations: Xinference  <https://docs.llamaindex.ai/en/stable/examples/llm/xinference_local_deployment.html>`__
