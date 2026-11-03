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

Chat Models
===================

Chat API 
------------

The Chat API mimics OpenAI's `Chat Completion API <https://platform.openai.com/docs/api-reference/chat/create>`__. 
We can try Chat API out either via cURL, OpenAI Client, or Xinference's python client:

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
        model="<MODEL_UID>",
        messages=[
            {
                "content": "What is the largest animal?",
                "role": "user",
            }
        ],
        max_tokens=512,
        temperature=0.7
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import RESTfulClient

    client = RESTfulClient("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<MODEL_UID>")
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the largest animal?"}]
    model.chat(
        messages,
        generate_config={
          "max_tokens": 512,
          "temperature": 0.7
        }        
    )

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


Reasoning effort and thinking budgets
-------------------------------------

For models with declared native support, OpenAI-compatible chat requests accept
``reasoning_effort``. Xinference forwards it to the model's official template or
encoder without adding generic system instructions. Supported values are exposed
in the model description's ``reasoning_effort`` metadata. A ``reasoning`` ability
alone does not imply effort support.

The Web UI's chat settings show effort controls only for models with declared
support. Numeric effort is available when the model declares an integer range.
The thinking-token budget appears only when all serving decoders report
runtime support. Leave a control at its default to omit the parameter; when
setting a token budget, choose a larger positive output-token limit. The model
details API exposes runtime budget support as ``supports_thinking_budget``.

GLM-5.3 accepts ``low``, ``high``, and ``max``; it has no ``medium`` level.
DeepSeek-V4.1-Flash-1017 accepts integers from 1 to 100, or ``low`` (50),
``high`` (75), and ``max`` (100). These numbers express effort, not token counts.
Undeclared effort support returns HTTP 501. Invalid values or conflicting
request-level and template-level values return HTTP 400.

.. code-block:: python

   client.chat.completions.create(
       model="YOUR_GLM_5_3_MODEL_UID",
       messages=[{"role": "user", "content": "Solve this problem."}],
       reasoning_effort="high",
       max_completion_tokens=8192,
   )

An independent Xinference extension, ``thinking_budget``, limits thinking tokens
on compatible vLLM, SGLang, and MLX runtimes. Send it through the OpenAI SDK's
``extra_body`` or as ``chat_template_kwargs.thinking_budget``. It must be a
positive integer smaller than ``max_completion_tokens`` (or ``max_tokens``),
and enables thinking. Reaching the budget closes thinking and leaves the
remaining output allowance for the answer. Unsupported runtimes return HTTP 501
before streaming starts. SGLang requires ``enable_strict_thinking=True``.
MLX Xavier, forced tool choice, and structured output cannot be combined with
OpenAI thinking budgets.

.. code-block:: python

   client.chat.completions.create(
       model="YOUR_REASONING_MODEL_UID",
       messages=[{"role": "user", "content": "Solve this problem."}],
       max_completion_tokens=8192,
       extra_body={"thinking_budget": 2048},
   )

Custom model registrations can declare ``reasoning_effort`` with a ``values``
list, an optional inclusive ``integer_range``, and optional numeric ``aliases``.
Only declare values consumed by the actual template or encoder. Models derived
from a registered ``model_family`` inherit its declaration unless they supply
their own.

.. code-block:: json

   "reasoning_effort": {
     "values": ["low", "high", "max"],
     "integer_range": [1, 100],
     "aliases": {"low": 50, "high": 75, "max": 100}
   }


Hybrid Thinking Models
----------------------

Some LLMs are marked as ``hybrid`` and can run with or without thinking mode.

.. versionadded:: v1.17.0
  Request-level ``enable_thinking`` is added in v1.17.0

Xinference exposes a request-level ``enable_thinking`` switch that works across different model templates (e.g. Qwen
uses ``enable_thinking`` while some DeepSeek templates use ``thinking``).

Usage examples:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/chat/completions' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "messages": [
            {"role": "user", "content": "What is the largest animal?"}
        ],
        "enable_thinking": false
      }'

  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty",
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.chat.completions.create(
        model="<MODEL_UID>",
        messages=[
            {"role": "user", "content": "What is the largest animal?"}
        ],
        extra_body={"enable_thinking": False}
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import RESTfulClient

    client = RESTfulClient("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<MODEL_UID>")
    model.chat(
        [{"role": "user", "content": "What is the largest animal?"}],
        enable_thinking=False,
    )

  .. code-tab:: python Xinference Python Client (explicit chat_template_kwargs)

    model.chat(
        [{"role": "user", "content": "What is the largest animal?"}],
        generate_config={"chat_template_kwargs": {"enable_thinking": False}},
    )


DeepSeek-V4-Flash-0731
~~~~~~~~~~~~~~~~~~~~~~

``DeepSeek-V4-Flash-0731`` is a separate built-in model from the
``DeepSeek-V4-Flash`` preview entry. It uses the native FP8 checkpoint and
requires vLLM 0.20.1 or newer.

Launch it with:

.. code-block:: bash

   xinference launch \
       --model-engine vLLM \
       --model-name DeepSeek-V4-Flash-0731 \
       --size-in-billions 304 \
       --model-format fp8 \
       --quantization fp8

The upstream repository supplies DeepSeek-V4-specific encoding code. Enable
trusted repository code before launch:

.. code-block:: bash

   export XINFERENCE_TRUST_REMOTE_CODE=1

The ``chat_template_kwargs`` option selects thinking mode and reasoning level:

.. code-block:: json

   {
     "chat_template_kwargs": {
       "enable_thinking": true,
       "reasoning_effort": "high"
     }
   }

Set ``enable_thinking`` to ``false`` for chat mode. Supported reasoning levels
are provided by the model repository; ``low``, ``high``, and ``max`` are
forwarded without Xinference rewriting them.

The checkpoint includes a DSpark speculative decoding module, but Xinference
does not enable it automatically. Enable it through vLLM model configuration
when supported:

.. code-block:: json

   {
     "speculative_config": {
       "method": "dspark",
       "num_speculative_tokens": 7,
       "draft_sample_method": "greedy"
     }
   }

The following parameters are the upstream example for a single 4xGB300 node
and are not Xinference defaults:

.. code-block:: bash

   --kv-cache-dtype fp8 \
   --block-size 256 \
   --data-parallel-size 4 \
   --enable-expert-parallel \
   --moe-backend deep_gemm_mega_moe \
   --attention-config '{"use_fp4_indexer_cache": true}'


Generate Models
================

Generate API
-------------

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
        messages=[
            {"role": "user", "content": "What is the largest animal?"}
        ],
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

* `LangChain LLMs: Xinference <https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference>`__

* `LlamaIndex LLM integrations: Xinference  <https://docs.llamaindex.ai/en/stable/examples/llm/xinference_local_deployment.html>`__
