.. _user_guide_continuous_batching:

===================
Continuous Batching
===================

Continuous batching, as a means to improve throughput during model serving, has already been implemented in inference engines like ``VLLM``.
Xinference aims to provide this optimization capability when using the transformers engine as well.

Usage
=====

LLM
---
Currently, this feature can be enabled under the following conditions:

* First, set the environment variable ``XINFERENCE_TRANSFORMERS_ENABLE_BATCHING`` to ``1`` when starting xinference. For example:

.. code-block::

    XINFERENCE_TRANSFORMERS_ENABLE_BATCHING=1 xinference-local --log-level debug


.. note::
   Since ``v0.16.0``, this feature is turned on by default and
   is no longer required to set the ``XINFERENCE_TRANSFORMERS_ENABLE_BATCHING`` environment variable.
   This environment variable has been removed.


* Then, ensure that the ``transformers`` engine is selected when launching the model. For example:

.. tabs::

  .. code-tab:: bash shell

    xinference launch -e <endpoint> --model-engine transformers -n qwen1.5-chat -s 4 -f pytorch -q none

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://127.0.0.1:9997/v1/models' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "model_engine": "transformers",
      "model_name": "qwen1.5-chat",
      "model_format": "pytorch",
      "size_in_billions": 4,
      "quantization": "none"
    }'

  .. code-tab:: python

    from xinference.client import Client
    client = Client("http://127.0.0.1:9997")
    model_uid = client.launch_model(
      model_engine="transformers",
      model_name="qwen1.5-chat",
      model_format="pytorch",
      model_size_in_billions=4,
      quantization="none"
    )
    print('Model uid: ' + model_uid)


Once this feature is enabled, all requests for LLMs will be managed by continuous batching,
and the average throughput of requests made to a single model will increase.
The usage of the LLM interface remains exactly the same as before, with no differences.

Embedding
---------

With vLLM >= 0.19.0, text embedding models use the native asynchronous pooling scheduler. Configure ``max_num_seqs`` and ``max_num_batched_tokens`` at model launch; Xinference ``batch_size`` and ``batch_interval`` do not control this path. Older vLLM versions, VACC, WeMM and Qwen3-VL embeddings retain the existing Xinference batching path.

Rerank
------

With vLLM >= 0.19.0, text cross-encoder rerank models also use native asynchronous pooling. The same ``max_num_seqs`` and ``max_num_batched_tokens`` settings apply instead of Xinference batch settings. Older vLLM versions, VACC, Qwen3-VL rerankers and other scoring types retain the existing path. Ranking, ``top_n``, returned documents and token counts keep the same API behavior.

Image Model
-----------
For image-model batching, select the ``vLLM`` or ``SGLang`` engine with ``--model-engine``. SGLang supports dynamic request batching via ``batching_max_size`` and ``batching_delay_ms``; new requests wait while a batch is running. vLLM-Omni supports request batching via ``max_num_seqs`` and, on supported models, experimental step-level continuous batching with ``step_execution=true`` and ``max_num_seqs>1``. Diffusers supports ordinary image generation and multiple images per request, but does not provide Xinference-managed continuous batching. See :ref:`image` for supported models, required versions, and launch instructions.

Abort your request
==================
For LLMs using Xinference's continuous batching, you can abort requests that are in the process of inference.

#. First, add ``request_id`` option in ``generate_config``. For example:

.. code-block:: bash

    from xinference.client import Client
    client = Client("http://127.0.0.1:9997")
    model = client.get_model("<model_uid>")
    model.chat([{"role": "user", "content": "<prompt>"}], generate_config={"request_id": "<your_unique_request_id>"})

#. Then, abort the request using the ``request_id`` you have set. For example:

.. code-block:: bash

    from xinference.client import Client
    client = Client("http://127.0.0.1:9997")
    client.abort_request("<model_uid>", "<your_unique_request_id>")

Note that if your request has already finished, aborting the request will be a no-op.

Note
====

* Currently, for ``LLM`` models, this feature only supports the ``generate``, ``chat``, ``tool call`` and ``vision`` tasks.

* For ``vision`` tasks, currently only ``qwen2-vl-instruct``, ``qwen2.5-vl-instruct``, ``QvQ-72B-Preview``, ``glm-4v`` and ``MiniCPM-V-2.6`` (only for image tasks) models are supported. More models will be supported in the future. Please let us know your requirements.

* If using GPU inference, this method will consume more GPU memory. Please be cautious when increasing the number of concurrent requests to the same model.
  The ``launch_model`` interface provides the ``max_num_seqs`` parameter to adjust the concurrency level, with a default value of ``16``.
