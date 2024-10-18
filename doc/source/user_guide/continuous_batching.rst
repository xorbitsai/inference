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

Image Model
-----------
Currently, for image models, only the ``text_to_image`` interface is supported for ``FLUX.1`` series models.

Enabling this feature requires setting the environment variable ``XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE``, which indicates the ``size`` of the generated images.

For example, starting xinference like this:

.. code-block::

    XINFERENCE_TEXT_TO_IMAGE_BATCHING_SIZE=1024*1024 xinference-local --log-level debug


Then just use the ``text_to_image`` interface as before, and nothing else needs to be changed.

Abort your request
==================
In this mode, you can abort requests that are in the process of inference.

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
Image models also support this feature.

Note
====

* Currently, for ``LLM`` models, this feature only supports the ``generate``, ``chat``, ``tool call`` and ``vision`` tasks.

* Currently, for ``image`` models, this feature only supports the ``text_to_image`` tasks. Only ``FLUX.1`` series models are supported.

* For ``vision`` tasks, currently only ``qwen-vl-chat``, ``cogvlm2``, ``glm-4v`` and ``MiniCPM-V-2.6`` (only for image tasks) models are supported. More models will be supported in the future. Please let us know your requirements.

* If using GPU inference, this method will consume more GPU memory. Please be cautious when increasing the number of concurrent requests to the same model.
  The ``launch_model`` interface provides the ``max_num_seqs`` parameter to adjust the concurrency level, with a default value of ``16``.
