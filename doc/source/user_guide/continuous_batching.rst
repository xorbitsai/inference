.. _user_guide_continuous_batching:

==================================
Continuous Batching (experimental)
==================================

Continuous batching, as a means to improve throughput during model serving, has already been implemented in inference engines like ``VLLM``.
Xinference aims to provide this optimization capability when using the transformers engine as well.

Usage
=====
Currently, this feature can be enabled under the following conditions:

* First, set the environment variable ``XINFERENCE_TRANSFORMERS_ENABLE_BATCHING`` to ``1`` when starting xinference. For example:

.. code-block::

    XINFERENCE_TRANSFORMERS_ENABLE_BATCHING=1 xinference-local --log-level debug


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


Once this feature is enabled, all ``chat`` requests will be managed by continuous batching,
and the average throughput of requests made to a single model will increase.
The usage of the ``chat`` interface remains exactly the same as before, with no differences.

Note
====

* Currently, this feature only supports the ``chat`` interface for ``LLM`` models.

* If using GPU inference, this method will consume more GPU memory. Please be cautious when increasing the number of concurrent requests to the same model.
  The ``launch_model`` interface provides the ``max_num_seqs`` parameter to adjust the concurrency level, with a default value of ``16``.

* This feature is still in the experimental stage, and we welcome your active feedback on any issues.
