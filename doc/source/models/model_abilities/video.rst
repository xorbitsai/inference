.. _video:

====================
Video (Experimental)
====================

Learn how to generate videos with Xinference.


Introduction
==================


The Video API provides the ability to interact with videos:


* The text-to-video endpoint create videos from scratch based on a text prompt.


.. list-table::
   :widths: 25  50
   :header-rows: 1

   * - API ENDPOINT
     - OpenAI-compatible ENDPOINT

   * - Text-to-Video API
     - /v1/video/generations

Supported models
-------------------

The Text-to-video API is supported with the following models in Xinference:

* CogVideoX-2b


Quickstart
===================

Text-to-video
--------------------

You can try Text-to-video API out either via cURL, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/video/generations' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "prompt": "<your prompt>"
      }'


  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    input_text = "an apple"
    model.text_to_video(input_text)


Tips when running on GPU whose memory less than 24GB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Text-to-video will occupy huge GPU memory, for instance,
running CogVideoX may require up to around 35 GB GPU memory.
When running on GPU whose memory is less than 24 GB,
we recommend to add ``--cpu_offload True`` when launching model.


.. code-block:: bash

    xinference launch --model-name CogVideoX-2b --model-type video --cpu_offload True
