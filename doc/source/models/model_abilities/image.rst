.. _image:

======
Images
======

Learn how to generate images with Xinference.


Introduction
==================


The Images API provides two methods for interacting with images:


* The Text-to-image endpoint create images from scratch based on a text prompt.
* The Image-to-image endpoint allows you to generate a variation of a given image.


.. list-table:: 
   :widths: 25  50
   :header-rows: 1

   * - API ENDPOINT
     - OpenAI-compatible ENDPOINT

   * - Text-to-Image API
     - /v1/images/generations

   * - Image-to-image API
     - /v1/images/variations

Supported models
-------------------

The Text-to-image API is supported with the following models in Xinference:

* sd-turbo
* sdxl-turbo
* stable-diffusion-v1.5
* stable-diffusion-xl-base-1.0
* sd3-medium
* FLUX.1-schnell
* FLUX.1-dev


Quickstart
===================

Text-to-image
--------------------

The Text-to-image API mimics OpenAI's `create images API <https://platform.openai.com/docs/api-reference/images/create>`_.
We can try Text-to-image API out either via cURL, OpenAI Client, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/images/generations' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "model": "<MODEL_UID>",
        "prompt": "an apple",
      }'


  .. code-tab:: python OpenAI Python Client

    import openai

    client = openai.Client(
        api_key="cannot be empty", 
        base_url="http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1"
    )
    client.images.generate(
        model=<MODEL_UID>, 
        prompt="an apple"
    )

  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    input_text = "an apple"
    model.text_to_image(input_text)


  .. code-tab:: json output

    {
      "created": 1697536913,
      "data": [
        {
          "url": "/home/admin/.xinference/image/605d2f545ac74142b8031455af31ee33.jpg",
          "b64_json": null
        }
      ]
    }


Tips for Large Image Models including SD3-Medium, FLUX.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Useful extra parameters can be passed to launch including:

* ``--cpu_offload True``: specifying ``True`` will offload the components of the model to CPU during
  inference in order to save memory, while seeing a slight increase in inference latency.
  Model offloading will only move a model component onto the GPU when it needs to be executed,
  while keeping the remaining components on the CPU.
* ``--quantize_text_encoder <text encoder layer>``: We leveraged the ``bitsandbytes`` library
  to load and quantize the T5-XXL text encoder to 8-bit precision.
  This allows you to keep using all text encoders while only slightly impacting performance.
* ``--text_encoder_3 None``, for sd3-medium, removing the memory-intensive 4.7B parameter
  T5-XXL text encoder during inference can significantly decrease the memory requirements
  with only a slight loss in performance.

If you are trying to run large image models liek sd3-medium or FLUX.1 series on GPU card
that has less memory than 24GB, you may encounter OOM when launching or inference.
Try below solutions.

For FLUX.1 series, try to apply quantization.

.. code:: bash

    xinference launch --model-name FLUX.1-dev --model-type image --quantize_text_encoder text_encoder_2

For sd3-medium, apply quantization to ``text_encoder_3``.

.. code:: bash

    xinference launch --model-name sd3-medium --model-type image --quantize_text_encoder text_encoder_3


Or removing memory-intensive T5-XXL text encoder for sd3-medium.

.. code:: bash

    xinference launch --model-name sd3-medium --model-type image --text_encoder_3 None

Image-to-image
--------------------

You can find more examples of Images API in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Stable Diffusion ControlNet
      :link: https://github.com/xorbitsai/inference/blob/main/examples/StableDiffusionControlNet.ipynb
      
      Learn from a Stable Diffusion ControlNet example

