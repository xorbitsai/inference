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
* sd3.5-medium
* sd3.5-large
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
* ``--transformer_nf4 True``: use nf4 for transformer quantization.
* ``--quantize``: Only work for MLX on Mac, Flux.1-dev and Flux.1-schnell will switch to
  MLX engine on Mac, and ``quantize`` can be used to quantize the model.


.. note::

    From v0.16.1, Xinference by default enabled quantization for
    large image models like Flux.1 and SD3.5 series.
    Below list default options.

    ====  ====  ====  ===
    Model quantize_text_encoder quantize transformer_nf4
    ====  ====  ====  ===
    FLUX.1-dev text_encoder_2 True False
    FLUX.1-schnell text_encoder_2 True False
    sd3-medium text_encoder_3 N/A False
    sd3.5-medium text_encoder_3 N/A False
    sd3.5-large text_encoder_3 N/A True
    ====  ====  ====  ===

Image-to-image
--------------------

You can find more examples of Images API in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Stable Diffusion ControlNet
      :link: https://github.com/xorbitsai/inference/blob/main/examples/StableDiffusionControlNet.ipynb
      
      Learn from a Stable Diffusion ControlNet example

OCR
--------------------

The OCR API accepts image bytes and returns the OCR text.

We can try OCR API out either via cURL, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/images/ocr' \
      -F model=<MODEL_UID> \
      -F image=@xxx.jpg


  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    with open("xxx.jpg", "rb") as f:
        model.ocr(f.read())


  .. code-tab:: text output

    <OCR result string>
