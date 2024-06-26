.. _image:

=====================
Images (Experimental)
=====================

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


.. note::

  If you are running ``sd3-medium`` on a GPU less than 24GB and encountering out of memory,
  consider to add an extra param for launching according to `this article <https://huggingface.co/docs/diffusers/v0.29.1/en/api/pipelines/stable_diffusion/stable_diffusion_3#dropping-the-t5-text-encoder-during-inference>`_.

  xinference launch --model-name sd3-medium --model-type image --text_encoder_3 None

Image-to-image
--------------------

You can find more examples of Images API in the tutorial notebook:

.. grid:: 1

   .. grid-item-card:: Stable Diffusion ControlNet
      :link: https://github.com/xorbitsai/inference/blob/main/examples/StableDiffusionControlNet.ipynb
      
      Learn from a Stable Diffusion ControlNet example

