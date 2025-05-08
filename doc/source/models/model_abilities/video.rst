.. _video:

====================
Video (Experimental)
====================

Learn how to generate videos with Xinference.


Introduction
==================


The Video API provides the ability to interact with videos:


* The text-to-video endpoint create videos from scratch based on a text prompt.
* The image-to-video endpoint create videos from scratch based on an input image.


.. list-table::
   :widths: 25  50
   :header-rows: 1

   * - API
     - Endpoint

   * - Text-to-Video API
     - /v1/video/generations

   * - Image-to-Video API
     - /v1/video/generations/image

Supported models
-------------------

The text-to-video API is supported with the following models in Xinference:

* :ref:`CogVideoX-2b <models_builtin_cogvideox-2b>`
* :ref:`CogVideoX-5b <models_builtin_cogvideox-5b>`
* :ref:`HunyuanVideo <models_builtin_hunyuanvideo>`
* :ref:`Wan2.1-1.3B <models_builtin_wan2.1-1.3b>`
* :ref:`Wan2.1-14B <models_builtin_wan2.1-14b>`

The image-to-video API is supported with the following models in Xinference:

* :ref:`Wan2.1-i2v-14B-480p <models_builtin_wan2.1-i2v-14b-480p>`
* :ref:`Wan2.1-i2v-14B-720p <models_builtin_wan2.1-i2v-14b-720p>`

Quickstart
===================

Text-to-video
--------------------

You can try text-to-video API out either via cURL, or Xinference's python client:

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

Image-to-video
--------------------

You can try image-to-video API out either via cURL, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/video/generations/image' \
      -F model=<MODEL_UID> \
      -F image=@xxx.jpg \
      -F prompt=<prompt>


  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    with open("xxx.jpg", "rb") as f:
        prompt = ""
        model.image_to_video(image=f.read(), prompt=prompt)


Memory optimization
===================

Video generation will occupy huge GPU memory, for instance,
running CogVideoX may require up to around 35 GB GPU memory.

Xinference supports several options to optimize video model memory (VRAM) usage.

* CPU offloading or block level group offloading.
* Layerwise casting.

.. note::

  CPU offloading and Block Level Group Offloading cannot be enabled at the same time,
  but layerwise casting can be used in combination with either of them.

CPU offloading
--------------------

CPU offloading keeps the model weights on the CPU and only loads them to the GPU
when a forward pass needs to be executed. It is suitable for scenarios with extremely limited GPU memory,
but it has a significant impact on performance.

When running on GPU whose memory is less than 24 GB,
we recommend to add ``--cpu_offload True`` when launching model.
For Web UI, add an extra option, ``cpu_offload`` with value set to ``True``.

.. code-block:: bash

    xinference launch --model-name Wan2.1-i2v-14B-480p --model-type video --cpu_offload True

Block Level Group Offloading
-------------------------------

Block Level Group Offloading groups multiple internal layers of the model
(such as ``torch.nn.ModuleList`` or ``torch.nn.Sequential``) and loads these groups from the CPU to the GPU
as needed during inference. Compared to CPU offloading, it uses more memory but has less impact on performance.

For the command line, add the ``--group_offload True`` option; for the Web UI,
add an additional option ``group_offload`` with the value set to ``True``.

We can speed up group offloading inference, by enabling the use of CUDA streams. However,
using CUDA streams requires moving the model parameters into pinned memory.
This allocation is handled by Pytorch under the hood, and can result in a significant spike in CPU RAM usage.
Please consider this option if your CPU RAM is atleast 2X the size of the model you are group offloading.
Enable CUDA streams via adding ``--use_stream True`` for command line; for the Web UI,
add an additional option ``use_stream`` with the value set to ``True``.

.. code-block:: bash

    xinference launch --model-name Wan2.1-i2v-14B-480p --model-type video --group_offload True --use_stream True

Applying Layerwise Casting to the Transformer
------------------------------------------------

Layerwise casting will downcast each layer’s weights to ``torch.float8_e4m3fn``,
temporarily upcast to ``torch.bfloat16`` during the forward pass of the layer,
then revert to ``torch.float8_e4m3fn`` afterward. This approach reduces memory requirements
by approximately 50% while introducing a minor quality reduction in the generated video due to the precision trade-off.
Enable layerwise casting via adding ``--layerwise_cast True`` for command line; for the Web UI,
add an additional option ``layerwise_cast`` with the value set to ``True``.

This example will require 20GB of VRAM.

.. code-block:: bash

    xinference launch --model-name Wan2.1-i2v-14B-480p --model-type video --layerwise_cast True --cpu_offload True

