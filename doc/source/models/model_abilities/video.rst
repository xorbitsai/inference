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
* The firstlastframe-to-video endpoint creates videos based on the transition between a first and a last frame.


.. list-table::
   :widths: 25  50
   :header-rows: 1

   * - API
     - Endpoint

   * - Text-to-Video API
     - /v1/video/generations

   * - Image-to-Video API
     - /v1/video/generations/image

   * - FirstLastFrame-to-Video API
     - /v1/video/generations/flf

Supported models
-------------------

The text-to-video API is supported with the following models in Xinference:

* :ref:`MiniMax-H3 <models_builtin_minimax-h3>`
* :ref:`CogVideoX-2b <models_builtin_cogvideox-2b>`
* :ref:`CogVideoX-5b <models_builtin_cogvideox-5b>`
* :ref:`HunyuanVideo <models_builtin_hunyuanvideo>`
* :ref:`Wan2.1-1.3B <models_builtin_wan2.1-1.3b>`
* :ref:`Wan2.1-14B <models_builtin_wan2.1-14b>`

The image-to-video API is supported with the following models in Xinference:

* :ref:`MiniMax-H3 <models_builtin_minimax-h3>`
* :ref:`Wan2.1-i2v-14B-480p <models_builtin_wan2.1-i2v-14b-480p>`
* :ref:`Wan2.1-i2v-14B-720p <models_builtin_wan2.1-i2v-14b-720p>`

The firstlastframe-to-video API is supported with the following models in Xinference:

* :ref:`MiniMax-H3 <models_builtin_minimax-h3>`
* :ref:`Wan2.1-flf2v-14B-720p <models_builtin_wan2.1-flf2v-14b-720p>`

Video engines
-------------

Video runtimes are selected with ``--model-engine``. All currently built-in
video models use the ``diffusers`` engine, which is also the default. For
example, the explicit form is:

.. code-block:: bash

    xinference launch --model-name MiniMax-H3 --model-type video --model-engine diffusers

The Web UI obtains the engines supported by each model from the engine query
API and presents them in the launch dialog. Additional video runtimes can be
registered independently without changing the Video API.

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

FirstLastFrame-to-video
--------------------------

You can try firstlastframe-to-video API out either via cURL, or Xinference's python client:

.. tabs::

  .. code-tab:: bash cURL

    curl -X 'POST' \
      'http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/video/generations/flf' \
      -F model=<MODEL_UID> \
      -F first_frame=@xxx.jpg \
      -F last_frame=@xxx2.jpg \
      -F prompt=<prompt>


  .. code-tab:: python Xinference Python Client

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")

    model = client.get_model("<MODEL_UID>")
    with open("xxx.jpg", "rb") as f1, open("xxx2.jpg", "rb") as f2:
        prompt = ""
        model.flf_to_video(first_frame=f1.read(), last_frame=f2.read(), prompt=prompt)


Lightning LoRA acceleration
===========================

Lightning LoRA checkpoints distill a video model into fewer denoising steps.
Select a supported version with ``--lightning_version`` when launching the model;
Xinference downloads the LoRA, applies its training alpha and scheduler shifts,
and uses the version's recommended inference-step count when the request does not
override ``num_inference_steps``.
Lightning reduces denoising time, but does not reduce model size or peak memory;
MiniMax-H3's default INT4 quantization and group offload remain enabled.

.. list-table::
   :widths: 25 30 15 15 15
   :header-rows: 1

   * - Model
     - Lightning version
     - Evaluations
     - Video shift
     - Recommended canvas
   * - MiniMax-H3
     - ``4step_v0.1``
     - 4
     - 12
     - 544p mixed aspect ratios
   * - MiniMax-H3
     - ``8step_v1.0_bf16``
     - 8
     - 12
     - 544p mixed aspect ratios
   * - MiniMax-H3
     - ``4step_v1.0_768p_bf16``
     - 4
     - 6
     - 1344x768

In the Web UI, open the MiniMax-H3 launch dialog, expand **Advanced
Configuration**, and select a value under **Lightning Versions**. Leave
**Lightning Model Path** empty to download the selected checkpoint
automatically. After the model starts, set **Inference Steps** on the video
generation page to the evaluation count in the table. The generation page
currently starts with 25 steps, which overrides the Lightning default if left
unchanged.

For example, launch the 768p four-step version from the command line::

    xinference launch --model-name MiniMax-H3 --model-type video \
        --lightning_version 4step_v1.0_768p_bf16

Then generate with four inference steps. MiniMax-H3 outputs at a fixed 24 FPS;
124 frames produce a video of about five seconds::

    from xinference.client import Client

    client = Client("http://<XINFERENCE_HOST>:<XINFERENCE_PORT>")
    model = client.get_model("<MODEL_UID>")
    model.text_to_video(
        prompt="A running cat",
        width=1344,
        height=768,
        num_frames=124,
        fps=24,
        num_inference_steps=4,
    )

Xinference downloads the Lightning checkpoint from the same hub selected for
the base model. Both Hugging Face and ModelScope are supported. To use an already
downloaded checkpoint, pass both its path and version::

    xinference launch --model-name MiniMax-H3 --model-type video \
        --lightning_version 4step_v0.1 \
        --lightning_model_path /path/to/minimax_h3_fl2v_turbo_4step_v0.1.safetensors

``num_inference_steps`` represents actual transformer evaluations and remains a
per-request override. Match it to the selected Lightning version: use 4 for a
``4step`` checkpoint and 8 for an ``8step`` checkpoint. MiniMax-H3's scheduler
internally adds the terminal sigma grid point required to run that number of
evaluations.

.. note::

   The evaluation-count semantics apply to MiniMax-H3 with or without Lightning.
   A request for N evaluations now passes N + 1 scheduler grid points so the
   terminal sigma does not consume one of the requested evaluations. Therefore,
   non-Lightning output may differ from earlier Xinference versions for the same
   ``num_inference_steps`` value.


Memory optimization
===================

Video generation will occupy huge GPU memory, for instance,
running CogVideoX may require up to around 35 GB GPU memory.

Xinference supports several options to optimize video model memory (VRAM) usage.

* CPU offloading or block level group offloading.
* Layerwise casting.
* Weight quantization.

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

Weight quantization
-------------------

Some video models support weight-only quantization through the
``--quantization`` option. Quantization lowers both GPU and host memory usage,
with a possible quality and performance trade-off.

MiniMax-H3 supports the following values for ``quantization``:

* ``int4``: the default. Most large linear weights use TorchAO INT4, while a few
  BF16 blocks remain on the CPU during loading. Together with block-level group
  offloading, this allows the model to load on a 24GB consumer GPU without
  additional launch options. CUDA streams are disabled on this path to avoid an
  extra pinned host-memory copy.
* ``int8``: use TorchAO INT8 weight-only quantization for higher weight precision.
  This requires at least 75GB of available host RAM.
* ``none`` or ``bf16``: disable weight quantization. With the default
  ``torch_dtype``, weights are loaded in BF16 and require substantially more GPU
  and host memory.
* ``torchao``: a compatibility alias for ``int8``. Use ``int8`` in new launch
  configurations.

For example, select INT8 with::

    xinference launch --model-name MiniMax-H3 --model-type video \
        --quantization int8

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
