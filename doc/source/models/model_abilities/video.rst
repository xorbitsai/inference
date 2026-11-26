.. _video:

=====
Video
=====

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
* :ref:`LTX-2.5 <models_builtin_ltx-2.5>`
* :ref:`LTX-2.5-Dev <models_builtin_ltx-2.5-dev>`
* :ref:`Wan2.1-1.3B <models_builtin_wan2.1-1.3b>`
* :ref:`Wan2.1-14B <models_builtin_wan2.1-14b>`
* :ref:`Wan2.2-A14B <models_builtin_wan2.2-a14b>`
* :ref:`Wan2.2-ti2v-5B <models_builtin_wan2.2-ti2v-5b>`
* :ref:`LTX-2-distilled <models_builtin_ltx-2-distilled>`
* :ref:`LTX-2-dev <models_builtin_ltx-2-dev>`
* :ref:`LTX-2.3-distilled <models_builtin_ltx-2.3-distilled>`
* :ref:`LTX-2.3-dev <models_builtin_ltx-2.3-dev>`

The image-to-video API is supported with the following models in Xinference:

* :ref:`MiniMax-H3 <models_builtin_minimax-h3>`
* :ref:`LTX-2.5 <models_builtin_ltx-2.5>`
* :ref:`LTX-2.5-Dev <models_builtin_ltx-2.5-dev>`
* :ref:`Wan2.1-i2v-14B-480p <models_builtin_wan2.1-i2v-14b-480p>`
* :ref:`Wan2.1-i2v-14B-720p <models_builtin_wan2.1-i2v-14b-720p>`
* :ref:`Wan2.2-i2v-A14B <models_builtin_wan2.2-i2v-a14b>`
* :ref:`Wan2.2-ti2v-5B <models_builtin_wan2.2-ti2v-5b>`
* :ref:`Wan2.2-Animate-2-14B <models_builtin_wan2.2-animate-2-14b>`
* :ref:`Wan2.2-Animate-2-14B-Distilled <models_builtin_wan2.2-animate-2-14b-distilled>`
* :ref:`LTX-2-distilled <models_builtin_ltx-2-distilled>`
* :ref:`LTX-2-dev <models_builtin_ltx-2-dev>`
* :ref:`LTX-2.3-distilled <models_builtin_ltx-2.3-distilled>`
* :ref:`LTX-2.3-dev <models_builtin_ltx-2.3-dev>`

The firstlastframe-to-video API is supported with the following models in Xinference:

* :ref:`MiniMax-H3 <models_builtin_minimax-h3>`
* :ref:`LTX-2.5 <models_builtin_ltx-2.5>`
* :ref:`LTX-2.5-Dev <models_builtin_ltx-2.5-dev>`
* :ref:`Wan2.1-flf2v-14B-720p <models_builtin_wan2.1-flf2v-14b-720p>`
* :ref:`LTX-2-distilled <models_builtin_ltx-2-distilled>`
* :ref:`LTX-2-dev <models_builtin_ltx-2-dev>`
* :ref:`LTX-2.3-distilled <models_builtin_ltx-2.3-distilled>`
* :ref:`LTX-2.3-dev <models_builtin_ltx-2.3-dev>`

Video engines
-------------

Video runtimes are selected with ``--model-engine``. The ``diffusers`` engine
remains the default for models that expose both runtimes. For example, launch
Wan2.2-A14B with either engine explicitly:

.. code-block:: bash

    xinference launch --model-name Wan2.2-A14B --model-type video --model-engine diffusers
    xinference launch --model-name Wan2.2-A14B --model-type video --model-engine MLX

On Linux with CUDA, ``Wan2.1-1.3B``, ``Wan2.1-14B`` and ``Wan2.2-A14B`` also support text-to-video with ``SGLang`` (0.5.20 or newer) and ``vLLM`` (vLLM-Omni and vLLM 0.28). Concurrent requests are submitted to the native engine scheduler. Use ``batching_max_size`` and ``batching_delay_ms`` for SGLang, or ``max_num_seqs`` for vLLM. These engine variants do not expose image-to-video or Lightning/GGUF adapters. Diffusers remains the default.

.. code-block:: bash

    xinference launch --model-name Wan2.1-1.3B --model-type video --model-engine SGLang --batching_max_size 2 --batching_delay_ms 5
    xinference launch --model-name Wan2.1-1.3B --model-type video --model-engine vLLM --max_num_seqs 2

The ``MLX`` engine uses `Blaizzy/mlx-video
<https://github.com/Blaizzy/mlx-video>`_ and is available on Apple Silicon with
Python 3.11 or newer. It supports Wan2.1 T2V, Wan2.2 T2V/I2V/TI2V, and the
LTX-2/LTX-2.3 distilled and dev models listed above. LTX-2 and LTX-2.3 are
MLX-only in Xinference.

Wan2.1's official checkpoints are converted to the native MLX layout on first
load and the converted copy is reused by later launches. This first launch
therefore needs additional time and disk space. Wan2.2 and the MLX LTX models use
pre-converted checkpoints.

The Web UI obtains the engines supported by each model from the engine query
API and presents them in the launch dialog. Additional video runtimes can be
registered independently without changing the Video API.

MiniMax-H3 also supports ``SGLang`` (0.5.22 or newer, below 0.6) and ``vLLM``
(matching vLLM-Omni and vLLM 0.32.x) on Linux with CUDA. Both engines implement
text-to-video, image-to-video and first/last-frame generation through the
FL2VA checkpoint, retaining the synchronized stereo soundtrack at 24 fps.
Ref2VA is not exposed through these Xinference Video API endpoints.

MiniMax-H3 downloads only the files needed by the selected engine. Diffusers
uses the root FL2VA components; SGLang and vLLM share the native ``FL2VA/``
checkpoint. Each layout is about 134 GiB in BF16 before runtime quantization.
Lightning weights are downloaded separately.

The H3 launch dialog offers a shared runtime precision selector, also available
as ``runtime_precision`` in launch requests: ``bf16`` and ``fp8`` on all three
engines, plus ``int8`` and ``int4`` on Diffusers. An explicit choice overrides
engine-specific precision parameters for the DiT and text encoder; VAEs retain
their native precision. FP8 uses each engine's native scaling implementation
and is not an identical quantization scheme across engines. Omitting this
option or selecting ``auto`` preserves advanced parameters and engine defaults
(INT4 on Diffusers, BF16 on SGLang and vLLM). Switching engines retains an
explicit precision choice and rejects unsupported values.

For online FP8 quantization of H3, pass ``quantization_config="fp8"`` to
Diffusers or vLLM. For SGLang, pass
``component_quantizations={"transformer": "fp8", "text_encoder": "fp8"}``.
This selects FP8 weights with dynamically quantized FP8 activations for
supported DiT and text encoder linear layers, while leaving the VAEs unquantized.
FP8 requires compatible GPU hardware and kernels.

For an unquantized H3 Diffusers baseline, set ``quantization_config="bf16"``.
The ``group_offload`` option applies to both quantized and unquantized H3 models.
With streaming offload, ``low_cpu_mem_usage=True`` pins CPU weights on demand.

Both native H3 engines accept the LightX2V ``lightning_version`` values listed
below. Xinference loads the adapter and selects its step count and video/audio
flow shifts automatically. Conflicting per-request step counts or flow shifts
are rejected. For vLLM, keep the published Diffusers checkpoint filename;
ComfyUI exports are incompatible.

Native H3 requests require a duration of 4–15 seconds at 24 fps. For a first
test, use ``width=960``, ``height=544`` and ``num_frames=124`` (about 5 seconds).
With Lightning active, omit ``num_inference_steps`` to use the selected schedule,
or set it to 4 for either 4-step adapter and 8 for the 8-step adapter.

For SGLang, install the system ``ffmpeg`` package before launching H3 and ensure
both ``ffmpeg`` and ``ffprobe`` are on ``PATH``. The Python packages ``av`` and
``imageio-ffmpeg`` do not satisfy this requirement.

With vLLM, the Web UI reports denoising progress for native H3. The request
completes after audio/video decoding and encoding.

Configure GPU parallelism and memory options for your hardware. For example,
on four suitable CUDA GPUs, launch H3 with Lightning using::

    xinference launch --model-name MiniMax-H3 --model-type video --model-engine SGLang \
        --n-gpu 4 --num_gpus 4 --tp_size 2 --ulysses_degree 2 --lightning_version 4step_v0.1
    xinference launch --model-name MiniMax-H3 --model-type video --model-engine vLLM \
        --n-gpu 4 --num_gpus 4 --ulysses_degree 4 --vae_patch_parallel_size 4 --vae_parallel_mode tile \
        --vae_use_tiling True --lightning_version 4step_v1.0_768p_bf16

.. _ltx_2_5_usage:

LTX-2.5
-------

Xinference provides two official-only LTX-2.5 configurations:

* ``LTX-2.5`` loads the Distilled BF16 transformer and is the recommended
  default. It uses the exact eight-value sigma schedule published by
  Lightricks and Hugging Face.
* ``LTX-2.5-Dev`` loads the Full/Dev BF16 transformer from
  ``transformer_full`` and uses the official guided scheduler configuration.

Both configurations use the official Diffusers-format repository from
`Hugging Face <https://huggingface.co/Lightricks/LTX-2.5-Diffusers>`_ or
`ModelScope <https://modelscope.cn/models/Lightricks/LTX-2.5-Diffusers>`_.
The separate `ModelScope native checkpoint repository
<https://modelscope.cn/models/Lightricks/LTX-2.5>`_ uses the ``ltx-pipelines``
split format and cannot be loaded directly by ``LTX2ConditionPipeline``;
Xinference therefore uses Lightricks' official Diffusers-format mirror.
Xinference does not register third-party, modified, repacked, GGUF, INT8, FP8,
or NVFP4 checkpoints for these models and only loads the official BF16 files.
The LTX adapter rejects custom ``model_path`` overrides and alternate-weight
options it receives.

Access and dependencies
-----------------------

When downloading from Hugging Face, the official repository is gated. Before
launching the model from that hub:

#. Open the official model page and accept its access terms.
#. Authenticate the Xinference process with a Hugging Face token that can read
   the gated repository.
#. Keep model virtual environments enabled. Xinference pins the official
   Diffusers LTX-2.5 implementation to a tested PyPI release.

The runtime requires Python 3.10 or newer, PyTorch 2.7 or newer, a
BF16-capable accelerator, PyAV, and an FFmpeg build with H.264/AAC encoding.
The official model pack is large, and CPU offload is enabled by default.
Downloading and running it also requires substantial disk space and system
RAM.

Model CPU offload still moves the complete roughly 38 GB transformer onto one
accelerator while it runs. On an accelerator that cannot hold that component,
launch with ``sequential_cpu_offload=true``; it uses substantially less device
memory but is slower. This adapter does not implement LTX's native distributed
transformer sharding.

.. warning::

   The LTX-2.5 weights use the `LTX-2.x Community License
   <https://github.com/Lightricks/LTX-2/blob/main/LICENSE.md>`_, not Xinference's
   Apache license. Review that license before use. In particular, organizations
   at or above the license's revenue threshold may require a separate commercial
   agreement. This documentation is not legal advice.

Generation behavior
-------------------

The default request generates 121 frames at 960x544 and 24 FPS. The MP4 output
contains LTX-2.5's synchronized generated audio; the audio sampling rate is read
from the official vocoder configuration instead of being hard-coded.

LTX-2.5 generation has the following constraints:

* ``width`` and ``height`` must be positive multiples of 32.
* ``fps``/``frame_rate`` must be a positive integer so the generated audio and
  encoded video retain the same timing.
* An explicit ``num_frames`` must satisfy ``num_frames % 8 == 1``.
* The default is 121 frames. Explicitly pass ``num_frames=None`` from Python,
  or ``"num_frames": null`` inside the REST ``kwargs`` JSON, to enable the
  official duration head for a single prompt. Use ``min_seconds`` and
  ``max_seconds`` to bound the prediction. Prompt batches must specify an
  integer ``num_frames``.
* Distilled generation always uses the official fixed sigma schedule and fixed
  guidance values. It rejects ``num_inference_steps``, custom timesteps, custom
  sigma schedules, and incompatible guidance values rather than silently
  degrading quality. Because classifier-free guidance is disabled in this
  configuration, ``negative_prompt`` is ignored; it is effective with
  ``LTX-2.5-Dev``.
* Prompt enhancement is not supported. Xinference skips the prompt-enhancer
  weights when downloading and loading the model.
* Image conditions use the official LTX-2.5 compression behavior (CRF 18 by
  default). Pass ``image_crf=0`` to explicitly disable recompression.

Implementation scope
--------------------

This integration targets the official LTX-2.5 support in the tested Hugging
Face Diffusers release and the official Diffusers-format checkpoint. It
covers all three video abilities currently exposed by Xinference: text-to-video,
image-to-video, and first/last-frame interpolation, including synchronized
audio.

It does not claim bit-for-bit or sampler-equivalent output with Lightricks'
separate native ``ltx-pipelines`` v1.2 production recipes. In particular,
Xinference currently uses the official Diffusers convolutional VAE path,
provides the Diffusers latent-upscaler recipe only for the Distilled model, and
does not expose the native diffusion decoder, Dev two-stage recipe, audio/video
editing pipelines, or native multi-GPU execution.

For example, a text-to-video request can use the model defaults::

   curl -X POST http://<XINFERENCE_HOST>:<XINFERENCE_PORT>/v1/video/generations \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "<MODEL_UID>",
       "prompt": "A slow cinematic tracking shot through a rain-soaked market"
     }'

Distilled two-stage generation
------------------------------

``LTX-2.5`` also supports the official latent spatial upscaler. Set
``two_stage=true`` to generate the first stage at half resolution, upscale its
video latents by 2x, and refine them with the official stage-two sigma schedule.
For this mode, ``width`` and ``height`` are the final output size and must be
multiples of 64; ``num_frames`` must be explicit. For example, the following
uses a 960x544 first stage and produces 1920x1088 output::

   model.text_to_video(
       "A detailed aerial shot of waves breaking over black volcanic rock",
       width=1920,
       height=1088,
       num_frames=121,
       two_stage=True,
   )

``two_stage`` is not available for ``LTX-2.5-Dev``. The Dev model is exposed as
the official Full/SFT single-stage configuration; it is intended for users who
need guided generation rather than the fastest distilled path.

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
automatically.

On Diffusers, SGLang, and vLLM, the video generation page shows a **Lightning**
switch once the adapter is loaded. Pass ``use_lightning=False`` in a Python client generation
call, or include ``"use_lightning": false`` in the video REST request's ``kwargs``
object, to use the base model and its sampling defaults for that request.
Omitting the option uses the adapter selected at launch.

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
