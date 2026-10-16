.. _sd_webui:

SD WebUI API compatibility
==========================

Xinference exposes Stable Diffusion WebUI-compatible routes alongside its
OpenAI-compatible image API. The extended workflow supports Diffusers Stable
Diffusion and SDXL models. Other image engines retain the basic generation
interface; they do not implement ControlNet, ADetailer, or Hires Fix.

Installation
------------

Install the optional dependencies on every worker that runs this workflow::

    pip install 'xinference[sdapi]'

Preprocessor-specific dependencies, such as MediaPipe, FaceXLib, and
InsightFace, are loaded only when their corresponding preprocessor is selected.
RealESRGAN and ADetailer download their weights on first use. Source licenses
and model-weight licenses are separate; vendored preprocessor notices and
licenses are included under ``xinference/thirdparty/controlnet``.

Generation
----------

Use the UID of an already launched image model in ``model`` or
``override_settings.sd_model_checkpoint``. For example::

    curl http://localhost:9997/sdapi/v1/txt2img \
      -H 'Content-Type: application/json' \
      -d '{"model":"sd-model","prompt":"a (small cat:1.2)",
           "seed":42,"steps":20,"width":512,"height":512,
           "batch_size":2,"n_iter":1,"request_id":"example-job"}'

The response contains base64 ``images``, the effective ``parameters``, and an
``info`` object with seeds and generation metadata. ``batch_size`` is the number
of images in each iteration; ``n_iter`` is the number of iterations. The seed
increments for each image, except when ``subseed_strength`` is nonzero. In that
case the primary seed stays fixed and the subseed increments.

Weighted prompts support parentheses, explicit weights, square brackets,
escaped delimiters, and long prompts split into CLIP token chunks. SDXL uses both
text encoders and supplies the pooled conditioning required by its pipeline.

For Hires Fix, set ``enable_hr`` to true, ``hr_scale`` (default 2),
``hr_upscaler`` (for example ``Latent``, ``Lanczos`` or ``R-ESRGAN 4x+``), and
optionally ``hr_second_pass_steps``. ``denoising_strength`` must be greater than
zero and at most one. The image is generated, enlarged, and refined in a second
image-to-image pass.

``POST /sdapi/v1/img2img`` accepts ``init_images`` containing raw base64,
image data URLs, or public HTTP(S) image URLs. It supports ``mask``,
``mask_blur``, ``inpainting_mask_invert``, ``inpaint_full_res``, and
``inpaint_full_res_padding``. ``resize_mode`` is 0 (resize), 1 (crop to fill), or
2 (fit and fill borders). URL downloads reject private addresses and validate
every redirect; image responses are limited to 64 MiB.

ControlNet and ADetailer
-----------------------

Pass WebUI scripts through ``alwayson_scripts``::

    {
      "ControlNet": {
        "args": [{"enabled": true, "module": "canny", "model": "controlnet-canny",
                  "image": "<base64 image>", "weight": 1.0,
                  "pixel_perfect": true, "resize_mode": "Crop and Resize"}]
      },
      "ADetailer": {
        "args": [true, {"ad_model": "face_yolov8n.pt", "ad_prompt": "[PROMPT]"}]
      }
    }

Select a ControlNet model listed by the launched model's family. Multiple units
are supported, with independent conditioning weights and guidance start/end.
The reference-only workflow is available for SD 1.5. Preprocessors include Canny,
depth, OpenPose, MLSD, and line art; individual detector dependencies and weights
must be available on the worker. IP-Adapter tensor-only workflows are not part
of this SDAPI port.

ADetailer runs after generation or Hires Fix and supports detection masks,
local inpainting, ``[PROMPT]``, ``[SEP]``, and ``[SKIP]`` prompt directives.

LoRA
----

Register a custom image model with ``model_family: "lora"`` and a local
``model_uri`` or a model hub repository containing one safetensors file. Reference
it as ``<lora:model-name:0.8>`` in the prompt. ``metadata.ss_output_name`` can supply
an alternative alias. ``override_settings.strict`` rejects unresolved LoRAs.
Adapters are removed after the request, including when inference fails.

An HTTPS CivitAI download URL can also be supplied as the registered LoRA's
``model_uri``. If authorization is needed, set ``CIVITAI_API_TOKEN`` in the worker
environment. Tokens are not forwarded to redirected CDN hosts. Downloads use a
temporary file followed by atomic replacement.

Querying and cancelling
-----------------------

The following routes use the same authentication requirements as image generation:

* ``GET /sdapi/v1/loras`` and ``GET /sdapi/v1/upscalers`` list available adapters
  and upscalers.
* ``GET /controlnet/model_list``, ``GET /controlnet/module_list``, and
  ``GET /controlnet/control_types`` query the running models and preprocessors.
* ``POST /controlnet/detect`` accepts ``controlnet_module``,
  ``controlnet_input_images`` (or the alias ``controlnet_images``),
  ``controlnet_processor_res``, and threshold parameters.
* ``GET /sdapi/v1/progress?request_id=example-job`` returns request progress.
* ``POST /sdapi/v1/interrupt`` accepts
  ``{"model":"sd-model","request_id":"example-job"}`` and cancels that request.
  This route requires an explicit model and request ID; it is not a global stop.

ComfyUI is outside this API compatibility layer.
