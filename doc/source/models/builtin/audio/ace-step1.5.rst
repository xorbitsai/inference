.. _models_builtin_ace-step1.5:

============
ACE-Step1.5
============

- **Model Name:** ACE-Step1.5
- **Model Family:** ace_step_1_5
- **Abilities:** ['text2music']
- **Multilingual:** True
- **Engine:** PyTorch
- **Python:** 3.11 or 3.12
- **License:** `MIT <https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE>`_

Specifications
^^^^^^^^^^^^^^

The built-in model downloads the complete ACE-Step 1.5 checkpoint bundle from
`Hugging Face <https://huggingface.co/ACE-Step/Ace-Step1.5>`_ or
`ModelScope <https://modelscope.cn/models/ACE-Step/Ace-Step1.5>`_. The bundle
contains the default ``acestep-v15-turbo`` DiT, ``vae``,
``Qwen3-Embedding-0.6B``, and ``acestep-5Hz-lm-1.7B``. Xinference uses the
official
`ACE-Step 1.5 Python API <https://github.com/ace-step/ACE-Step-1.5>`_ in a
per-model virtual environment.

This initial integration intentionally supports only those bundled DiT and LM
checkpoints. Standalone ACE-Step DiT, LM, and VAE combinations are not selected
through ``config_path`` or ``lm_model_path`` yet.

ACE-Step supports CUDA, ROCm, Apple Silicon, Intel XPU, and CPU. Accelerator
availability and performance depend on the installed system PyTorch build.

Launch the default DiT-only configuration::

   xinference launch --model-name ACE-Step1.5 --model-type audio --model-engine PyTorch

The default avoids loading the 5Hz language model. To enable LM planning,
metadata completion, and audio-code reasoning, load the bundled 1.7B LM::

   xinference launch --model-name ACE-Step1.5 --model-type audio \
     --model-engine PyTorch --lm_model_path acestep-5Hz-lm-1.7B

The LM uses its PyTorch backend by default. ``offload_to_cpu``,
``offload_dit_to_cpu``, ``quantization``, and ``compile_model`` can be supplied
as launch options for supported hardware. ``lm_backend`` accepts ``pt``,
``vllm``, or ``mlx``; ``vllm`` requires CUDA for native execution, while
``mlx`` targets Apple Silicon. Unsupported hardware falls back according to the
upstream ACE-Step runtime.

See :ref:`ACE-Step1.5 speech usage <ace_step1_5_speech>` for request examples
and parameter limits.
