.. _installation:

============
Installation
============
Xinference can be installed with ``pip`` on Linux, Windows, and macOS. To run models using Xinference, you will need to install the backend corresponding to the type of model you intend to serve.

If you aim to serve all supported models, you can install all the necessary dependencies with a single command::

   pip install "xinference[all]"

Token Router
~~~~~~~~~~~~
The Token Router Runtime and Router Agent can be installed separately from
the model-serving backends::

   pip install "xinference[router]"

When installing Xinference from a source checkout, use::

   uv pip install -e ".[router]"

The ``router`` extra provides the dependencies required by the Token Router
Runtime and Router Agent. The ``all`` extra includes the ``router`` extra.

The Supervisor can use the base installation::

   pip install xinference

Installing the package creates the ``xinference-router`` and
``xinference-router-agent`` commands. The extra controls dependency
installation; it does not start any Supervisor, Worker, Router, or Router
Agent process.

.. versionchanged:: v1.8.1

   Due to irreconcilable package dependency conflicts between vLLM and sglang, we have removed sglang from the all extra. If you want to use sglang, please install it separately via ``pip install 'xinference[sglang]'``.

.. _one_line_install:

One-command installation and startup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The installer prepares uv, Python 3.12, and an isolated Xinference tool environment, then starts the local server in the foreground. Press Ctrl+C to stop it. On Linux and macOS::

   curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh

On Windows, run this in PowerShell::

   irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex

Set ``XINFERENCE_START=0`` to install without starting. Use ``XINFERENCE_VERSION`` to pin a release, ``XINFERENCE_PYTHON`` to select Python, ``XINFERENCE_EXTRAS`` to preinstall backends, and ``XINFERENCE_TOOL_DIR`` to change the uv tool store. ``XINFERENCE_HOME`` controls persistent model data; it is separate from the tool environment.

On Linux and Windows, ``XINFERENCE_BACKEND=auto`` selects PyTorch for the detected GPU driver or CPU. Override it with ``cpu``, ``cu128``, or another backend supported by uv. macOS uses its native PyTorch wheel. Recent PyTorch wheels require Apple Silicon; Intel Macs may resolve older dependencies or fail installation. Model virtual environments install engine dependencies on demand. Engine platform restrictions still apply, including Linux-only vLLM and Apple-Silicon-only MLX.

To run without a persistent tool installation, use ``uvx``. This requires uv; the first invocation downloads the Python environment and dependencies::

   uvx --python 3.12 --from xinference xinference-local

Use a recent uv supporting ``uv tool install --torch-backend`` (tested with uv 0.11.26). Services require a persistent installation; uvx environments can be removed by cache cleanup.

Upgrading an installation
~~~~~~~~~~~~~~~~~~~~~~~~~
Run the same one-command installer again to update to the latest stable release. Set ``XINFERENCE_VERSION`` each time to select a specific release, including a downgrade. Without this variable, a previously pinned installation also updates to the latest stable release. If the selected version, Python, extras, and PyTorch backend are unchanged, the installer keeps the existing environment and does not restart a running service.

The default ``XINFERENCE_SERVICE=auto`` detects an existing managed service for this tool environment. Upgrades preserve its service account, address, port, data directory, and registration. Python, extras, and the PyTorch backend are also retained unless explicitly overridden. Run as the original installation account and reuse any custom ``XINFERENCE_TOOL_DIR`` and ``UV_TOOL_BIN_DIR``. Stop a foreground server before updating it.

During an upgrade, the installer downloads and checks the candidate environment before stopping a service. It then backs up the complete previous environment, switches versions, and waits for the service to become ready. If installation or service startup fails, it restores the previous environment and the service's original running or stopped state. Preparing the candidate and keeping the full backup require additional disk space. On a first installation, failed startup removes a newly registered service and keeps the installed package for retrying. Model data and caches are preserved; models that were running must be launched again after a service restart. Foreground startup does not perform a service readiness check or automatic startup rollback.

Concurrent installers for the same tool store are rejected. If an upgrade is interrupted, the next run restores the saved environment before trying again. Keep the recovery files in the tool store until recovery completes. ``XINFERENCE_TIMEOUT`` sets the service readiness timeout in seconds (default 120). With ``XINFERENCE_START=0``, an updated service remains stopped.

System services
~~~~~~~~~~~~~~~
After installing with pip, Conda, or uv, register and start a local service with::

   xinference service install --start
   xinference service status
   xinference service logs
   xinference service restart
   xinference service stop
   xinference service uninstall

Linux uses a systemd user service. To start it at boot without logging in, enable lingering for the account with ``loginctl enable-linger USER``. macOS uses a LaunchAgent that starts at login. For a systemd system service or macOS LaunchDaemon, use an elevated terminal, an absolute command path, and ``--system`` before the action::

   sudo /absolute/path/to/xinference service --system install --user USER --start
   sudo /absolute/path/to/xinference service --system status
   sudo /absolute/path/to/xinference service --system uninstall

Windows services currently require x86-64 Windows and an Administrator PowerShell terminal and run as LocalSystem. The installer downloads a pinned, checksum-verified WinSW wrapper. Service control files and the working directory use ``%ProgramFiles%\Xinference\service``. Use an administrator-controlled Python installation and model data because the service executes model code with its account's privileges. Service data defaults to ``%PROGRAMDATA%\Xinference\data``. Linux and macOS default to the runtime account's ``~/.xinference``. Override the data directory with ``--home`` and the address with ``--host`` and ``--port`` on ``service install``.

A macOS user service requires a logged-in graphical session; use system mode on headless hosts. The launchd ``console.log`` does not rotate automatically.

For one-command service installation, set ``XINFERENCE_SERVICE=user`` on Linux/macOS or ``XINFERENCE_SERVICE=system`` on any supported platform. System mode uses sudo on Linux/macOS after installing as your account, and requires an Administrator Windows terminal. ``XINFERENCE_START=0`` registers without starting. ``XINFERENCE_HOST`` and ``XINFERENCE_PORT`` default to ``127.0.0.1`` and ``9997``. Service mode requires a Xinference release containing the service CLI::

   curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | XINFERENCE_SERVICE=user sh

   curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | XINFERENCE_SERVICE=system sh

   $env:XINFERENCE_SERVICE='system'; irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex

Services restart after failures. Starting waits for the cluster's ``/status`` response; a failed readiness check stops a newly started service. Repeated installation reuses the same configuration. Stop and uninstall before changing the installation environment or service settings. Uninstall preserves model data and logs. This service interface manages single-machine local mode.


Several usage scenarios require special attention.

.. admonition:: **GGUF format** with **llama.cpp engine**

   In this situation, it's advised to install its dependencies manually based on your hardware specifications to enable acceleration. For more details, see the :ref:`installation_gguf` section.

.. admonition:: **AWQ or GPTQ** format with **transformers engine**

   **This section is added in v1.6.0.**

   This is because the dependencies at this stage require special options and are difficult to install. Please run command below in advance

   .. code-block:: bash

      pip install "xinference[transformers_quantization]" --no-build-isolation

   Some dependencies like ``transformers`` might be downgraded, you can run ``pip install "xinference[all]"`` afterwards.


If you want to install only the necessary backends, here's a breakdown of how to do it.

.. _inference_backend:

Transformers Backend
~~~~~~~~~~~~~~~~~~~~
PyTorch (transformers) supports the inference of most state-of-art models. It is the default backend for models in PyTorch format::

   pip install "xinference[transformers]"

Notes:

- The transformers engine supports ``pytorch`` / ``gptq`` / ``awq`` / ``bnb`` / ``fp4`` formats.
- FP4 format requires ``transformers`` with ``FPQuantConfig`` support. If you see an import error,
  please upgrade ``transformers`` to a newer version.


vLLM Backend
~~~~~~~~~~~~
vLLM is a fast and easy-to-use library for LLM inference and serving. Xinference will choose vLLM as the backend to achieve better throughput when the following conditions are met:

- The model format is ``pytorch``, ``gptq``, ``awq``, ``fp4``, ``fp8`` or ``bnb``.
- When the model format is ``pytorch``, the quantization is ``none``.
- When the model format is ``awq``, the quantization is ``Int4``.
- When the model format is ``gptq``, the quantization is ``Int3``, ``Int4`` or ``Int8``.
- The system is Linux and has at least one CUDA device
- The model family (for custom models) / model name (for builtin models) is within the list of models supported by vLLM

Currently, supported models include:

.. vllm_start

- ``code-llama``, ``code-llama-instruct``, ``code-llama-python``, ``deepseek``, ``deepseek-chat``, ``deepseek-coder``, ``deepseek-coder-instruct``, ``deepseek-r1-distill-llama``, ``HuatuoGPT-o1-LLaMA-3.1``, ``llama-2``, ``llama-2-chat``, ``llama-3``, ``llama-3-instruct``, ``llama-3.1``, ``llama-3.1-instruct``, ``llama-3.3-instruct``, ``minicpm5-1b``, ``tiny-llama``, ``Yi``, ``Yi-1.5``, ``Yi-1.5-chat``, ``Yi-1.5-chat-16k``, ``Yi-200k``, ``Yi-chat``
- ``codestral-v0.1``, ``mistral-instruct-v0.1``, ``mistral-instruct-v0.2``, ``mistral-instruct-v0.3``, ``mistral-large-instruct``, ``mistral-nemo-instruct``, ``mistral-v0.1``, ``openhermes-2.5``, ``seallm_v2``
- ``Baichuan-M2``, ``codeqwen1.5``, ``codeqwen1.5-chat``, ``deepseek-r1-distill-qwen``, ``DianJin-R1``, ``fin-r1``, ``HuatuoGPT-o1-Qwen2.5``, ``KAT-V1``, ``marco-o1``, ``qwen1.5-chat``, ``qwen2-instruct``, ``qwen2.5``, ``qwen2.5-coder``, ``qwen2.5-coder-instruct``, ``qwen2.5-instruct``, ``qwen2.5-instruct-1m``, ``qwenLong-l1``, ``QwQ-32B``, ``QwQ-32B-Preview``, ``seallms-v3``, ``skywork-or1``, ``skywork-or1-preview``, ``vibethinker``, ``XiYanSQL-QwenCoder-2504``
- ``llama-3.2-vision``, ``llama-3.2-vision-instruct``
- ``baichuan-2``, ``baichuan-2-chat``
- ``InternLM2ForCausalLM``
- ``qwen-chat``
- ``mixtral-8x22B-instruct-v0.1``, ``mixtral-instruct-v0.1``, ``mixtral-v0.1``
- ``cogagent``
- ``glm-edge-chat``, ``glm4-chat``, ``glm4-chat-1m``
- ``codegeex4``, ``glm-4v``
- ``qwen3.8-max``
- ``seallm_v2.5``
- ``orion-chat``
- ``qwen1.5-moe-chat``, ``qwen2-moe-instruct``
- ``CohereForCausalLM``
- ``deepseek-v2-chat``, ``deepseek-v2-chat-0628``, ``deepseek-v2.5``, ``deepseek-vl2``
- ``deepseek-prover-v2``, ``deepseek-r1``, ``deepseek-r1-0528``, ``deepseek-v3``, ``deepseek-v3-0324``, ``Deepseek-V3.1``, ``moonlight-16b-a3b-instruct``
- ``deepseek-r1-0528-qwen3``, ``qwen3``
- ``minicpm3-4b``
- ``internlm3-instruct``
- ``gemma-3-1b-it``
- ``glm4-0414``
- ``minicpm-2b-dpo-bf16``, ``minicpm-2b-dpo-fp16``, ``minicpm-2b-dpo-fp32``, ``minicpm-2b-sft-bf16``, ``minicpm-2b-sft-fp32``, ``minicpm4``
- ``Ernie4.5``
- ``Qwen3-Coder``, ``Qwen3-Instruct``, ``Qwen3-Thinking``
- ``glm-4.5``, ``GLM-4.6``, ``GLM-4.7``
- ``gpt-oss``
- ``seed-oss``
- ``Qwen3-Next-Instruct``, ``Qwen3-Next-Thinking``
- ``DeepSeek-V3.2``, ``DeepSeek-V3.2-Exp``
- ``MiniMax-M2``, ``MiniMax-M2.5``, ``MiniMax-M2.7``
- ``GLM-4.7-Flash``
- ``glm-5``, ``glm-5.1``, ``glm-5.2``
- ``DeepSeek-V4-Flash``, ``DeepSeek-V4-Flash-0731``, ``DeepSeek-V4-Pro``
- ``Hy-MT2-1.8B``, ``Hy-MT2-7B``
- ``Hy-MT2-30B-A3B``

.. vllm_end

To install Xinference and vLLM::

   pip install "xinference[vllm]"
   
   # FlashInfer is optional but required for specific functionalities such as sliding window attention with Gemma 2.
   # For CUDA 12.4 & torch 2.4 to support sliding window attention for gemma 2 and llama 3.1 style rope
   pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4
   # For other CUDA & torch versions, please check https://docs.flashinfer.ai/installation.html
   

.. _installation_gguf:

Llama.cpp Backend
~~~~~~~~~~~~~~~~~
Xinference supports models in ``gguf`` format via ``xllamacpp``.
`xllamacpp <https://github.com/xorbitsai/xllamacpp>`_ is developed by Xinference team,
and is the sole backend for llama.cpp since v1.6.0.

.. warning::

    Since Xinference v1.5.0, ``llama-cpp-python`` is deprecated.
    Since Xinference v1.6.0, ``llama-cpp-python`` has been removed.

Initial setup::

   pip install "xinference[llama_cpp]"

With per-model virtual environments enabled, Xinference 3.0 automatically
selects the matching ``xllamacpp`` GPU wheel for supported CUDA versions; see
:ref:`user_guide_backends`. For a manual installation into the process
environment, refer to https://github.com/xorbitsai/xllamacpp.

SGLang Backend
~~~~~~~~~~~~~~
SGLang has a high-performance inference runtime with RadixAttention. It significantly accelerates the execution of complex LLM programs by automatic KV cache reuse across multiple calls. And it also supports other common techniques like continuous batching and tensor parallelism.

Initial setup::

   pip install "xinference[sglang]"


MLX Backend
~~~~~~~~~~~
MLX-lm is designed for Apple silicon users to run LLM efficiently.

Initial setup::

   pip install "xinference[mlx]"

.. only:: zh_cn

   Other Platforms
   ~~~~~~~~~~~~~~~

   * :ref:`Ascend NPU <installation_npu>`
