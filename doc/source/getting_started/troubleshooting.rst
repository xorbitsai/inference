.. _troubleshooting:

===============
Troubleshooting
===============


No huggingface repo access
==========================

Sometimes, you may face errors accessing huggingface models, such as the following message when accessing `llama2`:

.. code-block:: text

   Cannot access gated repo for url https://huggingface.co/api/models/meta-llama/Llama-2-7b-hf.
   Repo model meta-llama/Llama-2-7b-hf is gated. You must be authenticated to access it.

This typically indicates either a lack of access rights to the repository or missing huggingface access tokens. 
The following sections provide guidance on addressing these issues.

Get access to the huggingface repo
----------------------------------

To obtain access, navigate to the desired huggingface repository and agree to its terms and conditions. 
As an illustration, for the `llama2` model, you can use this link:
`https://huggingface.co/meta-llama/Llama-2-7b-hf <https://huggingface.co/meta-llama/Llama-2-7b-hf>`_.

Set up credentials to access huggingface
----------------------------------------

Your credential to access huggingface can be found online at `https://huggingface.co/settings/tokens <https://huggingface.co/settings/tokens>`_.

You can set the token as an environmental variable, with ``export HUGGING_FACE_HUB_TOKEN=your_token_here``.


Incompatibility Between NVIDIA Driver and PyTorch Version
=========================================================

If you are using a NVIDIA GPU, you may face the following error:

.. code-block:: text

   UserWarning: CUDA initialization: The NVIDIA driver on your system is too old
   (found version 10010). Please update your GPU driver by downloading and installi
   ng a new version from the URL: http://www.nvidia.com/Download/index.aspx Alterna
   tively, go to: https://pytorch.org to install a PyTorch version that has been co
   mpiled with your version of the CUDA driver. (Triggered internally at  ..\c10\cu
   da\CUDAFunctions.cpp:112.)

This typically indicates that your CUDA driver version is not compatible with the PyTorch version you are using.

Go to `https://pytorch.org <https://pytorch.org>`_ to install a PyTorch version that has been compiled with your
version of the CUDA driver. **Do not install a cuda version smaller than 11.8, preferably between 11.8 and 12.1.**

Say if your CUDA driver version is 11.8, then you can install PyTorch with the following command:

.. code-block:: python

   pip install torch==2.0.1+cu118


Xinference service cannot be accessed from external systems through ``<IP>:9997``
=================================================================================

Use ``-H 0.0.0.0`` parameter in when starting Xinference:

.. code:: bash

   xinference-local -H 0.0.0.0

Then Xinference service will listen on all network interfaces (not limited to ``127.0.0.1`` or ``localhost``).

If you are using the :ref:`using_docker_image`, please add ``-p <PORT>:9997``
during the docker run command, then access is available through ``<IP>:<PORT>`` of
the local machine.

Launching a built-in model takes a long time, and sometimes the model fails to download
=======================================================================================

Xinference by default uses HuggingFace as the source for models. If your
machines are in Mainland China, there might be accessibility issues when
using built-in models.

To address this, add environment variable ``XINFERENCE_MODEL_SRC=modelscope`` when starting
the Xinference to change the model source to ModelScope, which is optimized
for Mainland China.

If you’re starting Xinference with Docker, include ``-e XINFERENCE_MODEL_SRC=modelscope``
during the docker run command.

When using the official Docker image, RayWorkerVllm died due to OOM, causing the model to fail to load
=======================================================================================================

Docker's ``--shm-size`` parameter is used to set the size of shared memory. 
The default size of shared memory (/dev/shm) is 64MB, which may be too small for vLLM backend.


You can increase its size by setting the ``--shm-size`` parameter as follows:

.. code:: bash

   docker run --shm-size=128g ...


Missing ``model_engine`` parameter when launching LLM models
============================================================

Since version ``v0.11.0``, launching LLM models requires an additional ``model_engine`` parameter.
For specific information, please refer to :ref:`here <about_model_engine>`.

Resolving MKL Threading Layer Conflicts
========================================

When starting the Xinference server, you may encounter the error: ``ValueError: Model architectures ['Qwen2ForCausalLM'] failed to be inspected. Please check the logs for more details.``

The underlying cause shown in the logs is:

.. code-block:: text

   Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
   Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.

This typically occurs when NumPy was installed via conda. Conda's NumPy is built with Intel MKL optimizations, which conflicts with the GNU OpenMP library (libgomp) already loaded in your environment.

Solution 1: Override the Threading Layer
-----------------------------------------

Force Intel's Math Kernel Library to use GNU's OpenMP implementation:

.. code-block:: bash

   MKL_THREADING_LAYER=GNU xinference-local

Solution 2: Reinstall NumPy with pip
-------------------------------------

Uninstall conda's NumPy and reinstall using pip:

.. code-block:: bash

   pip uninstall -y numpy && pip install numpy
   #Or just --force-reinstall
   pip install --force-reinstall numpy

Related Note: vLLM and PyTorch
-------------------------------

If you're using vLLM, avoid installing PyTorch with conda. Refer to the official vLLM installation guide for GPU-specific instructions: https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html

Configuring PyPI Mirrors to Speed Up Package Installation
==========================================================

If you're in Mainland China, using a PyPI mirror can significantly speed up package installation. Here are some commonly used mirrors:

- Tsinghua University: ``https://pypi.tuna.tsinghua.edu.cn/simple``
- Alibaba Cloud: ``https://mirrors.aliyun.com/pypi/simple/``
- Tencent Cloud: ``https://mirrors.cloud.tencent.com/pypi/simple``

However, be aware that some packages may not be available on certain mirrors. For example, if you're installing ``xinference[audio]`` using only the Aliyun mirror, the installation may fail.

This happens because ``num2words``, a dependency used by ``MeloTTS``, is not available on the Aliyun mirror. As a result, ``pip install xinference[audio]`` will resolve to older versions like ``xinference==1.2.0`` and ``xoscar==0.8.0`` (as of Oct 27, 2025).

These older versions are incompatible and will produce the error: ``MainActorPool.append_sub_pool() got an unexpected keyword argument 'start_method'``

.. code-block:: bash

   curl -s https://mirrors.aliyun.com/pypi/simple/num2words/ | grep -i "num2words"
   # Returns NOTHING! But it works on Tsinghua or Tencent mirrors.
   # uv pip install "xinference[audio]" will then install the following packages (as of Oct 27, 2025):
   + x-transformers==2.10.2
   + xinference==1.2.0
   + xoscar==0.8.0

To avoid this issue when installing the xinference audio package, use multiple mirrors:

.. code-block:: bash

   uv pip install xinference[audio] --index-url https://mirrors.aliyun.com/pypi/simple --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

   # Optional: Set this globally in your uv config
   mkdir -p ~/.config/uv
   cat >> ~/.config/uv/uv.toml << EOF
   [tool.uv]
   index-url = "https://mirrors.aliyun.com/pypi/simple"
   extra-index-url = ["https://pypi.tuna.tsinghua.edu.cn/simple"]
   EOF
