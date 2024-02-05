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