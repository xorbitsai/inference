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

Download models from ModelScope
===============================

When the network connection to HuggingFace is blocked, you can also choose to download models from ModelScope, especially for Chinese users.
For a detailed list of supported models and settings, please refer to :ref:`models_download`.


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


Access is possible through ``http://localhost:9997``, but not through ``ip+9997`` of the local machine.
=======================================================================================================

Add ``H 0.0.0.0`` when starting, as in

.. code:: bash

   xinference -H 0.0.0.0

If using docker (official docker is recommended), add ``p 9998:9997``
when starting, then access is available through ``ip+9998`` of the local
machine.

Can multiple models be loaded together?
=======================================

A single GPU can only support loading one LLM model at a time, but it is
possible to load an embedding model and a rerank model simultaneously.
With multiple GPUs, you can load multiple LLM models.

Issues with loading or slow downloads of the built-in models in xinference
==========================================================================

Xinference by default uses huggiface as the source for models. If your
machines are in Mainland China, there might be accessibility issues when
using built-in models.

To address this, add ``XINFERENCE_MODEL_SRC=modelscope`` when starting
the service to change the model source to ModelScope, which is optimized
for Mainland China.

If you’re starting xinference with Docker, include
``e XINFERENCE_MODEL_SRC=modelscope`` during the docker run command. For
more environment variable configurations, please refer to the official
[`Environment
Variables <https://inference.readthedocs.io/zh-cn/latest/getting_started/environments.html>`__]
documentation.

How to upgrade xinference
=========================

.. code:: bash

   pip install --upgrade xinference

Installation of xinference dependencies is slow
===============================================

We are recommended to use the official docker image for installation.
There is a nightly-main version based on the main branch updated daily.
For stable versions, see GitHub.

.. code:: bash

   docker pull xprobe/xinference

Does xinference support configuring LoRA?
=========================================

It is currently not supported; it requires manual integration with the
main model.

Can’t find a custom registration entry point for rerank models in xinference
============================================================================

Upgrade inference to the latest version, versions ``0.7.3`` and below
are not supported.

Does xinference support running on Huawei Ascend 310 or 910 hardware?
=====================================================================

Yes, it does.

Does xinference support an API that is compatible with OpenAI?
==============================================================

Yes, xinference not only supports an API compatible with OpenAI but also
has a client API available for use. For more details, please visit the
official website `Client
API <https://inference.readthedocs.io/zh-cn/latest/user_guide/client_api.html>`__.

When using xinference to load models, multi-GPU support is not functioning, and it only loads onto one card.
============================================================================================================

-  If you are using Docker for vLLM multi-GPU inference, you need to
   specify ``-shm-size``.

-  If the vLLM backend is in use, you should disable vLLM before
   performing the inference.

Does Xinference support setting up a chat model for embeddings?
===============================================================

It used to. But since the embedding performance of LLMs was poor, the
feature has been removed to prevent misuse.
