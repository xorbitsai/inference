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
version of the CUDA driver.

Say if your CUDA driver version is 11.8, then you can install PyTorch with the following command:

.. code-block:: python

   pip install torch==2.0.1+cu118
