.. _models_builtin_qwen_chat:

=========
Qwen Chat
=========

- **Model Name:** qwen-chat
- **Languages:** en, zh
- **Abilities:** embed, chat

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 7 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Model ID:** Qwen/Qwen-7B-Chat

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format pytorch --quantization ${quantization}

.. note::

   4-bit and 8-bit quantization are not supported on macOS.

Model Spec 2 (pytorch, 14 Billion)
++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 14
- **Quantizations:** none
- **Model ID:** Qwen/Qwen-14B-Chat

Execute the following command to launch the model::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format pytorch

.. note::

   4-bit and 8-bit quantization are not supported on macOS.

Model Spec 3 (ggmlv3, 7 Billion)
++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 7
- **Quantizations:** q4_0
- **Model ID:** Xorbits/qwen-chat-7B-ggml

You need to install ``qwen-cpp`` first:

.. code-block:: bash

    pip install -U qwen-cpp


If you want to use BLAS to accelerate:

- OpenBLAS:

.. code-block:: bash

    CMAKE_ARGS="-DGGML_OPENBLAS=ON" pip install -U qwen-cpp


- cuBLAS:

.. code-block:: bash

    CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install -U qwen-cpp


- Metal:

.. code-block:: bash

    CMAKE_ARGS="-DGGML_METAL=ON" pip install -U qwen-cpp


Execute the following command to launch the model::

   xinference launch --model-name qwen-chat --size-in-billions 7 --model-format ggmlv3


Model Spec 4 (ggmlv3, 14 Billion)
+++++++++++++++++++++++++++++++++

- **Model Format:** ggmlv3
- **Model Size (in billions):** 14
- **Quantizations:** q4_0
- **Model ID:** Xorbits/qwen-chat-14B-ggml

Install ``qwen-cpp`` as above.

Execute the following command to launch the model::

   xinference launch --model-name qwen-chat --size-in-billions 14 --model-format ggmlv3

