.. _user_guide_backends:

========
Backends
========

Xinference supports multiple backends for different models. After the user specifies the model,
xinference will automatically select the appropriate backend.

llama-cpp-python
~~~~~~~~~~~~~~~~
`llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_ is the python binding of
`llama.cpp`. `llama-cpp` is developed based on the tensor library `ggml`, supporting inference of
the LLaMA series models and their variants.

We recommend that users install `llama-cpp-python` on the worker themselves and adjust the `cmake`
parameters according to the hardware to achieve the best inference efficiency. Please refer to the
`llama-cpp-python installation guide <https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal>`_.

ctransformers
~~~~~~~~~~~~~
CTransformers provide python bindings for the Transformer models implemented in C/C++ using GGML library.

We recommend that users install `ctransformers` on the worker themselves and adjust the parameters
according to the hardware to achieve the best inference efficiency. Please refer to the
`ctransformers installation guide <https://github.com/marella/ctransformers#gpu>`_.


PyTorch
~~~~~~~
The PyTorch backend can support the inference of most PyTorch format models. It is the default
implementation for models in PyTorch format.

vLLM
~~~~
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:
- State-of-the-art serving throughput
- Efficient management of attention key and value memory with PagedAttention
- Continuous batching of incoming requests
- Optimized CUDA kernels

When the following conditions are met, Xinference will choose vLLM as the inference engine:

- The model format is pytorch
- The quantization method is none
- The system is Linux and has a CUDA device
- The model is within the list of models supported by vLLM.

Currently, supported model includes:
- ``llama-2``, ``llama-2-chat``
- ``baichuan``, ``baichuan-chat``
- ``internlm``, ``internlm-20b``, ``internlm-chat``, ``internlm-chat-20b``
- ``vicuna-v1.3``, ``vicuna-v1.5``
