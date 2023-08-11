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
`installation guide <https://github.com/abetlen/llama-cpp-python#installation-with-openblas--cublas--clblast--metal>`_.

PyTorch
~~~~~~~
The PyTorch backend can support the inference of most PyTorch format models.

ctransformers
~~~~~~~~~~~~~
Coming soon.

vLLM
~~~~
Coming soon.