.. _models_builtin_deepdoc:

=======
DeepDoc
=======

- **Model Name:** DeepDoc
- **Model Family:** ocr
- **Abilities:** ocr
- **Available ControlNet:** None

Specifications
^^^^^^^^^^^^^^

- **Model ID:** InfiniFlow/deepdoc (HuggingFace), Xorbits/deepdoc (ModelScope)
- **Inference package:** `deepdoc-lib <https://github.com/xorbitsai/deepdoc-lib>`_ (onnxruntime based)
- **Device:** GPU when CUDA is available, with CPU fallback. On Linux x86_64,
  Xinference installs the ``deepdoc-lib[gpu]`` extra (including
  ``onnxruntime-gpu``) and DeepDoc selects CUDAExecutionProvider. On workers
  without CUDA, Xinference installs the base ``deepdoc-lib`` package instead.
  Xinference reserves and exposes the assigned GPU to the model process.

Execute the following command to launch the model::

   xinference launch --model-name DeepDoc --model-type image --model-engine deepdoc

The ``/v1/images/ocr`` endpoint accepts a ``task`` kwarg: ``ocr`` (default,
returns plain text), ``layout`` (returns layout blocks) or ``table`` (returns
table structures). Structured results are returned as JSON objects.

