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
- **Device:** CPU only. ``deepdoc-lib`` runs on the CPU onnxruntime backend,
  so launching DeepDoc never reserves a GPU, even on GPU workers.

Execute the following command to launch the model::

   xinference launch --model-name DeepDoc --model-type image --model-engine deepdoc

The ``/v1/images/ocr`` endpoint accepts a ``task`` kwarg: ``ocr`` (default,
returns plain text), ``layout`` (returns layout blocks) or ``table`` (returns
table structures). Structured results are returned as JSON objects.



