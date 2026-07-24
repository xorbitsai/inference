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

Execute the following command to launch the model::

   xinference launch --model-name DeepDoc --model-type image --model-engine deepdoc

The ``/v1/images/ocr`` endpoint accepts a ``task`` kwarg: ``ocr`` (default,
plain text), ``layout`` (layout blocks as JSON) or ``table`` (table
structures as JSON).



