.. _models_builtin_nemotron-3.5-embed-8b-bf16:

==========================
Nemotron-3.5-Embed-8B-BF16
==========================

- **Model Name:** Nemotron-3.5-Embed-8B-BF16
- **Languages:** multilingual
- **Abilities:** embed

Specifications
^^^^^^^^^^^^^^

- **Dimensions:** 4096
- **Max Tokens:** 32768
- **Model ID:** nvidia/Nemotron-3.5-Embed-8B-BF16
- **Model Hubs**: `Hugging Face <https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16>`__, `ModelScope <https://modelscope.cn/models/nv-community/Nemotron-3.5-Embed-8B-BF16>`__

Execute the following command to launch the model::

   xinference launch --model-name Nemotron-3.5-Embed-8B-BF16 --model-type embedding


Multimodal retrieval
^^^^^^^^^^^^^^^^^^^^

Supported engines are ``sentence_transformers`` (default), ``transformers``,
and ``vllm``. Select one with ``--model-engine``.

Send ``input_type="query"`` for text queries and ``input_type="passage"``
(the default) for documents. ``prompt_name="query"`` and
``prompt_name="document"`` are equivalent. Queries accept text only;
documents accept text, one image, or one image combined with text.

Use ``image`` or ``image_url`` for an HTTP(S) URL, a base64 data URL, or a
local path on the Xinference server. Lists preserve input order. For example,
send this body to ``/v1/embeddings``:

.. code-block:: json

   {
     "model": "<MODEL_UID>",
     "input": [
       "A text document",
       {"image": "https://example.com/page.png"},
       {"text": "A page caption", "image": "https://example.com/page.png"}
     ],
     "input_type": "passage",
     "dimensions": 512
   }

``dimensions`` accepts a positive integer up to the native embedding size.
Vectors are mean-pooled and L2-normalized after truncating dimensions.
The 32,768-token context includes image tokens. Audio and video are not supported.

The Transformers and Sentence Transformers engines use Transformers 5.20.0
or later. The vLLM environment uses
vLLM 0.32.0 or later with ``transformers>=5.17.0,<5.20.0``; use a separate
environment from the Transformers and Sentence Transformers engines.
