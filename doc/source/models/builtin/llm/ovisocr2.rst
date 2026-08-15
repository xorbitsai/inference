.. _models_llm_ovisocr2:

========
OvisOCR2
========

- **Context Length:** 262144
- **Model Name:** OvisOCR2
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning, hybrid
- **Description:** OvisOCR2 is a compact 0.8B end-to-end model for page-level document parsing. It converts document page images into Markdown in natural reading order, including text, formulas, tables, and visual regions.

Specifications
^^^^^^^^^^^^^^

Model Spec 1 (pytorch, 0_8 Billion)
++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 0_8
- **Quantizations:** none
- **Engines:** vLLM, Transformers, SGLang
- **Model ID:** ATH-MaaS/OvisOCR2
- **Model Hubs:** `Hugging Face <https://huggingface.co/ATH-MaaS/OvisOCR2>`__, `ModelScope <https://modelscope.cn/models/ATH-MaaS/OvisOCR2>`__

Launch the model with one of the supported engines::

   xinference launch --model-engine ${engine} --model-name OvisOCR2 --size-in-billions 0_8 --model-format pytorch --quantization none

Usage notes
^^^^^^^^^^^

Send one document page image at a time through the OpenAI-compatible chat
completions API and ask the model to reproduce the page as Markdown while
preserving its natural reading order. For dense pages, the model authors
recommend allowing up to 16384 output tokens and using deterministic decoding.
The recommended image pixel range is 448 x 448 to 2880 x 2880. Pass these
generation and image-processing options explicitly when needed; Xinference
does not inject model-specific runtime defaults.
