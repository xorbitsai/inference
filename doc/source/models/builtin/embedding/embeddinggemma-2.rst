.. _models_builtin_embeddinggemma-2:

================
embeddinggemma-2
================

- **Model Name:** embeddinggemma-2
- **Languages:** multilingual
- **Abilities:** embed, embed_vision, embed_video, embed_audio

Specifications
^^^^^^^^^^^^^^

- **Dimensions:** 768
- **Max Tokens:** 8192
- **Model ID:** google/embeddinggemma-2 (PyTorch), unsloth/embeddinggemma-2-GGUF (GGUF)
- **Formats:** pytorch, ggufv2
- **GGUF Engine:** llama.cpp
- **GGUF Quantizations:** BF16, F16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL
- **Model Hubs**: `Hugging Face <https://huggingface.co/google/embeddinggemma-2>`__, `ModelScope <https://modelscope.cn/models/google/embeddinggemma-2>`__
- **GGUF Model Hubs**: `Hugging Face <https://huggingface.co/unsloth/embeddinggemma-2-GGUF>`__, `ModelScope <https://modelscope.cn/models/unsloth/embeddinggemma-2-GGUF>`__

Execute the following command to launch the model::

   xinference launch --model-name embeddinggemma-2 --model-type embedding
