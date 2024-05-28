.. _models_llm_yi-1.5-chat:

========================================
Yi-1.5-chat
========================================

- **Context Length:** 4096
- **Model Name:** Yi-1.5-chat
- **Languages:** en, zh
- **Abilities:** chat
- **Description:** Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 6
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** 01-ai/Yi-1.5-6B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-6B-Chat>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-6B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 6 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 9
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** 01-ai/Yi-1.5-9B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-9B-Chat>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-9B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 9 --model-format pytorch --quantization ${quantization}


Model Spec 3 (pytorch, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 34
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: vLLM, Transformers (vLLM only available for quantization none)
- **Model ID:** 01-ai/Yi-1.5-34B-Chat
- **Model Hubs**:  `Hugging Face <https://huggingface.co/01-ai/Yi-1.5-34B-Chat>`__, `ModelScope <https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 34 --model-format pytorch --quantization ${quantization}


Model Spec 4 (ggufv2, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 6
- **Quantizations:** Q3_K_L, Q4_K_M, Q5_K_M, Q6_K, Q8_0, f32
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Yi-1.5-6B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Yi-1.5-6B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 6 --model-format ggufv2 --quantization ${quantization}


Model Spec 5 (ggufv2, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 9
- **Quantizations:** Q3_K_L, Q4_K_M, Q5_K_M, Q6_K, Q8_0, f32
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Yi-1.5-9B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Yi-1.5-9B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 9 --model-format ggufv2 --quantization ${quantization}


Model Spec 6 (ggufv2, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** ggufv2
- **Model Size (in billions):** 34
- **Quantizations:** Q2_K, Q3_K_L, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- **Engines**: llama.cpp
- **Model ID:** lmstudio-community/Yi-1.5-34B-Chat-GGUF
- **Model Hubs**:  `Hugging Face <https://huggingface.co/lmstudio-community/Yi-1.5-34B-Chat-GGUF>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 34 --model-format ggufv2 --quantization ${quantization}


Model Spec 7 (gptq, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 6
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-6B-Chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-6B-Chat-GPTQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-6B-Chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 6 --model-format gptq --quantization ${quantization}


Model Spec 8 (gptq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 9
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-9B-Chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-9B-Chat-GPTQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-9B-Chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 9 --model-format gptq --quantization ${quantization}


Model Spec 9 (gptq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** gptq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-34B-Chat-GPTQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-34B-Chat-GPTQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-34B-Chat-GPTQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 34 --model-format gptq --quantization ${quantization}


Model Spec 10 (awq, 6 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 6
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-6B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-6B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-6B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 6 --model-format awq --quantization ${quantization}


Model Spec 11 (awq, 9 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 9
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-9B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-9B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-9B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 9 --model-format awq --quantization ${quantization}


Model Spec 12 (awq, 34 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 34
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** modelscope/Yi-1.5-34B-Chat-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/modelscope/Yi-1.5-34B-Chat-AWQ>`__, `ModelScope <https://modelscope.cn/models/AI-ModelScope/Yi-1.5-34B-Chat-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Yi-1.5-chat --size-in-billions 34 --model-format awq --quantization ${quantization}

