.. _models_llm_qwen3-vl-thinking:

========================================
Qwen3-VL-Thinking
========================================

- **Context Length:** 262144
- **Model Name:** Qwen3-VL-Thinking
- **Languages:** en, zh
- **Abilities:** chat, vision, reasoning, tools
- **Description:** Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 235
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format pytorch --quantization ${quantization}


Model Spec 2 (fp8, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 235
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-235B-A22B-Thinking-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format fp8 --quantization ${quantization}


Model Spec 3 (awq, 235 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 235
- **Quantizations:** Int4
- **Engines**: vLLM, Transformers
- **Model ID:** QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ
- **Model Hubs**:  `Hugging Face <https://huggingface.co/QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ>`__, `ModelScope <https://modelscope.cn/models/tclf90/Qwen3-VL-235B-A22B-Thinking-AWQ>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 235 --model-format awq --quantization ${quantization}


Model Spec 4 (pytorch, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 30
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Thinking
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format pytorch --quantization ${quantization}


Model Spec 5 (fp8, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** fp8
- **Model Size (in billions):** 30
- **Quantizations:** fp8
- **Engines**: vLLM, Transformers
- **Model ID:** Qwen/Qwen3-VL-30B-A3B-Thinking-FP8
- **Model Hubs**:  `Hugging Face <https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8>`__, `ModelScope <https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking-FP8>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format fp8 --quantization ${quantization}


Model Spec 6 (awq, 30 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** awq
- **Model Size (in billions):** 30
- **Quantizations:** 4bit, 8bit
- **Engines**: vLLM, Transformers
- **Model ID:** cpatonn/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}
- **Model Hubs**:  `Hugging Face <https://huggingface.co/cpatonn/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}>`__, `ModelScope <https://modelscope.cn/models/cpatonn-mirror/Qwen3-VL-30B-A3B-Thinking-AWQ-{quantization}>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name Qwen3-VL-Thinking --size-in-billions 30 --model-format awq --quantization ${quantization}

