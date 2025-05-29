.. _models_llm_deepseek-prover-v2:

========================================
deepseek-prover-v2
========================================

- **Context Length:** 163840
- **Model Name:** deepseek-prover-v2
- **Languages:** en, zh
- **Abilities:** chat, reasoning
- **Description:** We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 671 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 671
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-Prover-V2-671B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-Prover-V2-671B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-prover-v2 --size-in-billions 671 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** none
- **Engines**: vLLM, Transformers
- **Model ID:** deepseek-ai/DeepSeek-Prover-V2-7B
- **Model Hubs**:  `Hugging Face <https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B>`__, `ModelScope <https://modelscope.cn/models/deepseek-ai/DeepSeek-Prover-V2-7B>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-prover-v2 --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 3 (mlx, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** mlx
- **Model Size (in billions):** 7
- **Quantizations:** 4bit
- **Engines**: 
- **Model ID:** mlx-community/DeepSeek-Prover-V2-7B-4bit
- **Model Hubs**:  `Hugging Face <https://huggingface.co/mlx-community/DeepSeek-Prover-V2-7B-4bit>`__, `ModelScope <https://modelscope.cn/models/mlx-community/DeepSeek-Prover-V2-7B-4bit>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name deepseek-prover-v2 --size-in-billions 7 --model-format mlx --quantization ${quantization}

