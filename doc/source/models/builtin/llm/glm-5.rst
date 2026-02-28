.. _models_llm_glm-5:

========================================
glm-5
========================================

- **Context Length:** 202752
- **Model Name:** glm-5
- **Languages:** en, zh
- **Abilities:** chat, vision, tools, reasoning
- **Description:** We are launching GLM-5, targeting complex systems engineering and long-horizon agentic tasks. Scaling is still one of the most important ways to improve the intelligence efficiency of Artificial General Intelligence (AGI). Compared to GLM-4.5, GLM-5 scales from 355B parameters (32B active) to 744B parameters (40B active), and increases pre-training data from 23T to 28.5T tokens. GLM-5 also integrates DeepSeek Sparse Attention (DSA), largely reducing deployment cost while preserving long-context capacity.  Reinforcement learning aims to bridge the gap between competence and excellence in pre-trained models. However, deploying it at scale for LLMs is a challenge due to the RL training inefficiency. To this end, we developed slime, a novel asynchronous RL infrastructure that substantially improves training throughput and efficiency, enabling more fine-grained post-training iterations. With advances in both pre-training and post-training, GLM-5 delivers significant improvement compared to GLM-4.7 across a wide range of academic benchmarks and achieves best-in-class performance among all open-source models in the world on reasoning, coding, and agentic tasks, closing the gap with frontier models.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 744 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 744
- **Quantizations:** none
- **Engines**: Transformers
- **Model ID:** zai-org/GLM-5
- **Model Hubs**:  `Hugging Face <https://huggingface.co/zai-org/GLM-5>`__, `ModelScope <https://modelscope.cn/models/ZhipuAI/GLM-5>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name glm-5 --size-in-billions 744 --model-format pytorch --quantization ${quantization}

