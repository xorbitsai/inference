.. _models_llm_index:

=====================
Large language Models
=====================

The following is a list of built-in LLM in Xinference:

.. list-table::
   :widths: 25 25 25 50
   :header-rows: 1

   * - MODEL NAME
     - ABILITIES
     - COTNEXT_LENGTH
     - DESCRIPTION


   * - :ref:`baichuan-2 <models_llm_baichuan-2>`
     - generate
     - 4096
     - Baichuan2 is an open-source Transformer based LLM that is trained on both Chinese and English data.

   * - :ref:`baichuan-2-chat <models_llm_baichuan-2-chat>`
     - chat
     - 4096
     - Baichuan2-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

   * - :ref:`baichuan-m2 <models_llm_baichuan-m2>`
     - chat, reasoning, hybrid, tools
     - 131072
     - Baichuan-M2-32B is Baichuan AI's medical-enhanced reasoning model, the second medical model released by Baichuan. Designed for real-world medical reasoning tasks, this model builds upon Qwen2.5-32B with an innovative Large Verifier System. Through domain-specific fine-tuning on real-world medical questions, it achieves breakthrough medical performance while maintaining strong general capabilities.

   * - :ref:`code-llama <models_llm_code-llama>`
     - generate
     - 100000
     - Code-Llama is an open-source LLM trained by fine-tuning LLaMA2 for generating and discussing code.

   * - :ref:`code-llama-instruct <models_llm_code-llama-instruct>`
     - chat
     - 100000
     - Code-Llama-Instruct is an instruct-tuned version of the Code-Llama LLM.

   * - :ref:`code-llama-python <models_llm_code-llama-python>`
     - generate
     - 100000
     - Code-Llama-Python is a fine-tuned version of the Code-Llama LLM, specializing in Python.

   * - :ref:`codegeex4 <models_llm_codegeex4>`
     - chat
     - 131072
     - the open-source version of the latest CodeGeeX4 model series

   * - :ref:`codeqwen1.5 <models_llm_codeqwen1.5>`
     - generate
     - 65536
     - CodeQwen1.5 is the Code-Specific version of Qwen1.5. It is a transformer-based decoder-only language model pretrained on a large amount of data of codes.

   * - :ref:`codeqwen1.5-chat <models_llm_codeqwen1.5-chat>`
     - chat
     - 65536
     - CodeQwen1.5 is the Code-Specific version of Qwen1.5. It is a transformer-based decoder-only language model pretrained on a large amount of data of codes.

   * - :ref:`codeshell <models_llm_codeshell>`
     - generate
     - 8194
     - CodeShell is a multi-language code LLM developed by the Knowledge Computing Lab of Peking University. 

   * - :ref:`codeshell-chat <models_llm_codeshell-chat>`
     - chat
     - 8194
     - CodeShell is a multi-language code LLM developed by the Knowledge Computing Lab of Peking University.

   * - :ref:`codestral-v0.1 <models_llm_codestral-v0.1>`
     - generate
     - 32768
     - Codestrall-22B-v0.1 is trained on a diverse dataset of 80+ programming languages, including the most popular ones, such as Python, Java, C, C++, JavaScript, and Bash

   * - :ref:`cogagent <models_llm_cogagent>`
     - chat, vision
     - 4096
     - The CogAgent-9B-20241220 model is based on GLM-4V-9B, a bilingual open-source VLM base model. Through data collection and optimization, multi-stage training, and strategy improvements, CogAgent-9B-20241220 achieves significant advancements in GUI perception, inference prediction accuracy, action space completeness, and task generalizability. 

   * - :ref:`deepseek <models_llm_deepseek>`
     - generate
     - 4096
     - DeepSeek LLM, trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. 

   * - :ref:`deepseek-chat <models_llm_deepseek-chat>`
     - chat
     - 4096
     - DeepSeek LLM is an advanced language model comprising 67 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese.

   * - :ref:`deepseek-coder <models_llm_deepseek-coder>`
     - generate
     - 16384
     - Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. 

   * - :ref:`deepseek-coder-instruct <models_llm_deepseek-coder-instruct>`
     - chat
     - 16384
     - deepseek-coder-instruct is a model initialized from deepseek-coder-base and fine-tuned on 2B tokens of instruction data.

   * - :ref:`deepseek-prover-v2 <models_llm_deepseek-prover-v2>`
     - chat, reasoning
     - 163840
     - We introduce DeepSeek-Prover-V2, an open-source large language model designed for formal theorem proving in Lean 4, with initialization data collected through a recursive theorem proving pipeline powered by DeepSeek-V3. The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals. The proofs of resolved subgoals are synthesized into a chain-of-thought process, combined with DeepSeek-V3's step-by-step reasoning, to create an initial cold start for reinforcement learning. This process enables us to integrate both informal and formal mathematical reasoning into a unified model

   * - :ref:`deepseek-r1 <models_llm_deepseek-r1>`
     - chat, reasoning
     - 163840
     - DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.

   * - :ref:`deepseek-r1-0528 <models_llm_deepseek-r1-0528>`
     - chat, reasoning, tools
     - 163840
     - DeepSeek-R1, which incorporates cold-start data before RL. DeepSeek-R1 achieves performance comparable to OpenAI-o1 across math, code, and reasoning tasks.

   * - :ref:`deepseek-r1-0528-qwen3 <models_llm_deepseek-r1-0528-qwen3>`
     - chat, reasoning
     - 131072
     - The DeepSeek R1 model has undergone a minor version upgrade, with the current version being DeepSeek-R1-0528. In the latest update, DeepSeek R1 has significantly improved its depth of reasoning and inference capabilities by leveraging increased computational resources and introducing algorithmic optimization mechanisms during post-training. The model has demonstrated outstanding performance across various benchmark evaluations, including mathematics, programming, and general logic. Its overall performance is now approaching that of leading models, such as O3 and Gemini 2.5 Pro

   * - :ref:`deepseek-r1-distill-llama <models_llm_deepseek-r1-distill-llama>`
     - chat, reasoning
     - 131072
     - deepseek-r1-distill-llama is distilled from DeepSeek-R1 based on Llama

   * - :ref:`deepseek-r1-distill-qwen <models_llm_deepseek-r1-distill-qwen>`
     - chat, reasoning
     - 131072
     - deepseek-r1-distill-qwen is distilled from DeepSeek-R1 based on Qwen

   * - :ref:`deepseek-v2-chat <models_llm_deepseek-v2-chat>`
     - chat
     - 128000
     - DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. 

   * - :ref:`deepseek-v2-chat-0628 <models_llm_deepseek-v2-chat-0628>`
     - chat
     - 128000
     - DeepSeek-V2-Chat-0628 is an improved version of DeepSeek-V2-Chat. 

   * - :ref:`deepseek-v2.5 <models_llm_deepseek-v2.5>`
     - chat
     - 128000
     - DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions.

   * - :ref:`deepseek-v3 <models_llm_deepseek-v3>`
     - chat
     - 163840
     - DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

   * - :ref:`deepseek-v3-0324 <models_llm_deepseek-v3-0324>`
     - chat
     - 163840
     - DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. 

   * - :ref:`deepseek-v3.1 <models_llm_deepseek-v3.1>`
     - chat, reasoning, hybrid, tools
     - 131072
     - DeepSeek-V3.1 is a hybrid model that supports both thinking mode and non-thinking mode.

   * - :ref:`deepseek-v3.2 <models_llm_deepseek-v3.2>`
     - chat, reasoning, hybrid, tools
     - 163840
     - We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance

   * - :ref:`deepseek-v3.2-exp <models_llm_deepseek-v3.2-exp>`
     - chat, reasoning, hybrid, tools
     - 163840
     - We are excited to announce the official release of DeepSeek-V3.2-Exp, an experimental version of our model. As an intermediate step toward our next-generation architecture, V3.2-Exp builds upon V3.1-Terminus by introducing DeepSeek Sparse Attention—a sparse attention mechanism designed to explore and validate optimizations for training and inference efficiency in long-context scenarios.

   * - :ref:`deepseek-vl2 <models_llm_deepseek-vl2>`
     - chat, vision
     - 4096
     - DeepSeek-VL2, an advanced series of large Mixture-of-Experts (MoE) Vision-Language Models that significantly improves upon its predecessor, DeepSeek-VL. DeepSeek-VL2 demonstrates superior capabilities across various tasks, including but not limited to visual question answering, optical character recognition, document/table/chart understanding, and visual grounding.

   * - :ref:`dianjin-r1 <models_llm_dianjin-r1>`
     - chat, tools
     - 32768
     - Tongyi DianJin is a financial intelligence solution platform built by Alibaba Cloud, dedicated to providing financial business developers with a convenient artificial intelligence application development environment.

   * - :ref:`ernie4.5 <models_llm_ernie4.5>`
     - chat
     - 131072
     - ERNIE 4.5, a new family of large-scale multimodal models comprising 10 distinct variants.

   * - :ref:`fin-r1 <models_llm_fin-r1>`
     - chat
     - 131072
     - Fin-R1 is a large language model specifically designed for the field of financial reasoning

   * - :ref:`gemma-3-1b-it <models_llm_gemma-3-1b-it>`
     - chat
     - 32768
     - Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

   * - :ref:`gemma-3-it <models_llm_gemma-3-it>`
     - chat, vision
     - 131072
     - Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

   * - :ref:`glm-4.1v-thinking <models_llm_glm-4.1v-thinking>`
     - chat, vision, reasoning, tools
     - 65536
     - GLM-4.1V-9B-Thinking, designed to explore the upper limits of reasoning in vision-language models.

   * - :ref:`glm-4.5 <models_llm_glm-4.5>`
     - chat, reasoning, hybrid, tools
     - 131072
     - The GLM-4.5 series models are foundation models designed for intelligent agents. 

   * - :ref:`glm-4.5v <models_llm_glm-4.5v>`
     - chat, vision, reasoning, tools
     - 131072
     - GLM-4.5V is based on ZhipuAI’s next-generation flagship text foundation model GLM-4.5-Air (106B parameters, 12B active). It continues the technical approach of GLM-4.1V-Thinking, achieving SOTA performance among models of the same scale on 42 public vision-language benchmarks.

   * - :ref:`glm-4v <models_llm_glm-4v>`
     - chat, vision
     - 8192
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`glm-edge-chat <models_llm_glm-edge-chat>`
     - chat
     - 8192
     - The GLM-Edge series is our attempt to face the end-side real-life scenarios, which consists of two sizes of large-language dialogue models and multimodal comprehension models (GLM-Edge-1.5B-Chat, GLM-Edge-4B-Chat, GLM-Edge-V-2B, GLM-Edge-V-5B). Among them, the 1.5B / 2B model is mainly for platforms such as mobile phones and cars, and the 4B / 5B model is mainly for platforms such as PCs.

   * - :ref:`glm4-0414 <models_llm_glm4-0414>`
     - chat, tools
     - 32768
     - The GLM family welcomes new members, the GLM-4-32B-0414 series models, featuring 32 billion parameters. Its performance is comparable to OpenAI’s GPT series and DeepSeek’s V3/R1 series

   * - :ref:`glm4-chat <models_llm_glm4-chat>`
     - chat, tools
     - 131072
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`glm4-chat-1m <models_llm_glm4-chat-1m>`
     - chat, tools
     - 1048576
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`gorilla-openfunctions-v2 <models_llm_gorilla-openfunctions-v2>`
     - chat
     - 4096
     - OpenFunctions is designed to extend Large Language Model (LLM) Chat Completion feature to formulate executable APIs call given natural language instructions and API context.

   * - :ref:`gpt-2 <models_llm_gpt-2>`
     - generate
     - 1024
     - GPT-2 is a Transformer-based LLM that is trained on WebTest, a 40 GB dataset of Reddit posts with 3+ upvotes.

   * - :ref:`gpt-oss <models_llm_gpt-oss>`
     - chat, reasoning
     - 131072
     - gpt-oss series, OpenAI’s open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases.

   * - :ref:`huatuogpt-o1-llama-3.1 <models_llm_huatuogpt-o1-llama-3.1>`
     - chat, tools
     - 131072
     - HuatuoGPT-o1 is a medical LLM designed for advanced medical reasoning. It generates a complex thought process, reflecting and refining its reasoning, before providing a final response.

   * - :ref:`huatuogpt-o1-qwen2.5 <models_llm_huatuogpt-o1-qwen2.5>`
     - chat, tools
     - 32768
     - HuatuoGPT-o1 is a medical LLM designed for advanced medical reasoning. It generates a complex thought process, reflecting and refining its reasoning, before providing a final response.

   * - :ref:`internlm3-instruct <models_llm_internlm3-instruct>`
     - chat, tools
     - 32768
     - InternLM3 has open-sourced an 8-billion parameter instruction model, InternLM3-8B-Instruct, designed for general-purpose usage and advanced reasoning.

   * - :ref:`internvl3 <models_llm_internvl3>`
     - chat, vision
     - 8192
     - InternVL3, an advanced multimodal large language model (MLLM) series that demonstrates superior overall performance.

   * - :ref:`kat-v1 <models_llm_kat-v1>`
     - chat
     - 131072
     - Kwaipilot-AutoThink ranks first among all open-source models on LiveCodeBench Pro, a challenging benchmark explicitly designed to prevent data leakage, and even surpasses strong proprietary systems such as Seed and o3-mini.

   * - :ref:`llama-2 <models_llm_llama-2>`
     - generate
     - 4096
     - Llama-2 is the second generation of Llama, open-source and trained on a larger amount of data.

   * - :ref:`llama-2-chat <models_llm_llama-2-chat>`
     - chat
     - 4096
     - Llama-2-Chat is a fine-tuned version of the Llama-2 LLM, specializing in chatting.

   * - :ref:`llama-3 <models_llm_llama-3>`
     - generate
     - 8192
     - Llama 3 is an auto-regressive language model that uses an optimized transformer architecture

   * - :ref:`llama-3-instruct <models_llm_llama-3-instruct>`
     - chat
     - 8192
     - The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

   * - :ref:`llama-3.1 <models_llm_llama-3.1>`
     - generate
     - 131072
     - Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture

   * - :ref:`llama-3.1-instruct <models_llm_llama-3.1-instruct>`
     - chat, tools
     - 131072
     - The Llama 3.1 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

   * - :ref:`llama-3.2-vision <models_llm_llama-3.2-vision>`
     - generate, vision
     - 131072
     - The Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image...

   * - :ref:`llama-3.2-vision-instruct <models_llm_llama-3.2-vision-instruct>`
     - chat, vision
     - 131072
     - Llama 3.2-Vision instruction-tuned models are optimized for visual recognition, image reasoning, captioning, and answering general questions about an image...

   * - :ref:`llama-3.3-instruct <models_llm_llama-3.3-instruct>`
     - chat, tools
     - 131072
     - The Llama 3.3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks..

   * - :ref:`marco-o1 <models_llm_marco-o1>`
     - chat, tools
     - 32768
     - Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions

   * - :ref:`minicpm-2b-dpo-bf16 <models_llm_minicpm-2b-dpo-bf16>`
     - chat
     - 4096
     - MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

   * - :ref:`minicpm-2b-dpo-fp16 <models_llm_minicpm-2b-dpo-fp16>`
     - chat
     - 4096
     - MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

   * - :ref:`minicpm-2b-dpo-fp32 <models_llm_minicpm-2b-dpo-fp32>`
     - chat
     - 4096
     - MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

   * - :ref:`minicpm-2b-sft-bf16 <models_llm_minicpm-2b-sft-bf16>`
     - chat
     - 4096
     - MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

   * - :ref:`minicpm-2b-sft-fp32 <models_llm_minicpm-2b-sft-fp32>`
     - chat
     - 4096
     - MiniCPM is an End-Size LLM developed by ModelBest Inc. and TsinghuaNLP, with only 2.4B parameters excluding embeddings.

   * - :ref:`minicpm-v-2.6 <models_llm_minicpm-v-2.6>`
     - chat, vision
     - 32768
     - MiniCPM-V 2.6 is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Qwen2-7B with a total of 8B parameters.

   * - :ref:`minicpm-v-4.5 <models_llm_minicpm-v-4.5>`
     - chat, vision
     - 32768
     - MiniCPM-V 4.5 is an improved version in the MiniCPM-V series with enhanced multimodal capabilities and better performance.

   * - :ref:`minicpm3-4b <models_llm_minicpm3-4b>`
     - chat
     - 32768
     - MiniCPM3-4B is the 3rd generation of MiniCPM series. The overall performance of MiniCPM3-4B surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, being comparable with many recent 7B~9B models.

   * - :ref:`minicpm4 <models_llm_minicpm4>`
     - chat
     - 32768
     - MiniCPM4 series are highly efficient large language models (LLMs) designed explicitly for end-side devices, which achieves this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems.

   * - :ref:`minimax-m2 <models_llm_minimax-m2>`
     - chat, tools, reasoning
     - 196608
     - MiniMax-M2, a Mini model built for Max coding & agentic workflows.

   * - :ref:`mistral-instruct-v0.1 <models_llm_mistral-instruct-v0.1>`
     - chat
     - 8192
     - Mistral-7B-Instruct is a fine-tuned version of the Mistral-7B LLM on public datasets, specializing in chatting.

   * - :ref:`mistral-instruct-v0.2 <models_llm_mistral-instruct-v0.2>`
     - chat
     - 8192
     - The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.

   * - :ref:`mistral-instruct-v0.3 <models_llm_mistral-instruct-v0.3>`
     - chat
     - 32768
     - The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.

   * - :ref:`mistral-large-instruct <models_llm_mistral-large-instruct>`
     - chat
     - 131072
     - Mistral-Large-Instruct-2407 is an advanced dense Large Language Model (LLM) of 123B parameters with state-of-the-art reasoning, knowledge and coding capabilities.

   * - :ref:`mistral-nemo-instruct <models_llm_mistral-nemo-instruct>`
     - chat
     - 1024000
     - The Mistral-Nemo-Instruct-2407 Large Language Model (LLM) is an instruct fine-tuned version of the Mistral-Nemo-Base-2407

   * - :ref:`mistral-v0.1 <models_llm_mistral-v0.1>`
     - generate
     - 8192
     - Mistral-7B is a unmoderated Transformer based LLM claiming to outperform Llama2 on all benchmarks.

   * - :ref:`mixtral-8x22b-instruct-v0.1 <models_llm_mixtral-8x22b-instruct-v0.1>`
     - chat
     - 65536
     - The Mixtral-8x22B-Instruct-v0.1 Large Language Model (LLM) is an instruct fine-tuned version of the Mixtral-8x22B-v0.1, specializing in chatting.

   * - :ref:`mixtral-instruct-v0.1 <models_llm_mixtral-instruct-v0.1>`
     - chat
     - 32768
     - Mistral-8x7B-Instruct is a fine-tuned version of the Mistral-8x7B LLM, specializing in chatting.

   * - :ref:`mixtral-v0.1 <models_llm_mixtral-v0.1>`
     - generate
     - 32768
     - The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts.

   * - :ref:`moonlight-16b-a3b-instruct <models_llm_moonlight-16b-a3b-instruct>`
     - chat
     - 8192
     - Kimi Muon is Scalable for LLM Training

   * - :ref:`openhermes-2.5 <models_llm_openhermes-2.5>`
     - chat
     - 8192
     - Openhermes 2.5 is a fine-tuned version of Mistral-7B-v0.1 on primarily GPT-4 generated data.

   * - :ref:`opt <models_llm_opt>`
     - generate
     - 2048
     - Opt is an open-source, decoder-only, Transformer based LLM that was designed to replicate GPT-3.

   * - :ref:`orion-chat <models_llm_orion-chat>`
     - chat
     - 4096
     - Orion-14B series models are open-source multilingual large language models trained from scratch by OrionStarAI.

   * - :ref:`ovis2 <models_llm_ovis2>`
     - chat, vision
     - 32768
     - Ovis (Open VISion) is a novel Multimodal Large Language Model (MLLM) architecture, designed to structurally align visual and textual embeddings.

   * - :ref:`phi-2 <models_llm_phi-2>`
     - generate
     - 2048
     - Phi-2 is a 2.7B Transformer based LLM used for research on model safety, trained with data similar to Phi-1.5 but augmented with synthetic texts and curated websites.

   * - :ref:`phi-3-mini-128k-instruct <models_llm_phi-3-mini-128k-instruct>`
     - chat
     - 128000
     - The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets.

   * - :ref:`phi-3-mini-4k-instruct <models_llm_phi-3-mini-4k-instruct>`
     - chat
     - 4096
     - The Phi-3-Mini-4k-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets.

   * - :ref:`qvq-72b-preview <models_llm_qvq-72b-preview>`
     - chat, vision
     - 32768
     - QVQ-72B-Preview is an experimental research model developed by the Qwen team, focusing on enhancing visual reasoning capabilities.

   * - :ref:`qwen-chat <models_llm_qwen-chat>`
     - chat
     - 32768
     - Qwen-chat is a fine-tuned version of the Qwen LLM trained with alignment techniques, specializing in chatting.

   * - :ref:`qwen1.5-chat <models_llm_qwen1.5-chat>`
     - chat, tools
     - 32768
     - Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data.

   * - :ref:`qwen1.5-moe-chat <models_llm_qwen1.5-moe-chat>`
     - chat, tools
     - 32768
     - Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data.

   * - :ref:`qwen2-audio <models_llm_qwen2-audio>`
     - generate, audio
     - 32768
     - Qwen2-Audio: A large-scale audio-language model which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions.

   * - :ref:`qwen2-audio-instruct <models_llm_qwen2-audio-instruct>`
     - chat, audio
     - 32768
     - Qwen2-Audio: A large-scale audio-language model which is capable of accepting various audio signal inputs and performing audio analysis or direct textual responses with regard to speech instructions.

   * - :ref:`qwen2-instruct <models_llm_qwen2-instruct>`
     - chat, tools
     - 32768
     - Qwen2 is the new series of Qwen large language models

   * - :ref:`qwen2-moe-instruct <models_llm_qwen2-moe-instruct>`
     - chat, tools
     - 32768
     - Qwen2 is the new series of Qwen large language models. 

   * - :ref:`qwen2-vl-instruct <models_llm_qwen2-vl-instruct>`
     - chat, vision
     - 32768
     - Qwen2-VL: To See the World More Clearly.Qwen2-VL is the latest version of the vision language models in the Qwen model familities.

   * - :ref:`qwen2.5 <models_llm_qwen2.5>`
     - generate
     - 32768
     - Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters.

   * - :ref:`qwen2.5-coder <models_llm_qwen2.5-coder>`
     - generate
     - 32768
     - Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

   * - :ref:`qwen2.5-coder-instruct <models_llm_qwen2.5-coder-instruct>`
     - chat, tools
     - 32768
     - Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen).

   * - :ref:`qwen2.5-instruct <models_llm_qwen2.5-instruct>`
     - chat, tools
     - 32768
     - Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters.

   * - :ref:`qwen2.5-instruct-1m <models_llm_qwen2.5-instruct-1m>`
     - chat
     - 1010000
     - Qwen2.5-1M is the long-context version of the Qwen2.5 series models, supporting a context length of up to 1M tokens.

   * - :ref:`qwen2.5-omni <models_llm_qwen2.5-omni>`
     - chat, vision, audio, omni
     - 32768
     - Qwen2.5-Omni: the new flagship end-to-end multimodal model in the Qwen series.

   * - :ref:`qwen2.5-vl-instruct <models_llm_qwen2.5-vl-instruct>`
     - chat, vision
     - 128000
     - Qwen2.5-VL: Qwen2.5-VL is the latest version of the vision language models in the Qwen model familities.

   * - :ref:`qwen3 <models_llm_qwen3>`
     - chat, reasoning, hybrid, tools
     - 40960
     - Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support.

   * - :ref:`qwen3-coder <models_llm_qwen3-coder>`
     - chat, tools
     - 262144
     - we're announcing Qwen3-Coder, our most agentic code model to date

   * - :ref:`qwen3-instruct <models_llm_qwen3-instruct>`
     - chat, tools
     - 262144
     - We introduce the updated version of the Qwen3-235B-A22B non-thinking mode, named Qwen3-235B-A22B-Instruct-2507

   * - :ref:`qwen3-next-instruct <models_llm_qwen3-next-instruct>`
     - chat, tools
     - 262144
     - Qwen3-Next-80B-A3B is the first installment in the Qwen3-Next series

   * - :ref:`qwen3-next-thinking <models_llm_qwen3-next-thinking>`
     - chat, reasoning, tools
     - 262144
     - Qwen3-Next-80B-A3B is the first installment in the Qwen3-Next series

   * - :ref:`qwen3-omni-instruct <models_llm_qwen3-omni-instruct>`
     - chat, vision, audio, omni, tools
     - 262144
     - Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency.

   * - :ref:`qwen3-omni-thinking <models_llm_qwen3-omni-thinking>`
     - chat, vision, audio, omni, reasoning, tools
     - 262144
     - Qwen3-Omni is the natively end-to-end multilingual omni-modal foundation models. It processes text, images, audio, and video, and delivers real-time streaming responses in both text and natural speech. We introduce several architectural upgrades to improve performance and efficiency.

   * - :ref:`qwen3-thinking <models_llm_qwen3-thinking>`
     - chat, reasoning, tools
     - 262144
     - we have continued to scale the thinking capability of Qwen3-235B-A22B, improving both the quality and depth of reasoning

   * - :ref:`qwen3-vl-instruct <models_llm_qwen3-vl-instruct>`
     - chat, vision, tools
     - 262144
     - Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

   * - :ref:`qwen3-vl-thinking <models_llm_qwen3-vl-thinking>`
     - chat, vision, reasoning, tools
     - 262144
     - Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

   * - :ref:`qwenlong-l1 <models_llm_qwenlong-l1>`
     - chat
     - 32768
     - QwenLong-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning

   * - :ref:`qwq-32b <models_llm_qwq-32b>`
     - chat, reasoning, tools
     - 131072
     - QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models, QwQ, which is capable of thinking and reasoning, can achieve significantly enhanced performance in downstream tasks, especially hard problems. QwQ-32B is the medium-sized reasoning model, which is capable of achieving competitive performance against state-of-the-art reasoning models, e.g., DeepSeek-R1, o1-mini.

   * - :ref:`qwq-32b-preview <models_llm_qwq-32b-preview>`
     - chat
     - 32768
     - QwQ-32B-Preview is an experimental research model developed by the Qwen Team, focused on advancing AI reasoning capabilities.

   * - :ref:`seallm_v2 <models_llm_seallm_v2>`
     - generate
     - 8192
     - We introduce SeaLLM-7B-v2, the state-of-the-art multilingual LLM for Southeast Asian (SEA) languages

   * - :ref:`seallm_v2.5 <models_llm_seallm_v2.5>`
     - generate
     - 8192
     - We introduce SeaLLM-7B-v2.5, the state-of-the-art multilingual LLM for Southeast Asian (SEA) languages

   * - :ref:`seallms-v3 <models_llm_seallms-v3>`
     - chat
     - 32768
     - SeaLLMs - Large Language Models for Southeast Asia

   * - :ref:`seed-oss <models_llm_seed-oss>`
     - chat, reasoning, tools
     - 524288
     - Seed-OSS is a series of open-source large language models developed by ByteDance's Seed Team, designed for powerful long-context, reasoning, agent and general capabilities, and versatile developer-friendly features. Although trained with only 12T tokens, Seed-OSS achieves excellent performance on several popular open benchmarks.

   * - :ref:`skywork <models_llm_skywork>`
     - generate
     - 4096
     - Skywork is a series of large models developed by the Kunlun Group · Skywork team.

   * - :ref:`skywork-math <models_llm_skywork-math>`
     - generate
     - 4096
     - Skywork is a series of large models developed by the Kunlun Group · Skywork team.

   * - :ref:`skywork-or1 <models_llm_skywork-or1>`
     - chat
     - 131072
     - We release the final version of Skywork-OR1 (Open Reasoner 1) series of models, including

   * - :ref:`skywork-or1-preview <models_llm_skywork-or1-preview>`
     - chat
     - 32768
     - The Skywork-OR1 (Open Reasoner 1) model series consists of powerful math and code reasoning models trained using large-scale rule-based reinforcement learning with carefully designed datasets and training recipes.

   * - :ref:`telechat <models_llm_telechat>`
     - chat
     - 8192
     - The TeleChat is a large language model developed and trained by China Telecom Artificial Intelligence Technology Co., LTD. The 7B model base is trained with 1.5 trillion Tokens and 3 trillion Tokens and Chinese high-quality corpus.

   * - :ref:`tiny-llama <models_llm_tiny-llama>`
     - generate
     - 2048
     - The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens.

   * - :ref:`wizardcoder-python-v1.0 <models_llm_wizardcoder-python-v1.0>`
     - chat
     - 100000
     - 

   * - :ref:`wizardmath-v1.0 <models_llm_wizardmath-v1.0>`
     - chat
     - 2048
     - WizardMath is an open-source LLM trained by fine-tuning Llama2 with Evol-Instruct, specializing in math.

   * - :ref:`xiyansql-qwencoder-2504 <models_llm_xiyansql-qwencoder-2504>`
     - chat, tools
     - 32768
     - The XiYanSQL-QwenCoder models, as multi-dialect SQL base models, demonstrating robust SQL generation capabilities.

   * - :ref:`xverse <models_llm_xverse>`
     - generate
     - 2048
     - XVERSE is a multilingual large language model, independently developed by Shenzhen Yuanxiang Technology.

   * - :ref:`xverse-chat <models_llm_xverse-chat>`
     - chat
     - 2048
     - XVERSEB-Chat is the aligned version of model XVERSE.

   * - :ref:`yi <models_llm_yi>`
     - generate
     - 4096
     - The Yi series models are large language models trained from scratch by developers at 01.AI.

   * - :ref:`yi-1.5 <models_llm_yi-1.5>`
     - generate
     - 4096
     - Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

   * - :ref:`yi-1.5-chat <models_llm_yi-1.5-chat>`
     - chat
     - 4096
     - Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

   * - :ref:`yi-1.5-chat-16k <models_llm_yi-1.5-chat-16k>`
     - chat
     - 16384
     - Yi-1.5 is an upgraded version of Yi. It is continuously pre-trained on Yi with a high-quality corpus of 500B tokens and fine-tuned on 3M diverse fine-tuning samples.

   * - :ref:`yi-200k <models_llm_yi-200k>`
     - generate
     - 262144
     - The Yi series models are large language models trained from scratch by developers at 01.AI.

   * - :ref:`yi-chat <models_llm_yi-chat>`
     - chat
     - 4096
     - The Yi series models are large language models trained from scratch by developers at 01.AI.


.. toctree::
   :maxdepth: 3

  
   baichuan-2
  
   baichuan-2-chat
  
   baichuan-m2
  
   code-llama
  
   code-llama-instruct
  
   code-llama-python
  
   codegeex4
  
   codeqwen1.5
  
   codeqwen1.5-chat
  
   codeshell
  
   codeshell-chat
  
   codestral-v0.1
  
   cogagent
  
   deepseek
  
   deepseek-chat
  
   deepseek-coder
  
   deepseek-coder-instruct
  
   deepseek-prover-v2
  
   deepseek-r1
  
   deepseek-r1-0528
  
   deepseek-r1-0528-qwen3
  
   deepseek-r1-distill-llama
  
   deepseek-r1-distill-qwen
  
   deepseek-v2-chat
  
   deepseek-v2-chat-0628
  
   deepseek-v2.5
  
   deepseek-v3
  
   deepseek-v3-0324
  
   deepseek-v3.1
  
   deepseek-v3.2
  
   deepseek-v3.2-exp
  
   deepseek-vl2
  
   dianjin-r1
  
   ernie4.5
  
   fin-r1
  
   gemma-3-1b-it
  
   gemma-3-it
  
   glm-4.1v-thinking
  
   glm-4.5
  
   glm-4.5v
  
   glm-4v
  
   glm-edge-chat
  
   glm4-0414
  
   glm4-chat
  
   glm4-chat-1m
  
   gorilla-openfunctions-v2
  
   gpt-2
  
   gpt-oss
  
   huatuogpt-o1-llama-3.1
  
   huatuogpt-o1-qwen2.5
  
   internlm3-instruct
  
   internvl3
  
   kat-v1
  
   llama-2
  
   llama-2-chat
  
   llama-3
  
   llama-3-instruct
  
   llama-3.1
  
   llama-3.1-instruct
  
   llama-3.2-vision
  
   llama-3.2-vision-instruct
  
   llama-3.3-instruct
  
   marco-o1
  
   minicpm-2b-dpo-bf16
  
   minicpm-2b-dpo-fp16
  
   minicpm-2b-dpo-fp32
  
   minicpm-2b-sft-bf16
  
   minicpm-2b-sft-fp32
  
   minicpm-v-2.6
  
   minicpm-v-4.5
  
   minicpm3-4b
  
   minicpm4
  
   minimax-m2
  
   mistral-instruct-v0.1
  
   mistral-instruct-v0.2
  
   mistral-instruct-v0.3
  
   mistral-large-instruct
  
   mistral-nemo-instruct
  
   mistral-v0.1
  
   mixtral-8x22b-instruct-v0.1
  
   mixtral-instruct-v0.1
  
   mixtral-v0.1
  
   moonlight-16b-a3b-instruct
  
   openhermes-2.5
  
   opt
  
   orion-chat
  
   ovis2
  
   phi-2
  
   phi-3-mini-128k-instruct
  
   phi-3-mini-4k-instruct
  
   qvq-72b-preview
  
   qwen-chat
  
   qwen1.5-chat
  
   qwen1.5-moe-chat
  
   qwen2-audio
  
   qwen2-audio-instruct
  
   qwen2-instruct
  
   qwen2-moe-instruct
  
   qwen2-vl-instruct

   mineru2.5-2509-1.2b
  
   qwen2.5
  
   qwen2.5-coder
  
   qwen2.5-coder-instruct
  
   qwen2.5-instruct
  
   qwen2.5-instruct-1m
  
   qwen2.5-omni
  
   qwen2.5-vl-instruct
  
   qwen3
  
   qwen3-coder
  
   qwen3-instruct
  
   qwen3-next-instruct
  
   qwen3-next-thinking
  
   qwen3-omni-instruct
  
   qwen3-omni-thinking
  
   qwen3-thinking
  
   qwen3-vl-instruct
  
   qwen3-vl-thinking
  
   qwenlong-l1
  
   qwq-32b
  
   qwq-32b-preview
  
   seallm_v2
  
   seallm_v2.5
  
   seallms-v3
  
   seed-oss
  
   skywork
  
   skywork-math
  
   skywork-or1
  
   skywork-or1-preview
  
   telechat
  
   tiny-llama
  
   wizardcoder-python-v1.0
  
   wizardmath-v1.0
  
   xiyansql-qwencoder-2504
  
   xverse
  
   xverse-chat
  
   yi
  
   yi-1.5
  
   yi-1.5-chat
  
   yi-1.5-chat-16k
  
   yi-200k
  
   yi-chat
  

