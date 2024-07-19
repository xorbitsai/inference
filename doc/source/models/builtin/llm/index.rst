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


   * - :ref:`aquila2 <models_llm_aquila2>`
     - generate
     - 2048
     - Aquila2 series models are the base language models

   * - :ref:`aquila2-chat <models_llm_aquila2-chat>`
     - chat
     - 2048
     - Aquila2-chat series models are the chat models

   * - :ref:`aquila2-chat-16k <models_llm_aquila2-chat-16k>`
     - chat
     - 16384
     - AquilaChat2-16k series models are the long-text chat models

   * - :ref:`baichuan <models_llm_baichuan>`
     - generate
     - 4096
     - Baichuan is an open-source Transformer based LLM that is trained on both Chinese and English data.

   * - :ref:`baichuan-2 <models_llm_baichuan-2>`
     - generate
     - 4096
     - Baichuan2 is an open-source Transformer based LLM that is trained on both Chinese and English data.

   * - :ref:`baichuan-2-chat <models_llm_baichuan-2-chat>`
     - chat
     - 4096
     - Baichuan2-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

   * - :ref:`baichuan-chat <models_llm_baichuan-chat>`
     - chat
     - 4096
     - Baichuan-chat is a fine-tuned version of the Baichuan LLM, specializing in chatting.

   * - :ref:`c4ai-command-r-v01 <models_llm_c4ai-command-r-v01>`
     - chat
     - 131072
     - C4AI Command-R(+) is a research release of a 35 and 104 billion parameter highly performant generative model.

   * - :ref:`c4ai-command-r-v01-4bit <models_llm_c4ai-command-r-v01-4bit>`
     - generate
     - 131072
     - This model is 4bit quantized version of C4AI Command-R using bitsandbytes.

   * - :ref:`chatglm <models_llm_chatglm>`
     - chat
     - 2048
     - ChatGLM is an open-source General Language Model (GLM) based LLM trained on both Chinese and English data.

   * - :ref:`chatglm2 <models_llm_chatglm2>`
     - chat
     - 8192
     - ChatGLM2 is the second generation of ChatGLM, still open-source and trained on Chinese and English data.

   * - :ref:`chatglm2-32k <models_llm_chatglm2-32k>`
     - chat
     - 32768
     - ChatGLM2-32k is a special version of ChatGLM2, with a context window of 32k tokens instead of 8k.

   * - :ref:`chatglm3 <models_llm_chatglm3>`
     - chat, tools
     - 8192
     - ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

   * - :ref:`chatglm3-128k <models_llm_chatglm3-128k>`
     - chat
     - 131072
     - ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

   * - :ref:`chatglm3-32k <models_llm_chatglm3-32k>`
     - chat
     - 32768
     - ChatGLM3 is the third generation of ChatGLM, still open-source and trained on Chinese and English data.

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

   * - :ref:`cogvlm2 <models_llm_cogvlm2>`
     - chat, vision
     - 8192
     - CogVLM2 have achieved good results in many lists compared to the previous generation of CogVLM open source models. Its excellent performance can compete with some non-open source models.

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

   * - :ref:`deepseek-vl-chat <models_llm_deepseek-vl-chat>`
     - chat, vision
     - 4096
     - DeepSeek-VL possesses general multimodal understanding capabilities, capable of processing logical diagrams, web pages, formula recognition, scientific literature, natural images, and embodied intelligence in complex scenarios.

   * - :ref:`falcon <models_llm_falcon>`
     - generate
     - 2048
     - Falcon is an open-source Transformer based LLM trained on the RefinedWeb dataset.

   * - :ref:`falcon-instruct <models_llm_falcon-instruct>`
     - chat
     - 2048
     - Falcon-instruct is a fine-tuned version of the Falcon LLM, specializing in chatting.

   * - :ref:`gemma-2-it <models_llm_gemma-2-it>`
     - chat
     - 8192
     - Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

   * - :ref:`gemma-it <models_llm_gemma-it>`
     - chat
     - 8192
     - Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

   * - :ref:`glaive-coder <models_llm_glaive-coder>`
     - chat
     - 16384
     - A code model trained on a dataset of ~140k programming related problems and solutions generated from Glaive’s synthetic data generation platform.

   * - :ref:`glm-4v <models_llm_glm-4v>`
     - chat, vision
     - 8192
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`glm4-chat <models_llm_glm4-chat>`
     - chat, tools
     - 131072
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`glm4-chat-1m <models_llm_glm4-chat-1m>`
     - chat, tools
     - 1048576
     - GLM4 is the open source version of the latest generation of pre-trained models in the GLM-4 series launched by Zhipu AI.

   * - :ref:`gorilla-openfunctions-v1 <models_llm_gorilla-openfunctions-v1>`
     - chat
     - 4096
     - OpenFunctions is designed to extend Large Language Model (LLM) Chat Completion feature to formulate executable APIs call given natural language instructions and API context.

   * - :ref:`gorilla-openfunctions-v2 <models_llm_gorilla-openfunctions-v2>`
     - chat
     - 4096
     - OpenFunctions is designed to extend Large Language Model (LLM) Chat Completion feature to formulate executable APIs call given natural language instructions and API context.

   * - :ref:`gpt-2 <models_llm_gpt-2>`
     - generate
     - 1024
     - GPT-2 is a Transformer-based LLM that is trained on WebTest, a 40 GB dataset of Reddit posts with 3+ upvotes.

   * - :ref:`internlm-20b <models_llm_internlm-20b>`
     - generate
     - 16384
     - Pre-trained on over 2.3T Tokens containing high-quality English, Chinese, and code data.

   * - :ref:`internlm-7b <models_llm_internlm-7b>`
     - generate
     - 8192
     - InternLM is a Transformer-based LLM that is trained on both Chinese and English data, focusing on practical scenarios.

   * - :ref:`internlm-chat-20b <models_llm_internlm-chat-20b>`
     - chat
     - 16384
     - Pre-trained on over 2.3T Tokens containing high-quality English, Chinese, and code data. The Chat version has undergone SFT and RLHF training.

   * - :ref:`internlm-chat-7b <models_llm_internlm-chat-7b>`
     - chat
     - 4096
     - Internlm-chat is a fine-tuned version of the Internlm LLM, specializing in chatting.

   * - :ref:`internlm2-chat <models_llm_internlm2-chat>`
     - chat
     - 204800
     - The second generation of the InternLM model, InternLM2.

   * - :ref:`internvl-chat <models_llm_internvl-chat>`
     - chat, vision
     - 32768
     - InternVL 1.5 is an open-source multimodal large language model (MLLM) to bridge the capability gap between open-source and proprietary commercial models in multimodal understanding. 

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

   * - :ref:`minicpm-llama3-v-2_5 <models_llm_minicpm-llama3-v-2_5>`
     - chat, vision
     - 2048
     - MiniCPM-Llama3-V 2.5 is the latest model in the MiniCPM-V series. The model is built on SigLip-400M and Llama3-8B-Instruct with a total of 8B parameters.

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

   * - :ref:`omnilmm <models_llm_omnilmm>`
     - chat, vision
     - 2048
     - OmniLMM is a family of open-source large multimodal models (LMMs) adept at vision & language modeling.

   * - :ref:`openbuddy <models_llm_openbuddy>`
     - chat
     - 2048
     - OpenBuddy is a powerful open multilingual chatbot model aimed at global users.

   * - :ref:`openhermes-2.5 <models_llm_openhermes-2.5>`
     - chat
     - 8192
     - Openhermes 2.5 is a fine-tuned version of Mistral-7B-v0.1 on primarily GPT-4 generated data.

   * - :ref:`opt <models_llm_opt>`
     - generate
     - 2048
     - Opt is an open-source, decoder-only, Transformer based LLM that was designed to replicate GPT-3.

   * - :ref:`orca <models_llm_orca>`
     - chat
     - 2048
     - Orca is an LLM trained by fine-tuning LLaMA on explanation traces obtained from GPT-4.

   * - :ref:`orion-chat <models_llm_orion-chat>`
     - chat
     - 4096
     - Orion-14B series models are open-source multilingual large language models trained from scratch by OrionStarAI.

   * - :ref:`orion-chat-rag <models_llm_orion-chat-rag>`
     - chat
     - 4096
     - Orion-14B series models are open-source multilingual large language models trained from scratch by OrionStarAI.

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

   * - :ref:`platypus2-70b-instruct <models_llm_platypus2-70b-instruct>`
     - generate
     - 4096
     - Platypus-70B-instruct is a merge of garage-bAInd/Platypus2-70B and upstage/Llama-2-70b-instruct-v2.

   * - :ref:`qwen-chat <models_llm_qwen-chat>`
     - chat, tools
     - 32768
     - Qwen-chat is a fine-tuned version of the Qwen LLM trained with alignment techniques, specializing in chatting.

   * - :ref:`qwen-vl-chat <models_llm_qwen-vl-chat>`
     - chat, vision
     - 4096
     - Qwen-VL-Chat supports more flexible interaction, such as multiple image inputs, multi-round question answering, and creative capabilities.

   * - :ref:`qwen1.5-chat <models_llm_qwen1.5-chat>`
     - chat, tools
     - 32768
     - Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of data.

   * - :ref:`qwen1.5-moe-chat <models_llm_qwen1.5-moe-chat>`
     - chat, tools
     - 32768
     - Qwen1.5-MoE is a transformer-based MoE decoder-only language model pretrained on a large amount of data.

   * - :ref:`qwen2-instruct <models_llm_qwen2-instruct>`
     - chat, tools
     - 32768
     - Qwen2 is the new series of Qwen large language models

   * - :ref:`qwen2-moe-instruct <models_llm_qwen2-moe-instruct>`
     - chat, tools
     - 32768
     - Qwen2 is the new series of Qwen large language models. 

   * - :ref:`seallm_v2 <models_llm_seallm_v2>`
     - generate
     - 8192
     - We introduce SeaLLM-7B-v2, the state-of-the-art multilingual LLM for Southeast Asian (SEA) languages

   * - :ref:`seallm_v2.5 <models_llm_seallm_v2.5>`
     - generate
     - 8192
     - We introduce SeaLLM-7B-v2.5, the state-of-the-art multilingual LLM for Southeast Asian (SEA) languages

   * - :ref:`skywork <models_llm_skywork>`
     - generate
     - 4096
     - Skywork is a series of large models developed by the Kunlun Group · Skywork team.

   * - :ref:`skywork-math <models_llm_skywork-math>`
     - generate
     - 4096
     - Skywork is a series of large models developed by the Kunlun Group · Skywork team.

   * - :ref:`starchat-beta <models_llm_starchat-beta>`
     - chat
     - 8192
     - Starchat-beta is a fine-tuned version of the Starcoderplus LLM, specializing in coding assistance.

   * - :ref:`starcoder <models_llm_starcoder>`
     - generate
     - 8192
     - Starcoder is an open-source Transformer based LLM that is trained on permissively licensed data from GitHub.

   * - :ref:`starcoderplus <models_llm_starcoderplus>`
     - generate
     - 8192
     - Starcoderplus is an open-source LLM trained by fine-tuning Starcoder on RedefinedWeb and StarCoderData datasets.

   * - :ref:`starling-lm <models_llm_starling-lm>`
     - chat
     - 4096
     - We introduce Starling-7B, an open large language model (LLM) trained by Reinforcement Learning from AI Feedback (RLAIF). The model harnesses the power of our new GPT-4 labeled ranking dataset

   * - :ref:`telechat <models_llm_telechat>`
     - chat
     - 8192
     - The TeleChat is a large language model developed and trained by China Telecom Artificial Intelligence Technology Co., LTD. The 7B model base is trained with 1.5 trillion Tokens and 3 trillion Tokens and Chinese high-quality corpus.

   * - :ref:`tiny-llama <models_llm_tiny-llama>`
     - generate
     - 2048
     - The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens.

   * - :ref:`vicuna-v1.3 <models_llm_vicuna-v1.3>`
     - chat
     - 2048
     - Vicuna is an open-source LLM trained by fine-tuning LLaMA on data collected from ShareGPT.

   * - :ref:`vicuna-v1.5 <models_llm_vicuna-v1.5>`
     - chat
     - 4096
     - Vicuna is an open-source LLM trained by fine-tuning LLaMA on data collected from ShareGPT.

   * - :ref:`vicuna-v1.5-16k <models_llm_vicuna-v1.5-16k>`
     - chat
     - 16384
     - Vicuna-v1.5-16k is a special version of Vicuna-v1.5, with a context window of 16k tokens instead of 4k.

   * - :ref:`wizardcoder-python-v1.0 <models_llm_wizardcoder-python-v1.0>`
     - chat
     - 100000
     - 

   * - :ref:`wizardlm-v1.0 <models_llm_wizardlm-v1.0>`
     - chat
     - 2048
     - WizardLM is an open-source LLM trained by fine-tuning LLaMA with Evol-Instruct.

   * - :ref:`wizardmath-v1.0 <models_llm_wizardmath-v1.0>`
     - chat
     - 2048
     - WizardMath is an open-source LLM trained by fine-tuning Llama2 with Evol-Instruct, specializing in math.

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

   * - :ref:`yi-vl-chat <models_llm_yi-vl-chat>`
     - chat, vision
     - 4096
     - Yi Vision Language (Yi-VL) model is the open-source, multimodal version of the Yi Large Language Model (LLM) series, enabling content comprehension, recognition, and multi-round conversations about images.

   * - :ref:`zephyr-7b-alpha <models_llm_zephyr-7b-alpha>`
     - chat
     - 8192
     - Zephyr-7B-α is the first model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1.

   * - :ref:`zephyr-7b-beta <models_llm_zephyr-7b-beta>`
     - chat
     - 8192
     - Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1


.. toctree::
   :maxdepth: 3

  
   aquila2
  
   aquila2-chat
  
   aquila2-chat-16k
  
   baichuan
  
   baichuan-2
  
   baichuan-2-chat
  
   baichuan-chat
  
   c4ai-command-r-v01
  
   c4ai-command-r-v01-4bit
  
   chatglm
  
   chatglm2
  
   chatglm2-32k
  
   chatglm3
  
   chatglm3-128k
  
   chatglm3-32k
  
   code-llama
  
   code-llama-instruct
  
   code-llama-python
  
   codegeex4
  
   codeqwen1.5
  
   codeqwen1.5-chat
  
   codeshell
  
   codeshell-chat
  
   codestral-v0.1
  
   cogvlm2
  
   deepseek
  
   deepseek-chat
  
   deepseek-coder
  
   deepseek-coder-instruct
  
   deepseek-vl-chat
  
   falcon
  
   falcon-instruct
  
   gemma-2-it
  
   gemma-it
  
   glaive-coder
  
   glm-4v
  
   glm4-chat
  
   glm4-chat-1m
  
   gorilla-openfunctions-v1
  
   gorilla-openfunctions-v2
  
   gpt-2
  
   internlm-20b
  
   internlm-7b
  
   internlm-chat-20b
  
   internlm-chat-7b
  
   internlm2-chat
  
   internvl-chat
  
   llama-2
  
   llama-2-chat
  
   llama-3
  
   llama-3-instruct
  
   minicpm-2b-dpo-bf16
  
   minicpm-2b-dpo-fp16
  
   minicpm-2b-dpo-fp32
  
   minicpm-2b-sft-bf16
  
   minicpm-2b-sft-fp32
  
   minicpm-llama3-v-2_5
  
   mistral-instruct-v0.1
  
   mistral-instruct-v0.2
  
   mistral-instruct-v0.3
  
   mistral-v0.1
  
   mixtral-8x22b-instruct-v0.1
  
   mixtral-instruct-v0.1
  
   mixtral-v0.1
  
   omnilmm
  
   openbuddy
  
   openhermes-2.5
  
   opt
  
   orca
  
   orion-chat
  
   orion-chat-rag
  
   phi-2
  
   phi-3-mini-128k-instruct
  
   phi-3-mini-4k-instruct
  
   platypus2-70b-instruct
  
   qwen-chat
  
   qwen-vl-chat
  
   qwen1.5-chat
  
   qwen1.5-moe-chat
  
   qwen2-instruct
  
   qwen2-moe-instruct
  
   seallm_v2
  
   seallm_v2.5
  
   skywork
  
   skywork-math
  
   starchat-beta
  
   starcoder
  
   starcoderplus
  
   starling-lm
  
   telechat
  
   tiny-llama
  
   vicuna-v1.3
  
   vicuna-v1.5
  
   vicuna-v1.5-16k
  
   wizardcoder-python-v1.0
  
   wizardlm-v1.0
  
   wizardmath-v1.0
  
   xverse
  
   xverse-chat
  
   yi
  
   yi-1.5
  
   yi-1.5-chat
  
   yi-1.5-chat-16k
  
   yi-200k
  
   yi-chat
  
   yi-vl-chat
  
   zephyr-7b-alpha
  
   zephyr-7b-beta
  

