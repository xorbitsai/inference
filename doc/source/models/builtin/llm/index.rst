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

   * - :ref:`codeshell <models_llm_codeshell>`
     - generate
     - 8194
     - CodeShell is a multi-language code LLM developed by the Knowledge Computing Lab of Peking University. 

   * - :ref:`codeshell-chat <models_llm_codeshell-chat>`
     - chat
     - 8194
     - CodeShell is a multi-language code LLM developed by the Knowledge Computing Lab of Peking University.

   * - :ref:`deepseek-chat <models_llm_deepseek-chat>`
     - chat
     - 4096
     - DeepSeek LLM is an advanced language model comprising 67 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese.

   * - :ref:`deepseek-coder-instruct <models_llm_deepseek-coder-instruct>`
     - chat
     - 4096
     - deepseek-coder-instruct is a model initialized from deepseek-coder-base and fine-tuned on 2B tokens of instruction data.

   * - :ref:`falcon <models_llm_falcon>`
     - generate
     - 2048
     - Falcon is an open-source Transformer based LLM trained on the RefinedWeb dataset.

   * - :ref:`falcon-instruct <models_llm_falcon-instruct>`
     - chat
     - 2048
     - Falcon-instruct is a fine-tuned version of the Falcon LLM, specializing in chatting.

   * - :ref:`gemma-it <models_llm_gemma-it>`
     - chat
     - 8192
     - Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models.

   * - :ref:`glaive-coder <models_llm_glaive-coder>`
     - chat
     - 16384
     - A code model trained on a dataset of ~140k programming related problems and solutions generated from Glaive’s synthetic data generation platform.

   * - :ref:`gorilla-openfunctions-v1 <models_llm_gorilla-openfunctions-v1>`
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

   * - :ref:`llama-2 <models_llm_llama-2>`
     - generate
     - 4096
     - Llama-2 is the second generation of Llama, open-source and trained on a larger amount of data.

   * - :ref:`llama-2-chat <models_llm_llama-2-chat>`
     - chat
     - 4096
     - Llama-2-Chat is a fine-tuned version of the Llama-2 LLM, specializing in chatting.

   * - :ref:`mistral-instruct-v0.1 <models_llm_mistral-instruct-v0.1>`
     - chat
     - 8192
     - Mistral-7B-Instruct is a fine-tuned version of the Mistral-7B LLM on public datasets, specializing in chatting.

   * - :ref:`mistral-instruct-v0.2 <models_llm_mistral-instruct-v0.2>`
     - chat
     - 8192
     - The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of Mistral-7B-Instruct-v0.1.

   * - :ref:`mistral-v0.1 <models_llm_mistral-v0.1>`
     - generate
     - 8192
     - Mistral-7B is a unmoderated Transformer based LLM claiming to outperform Llama2 on all benchmarks.

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

   * - :ref:`yi-200k <models_llm_yi-200k>`
     - generate
     - 204800
     - The Yi series models are large language models trained from scratch by developers at 01.AI.

   * - :ref:`yi-chat <models_llm_yi-chat>`
     - chat
     - 204800
     - The Yi series models are large language models trained from scratch by developers at 01.AI.

   * - :ref:`yi-vl-chat <models_llm_yi-vl-chat>`
     - chat, vision
     - 204800
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

  
   baichuan
  
   baichuan-2
  
   baichuan-2-chat
  
   baichuan-chat
  
   chatglm
  
   chatglm2
  
   chatglm2-32k
  
   chatglm3
  
   chatglm3-32k
  
   code-llama
  
   code-llama-instruct
  
   code-llama-python
  
   codeshell
  
   codeshell-chat
  
   deepseek-chat
  
   deepseek-coder-instruct
  
   falcon
  
   falcon-instruct
  
   gemma-it
  
   glaive-coder
  
   gorilla-openfunctions-v1
  
   gpt-2
  
   internlm-20b
  
   internlm-7b
  
   internlm-chat-20b
  
   internlm-chat-7b
  
   internlm2-chat
  
   llama-2
  
   llama-2-chat
  
   mistral-instruct-v0.1
  
   mistral-instruct-v0.2
  
   mistral-v0.1
  
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
  
   qwen-chat
  
   qwen-vl-chat
  
   qwen1.5-chat
  
   skywork
  
   skywork-math
  
   starchat-beta
  
   starcoder
  
   starcoderplus
  
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
  
   yi-200k
  
   yi-chat
  
   yi-vl-chat
  
   zephyr-7b-alpha
  
   zephyr-7b-beta
  

