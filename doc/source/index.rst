.. _index:

Xorbits Inference: Model Serving Made Easy🤖
""""""""""""""""""""""""""""""""""""""""""""

Xorbits Inference(Xinference) is a powerful and versatile library designed to serve language,
speech recognition, and multimodal models. With Xorbits Inference, you can effortlessly deploy
and serve your or state-of-the-art built-in models using just a single command. Whether you are a
researcher, developer, or data scientist, Xorbits Inference empowers you to unleash the full
potential of cutting-edge AI models.


Key Features
------------

🌟 **Model Serving Made Easy**: Simplify the process of serving large language, speech
recognition, and multimodal models. You can set up and deploy your models
for experimentation and production with a single command.

⚡️ **State-of-the-Art Models**: Experiment with cutting-edge built-in models using a single
command. Inference provides access to state-of-the-art open-source models!

🖥 **Heterogeneous Hardware Utilization**: Make the most of your hardware resources with
`ggml <https://github.com/ggerganov/ggml>`_. Xorbits Inference intelligently utilizes heterogeneous
hardware, including GPUs and CPUs, to accelerate your model inference tasks.

⚙️ **Flexible API and Interfaces**: Offer multiple interfaces for interacting
with your models, supporting RPC, RESTful API(compatible with OpenAI API), CLI and WebUI
for seamless management and monitoring.

🌐 **Distributed Deployment**: Excel in distributed deployment scenarios,
allowing the seamless distribution of model inference across multiple devices or machines.

🔌 **Built-in Integration with Third-Party Libraries**: Xorbits Inference seamlessly integrates
with popular third-party libraries like `LangChain <https://python.langchain.com/docs/integrations/providers/xinference>`_
, `LlamaIndex <https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window>`_
, `Dify <https://docs.dify.ai/advanced/model-configuration/xinference>`_
, and `Chatbox <https://chatboxai.app/>`_.


🔥 Hot Topics
-------------

Framework Enhancements
~~~~~~~~~~~~~~~~~~~~~~
- Auto recover: `#694 <https://github.com/xorbitsai/inference/pull/694>`_
- Function calling API: `#701 <https://github.com/xorbitsai/inference/pull/701>`_ , here's example: https://github.com/xorbitsai/inference/blob/main/examples/FunctionCall.ipynb
- Support rerank model: `#672 <https://github.com/xorbitsai/inference/pull/672>`_
- Speculative decoding: `#509 <https://github.com/xorbitsai/inference/pull/509>`_
- Support grammar-based sampling for ggml models: `#525 <https://github.com/xorbitsai/inference/pull/525>`_
- Incorporate vLLM: `#445 <https://github.com/xorbitsai/inference/pull/445>`_


New Models
~~~~~~~~~~
- Built-in support for `Yi <https://huggingface.co/01-ai>`_: `#629 <https://github.com/xorbitsai/inference/pull/629>`_
- Built-in support for `zephyr-7b-alpha <https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha>`_ and `zephyr-7b-beta <https://huggingface.co/HuggingFaceH4/zephyr-7b-beta>`_: `#597 <https://github.com/xorbitsai/inference/pull/597>`_
- Built-in support for `chatglm3 <https://huggingface.co/THUDM/chatglm3-6b): [#587](https://github.com/xorbitsai/inference/pull/587>`_
- Built-in support for `mistral-v0.1 <https://huggingface.co/mistralai/Mistral-7B-v0.1>`_ and `mistral-instruct-v0.1 <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1>`_: `#510 <https://github.com/xorbitsai/inference/pull/510>`_


Integrations
~~~~~~~~~~~~
- `Dify <https://docs.dify.ai/advanced/model-configuration/xinference>`_: an LLMOps platform that enables developers (and even non-developers) to quickly build useful applications based on large language models, ensuring they are visual, operable, and improvable.
- `Chatbox <https://chatboxai.app/>`_: a desktop client for multiple cutting-edge LLM models, available on Windows, Mac and Linux.


License
-------
`Apache 2 <https://github.com/xorbitsai/inference/blob/main/LICENSE>`_


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   models/index
   user_guide/index
   examples/index
   reference/index
