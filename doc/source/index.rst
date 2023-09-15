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
- Embedding model support: `#418 <https://github.com/xorbitsai/inference/pull/418>`_
- LoRA support: `#271 <https://github.com/xorbitsai/inference/issues/271>`_
- Multi-GPU support for PyTorch models: `#226 <https://github.com/xorbitsai/inference/issues/226>`_
- Xinference dashboard: `#93 <https://github.com/xorbitsai/inference/issues/93>`_

New Models
~~~~~~~~~~
- Built-in support for `CodeLLama <https://github.com/facebookresearch/codellama>`_: `#414 <https://github.com/xorbitsai/inference/pull/414>`_ `#402 <https://github.com/xorbitsai/inference/pull/402>`_


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

   installation
   getting_started
   models/index
   user_guide/index
   examples/index
   reference/index
