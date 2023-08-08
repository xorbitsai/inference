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
with popular third-party libraries like LangChain and LlamaIndex. (Coming soon)


🔥 Hot topics
-------------

Framework Enhancements
~~~~~~~~~~~~~~~~~~~~~~
- LoRA support: `#271 <https://github.com/xorbitsai/inference/issues/271>`_
- Multi-GPU support for PyTorch models: `#226 <https://github.com/xorbitsai/inference/issues/226>`_
- Xinference dashboard: `#93 <https://github.com/xorbitsai/inference/issues/93>`_

New Models
~~~~~~~~~~
- Built-in support for `Starcoder` in GGML: `#289 <https://github.com/xorbitsai/inference/pull/289>`_
- Built-in support for `MusicGen <https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md>`_: `#313 <https://github.com/xorbitsai/inference/issues/313>`_
- Built-in support for `SD-XL <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`_: `318 <https://github.com/xorbitsai/inference/issues/318>`_

Tools
~~~~~
- LlamaIndex plugin: `7151 <https://github.com/jerryjliu/llama_index/pull/7151>`_


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
