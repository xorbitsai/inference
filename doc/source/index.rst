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

Builtin models
--------------

To view the builtin models, run the following command::

   xinference list --all


+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| Name              | Type             | Language  | Format  | Size (in billions) | Quantization                            |
+===================+==================+===========+=========+====================+=========================================+
| baichuan          | Foundation Model | en, zh    | ggmlv3  | 7                  | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| chatglm           | SFT Model        | en, zh    | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| chatglm2          | SFT Model        | en, zh    | ggmlv3  | 6                  | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| wizardlm-v1.0     | SFT Model        | en        | ggmlv3  | 7, 13, 33          | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| wizardlm-v1.1     | SFT Model        | en        | ggmlv3  | 13                 | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| vicuna-v1.3       | SFT Model        | en        | ggmlv3  | 7, 13              | 'q2_K', 'q3_K_L', ... , 'q6_K', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+
| orca              | SFT Model        | en        | ggmlv3  | 3, 7, 13           | 'q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0'  |
+-------------------+------------------+-----------+---------+--------------------+-----------------------------------------+

License
-------
`Apache 2 <https://github.com/xorbitsai/inference/blob/main/LICENSE>`_
