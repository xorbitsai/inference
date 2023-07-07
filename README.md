[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/xinference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/xinference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/xinference/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

# Xorbits inference: Model Serving Made Easy 🤖

Welcome to the Xorbits Inference GitHub repository!

Xorbits Inference is a powerful and versatile library designed to serve language, speech recognition, 
and multimodal models. With Inference, you can effortlessly deploy and serve your or state-of-the-art 
built-in models using just a single command. Whether you are a researcher, developer, or data scientist, 
Inference empowers you to unleash the full potential of cutting-edge AI models.

Currently, Xorbits Inference relies on ggml for model serving, which is specifically designed to 
enable large models and high performance on commodity hardware. We are actively working on expanding 
Inference's support to include additional runtimes, including PyTorch and JAX, in the near future.

## Key Features
🌟 **Model Serving Made Easy**: Inference simplifies the process of serving large language, speech recognition,
and multimodal models. With a single command, you can set up and deploy your models for experimentation and production.

⚡️ **State-of-the-Art Models**: Experiment with cutting-edge built-in models using a single command. Inference provides
access to state-of-the-art open-source models including LLaMA, Whisper and more!

🖥 **Heterogeneous Hardware Utilization**: Make the most of your hardware resources. Inference intelligently utilizes 
heterogeneous hardware, including GPUs and CPUs, to maximize performance and accelerate your model training and 
inference tasks.

⚙️ **Flexible API and Interfaces**: Inference offers multiple interfaces for interacting with your models. You can
utilize the RPC and RESTful API(compatible with OpenAI API) to integrate your models with existing systems or use 
the command-line interface (CLI) and the intuitive WebUI for seamless management and monitoring.

🌐 **Distributed Deployment**: Inference excels in distributed deployment scenarios, allowing the seamless
distribution of model inference across multiple devices or machines. It leverages distributed computing techniques 
to parallelize and scale the inference process.

🔌 **Built-in Integration with Third-Party Libraries**: Inference provides seamless integration with popular third-party 
libraries like LangChain and LlamaIndex.

🛠️ **Customize LLM with your data**: Fine-tune your LLM on your own data to suit your specific needs. (Coming soon)
