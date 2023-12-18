<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference: Model Serving Made Easy 🤖

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

English | [中文介绍](README_zh_CN.md) | [日本語](README_ja_JP.md)
</div>
<br />


Xorbits Inference(Xinference) is a powerful and versatile library designed to serve language, 
speech recognition, and multimodal models. With Xorbits Inference, you can effortlessly deploy 
and serve your or state-of-the-art built-in models using just a single command. Whether you are a 
researcher, developer, or data scientist, Xorbits Inference empowers you to unleash the full 
potential of cutting-edge AI models.

<div align="center">
<i><a href="https://join.slack.com/t/xorbitsio/shared_invite/zt-1z3zsm9ep-87yI9YZ_B79HLB2ccTq4WA">👉 Join our Slack community!</a></i>
</div>

## 🔥 Hot Topics
### Framework Enhancements
- Auto recover: [#694](https://github.com/xorbitsai/inference/pull/694)
- Function calling API: [#701](https://github.com/xorbitsai/inference/pull/701), here's example: https://github.com/xorbitsai/inference/blob/main/examples/FunctionCall.ipynb
- Support rerank model: [#672](https://github.com/xorbitsai/inference/pull/672)
- Speculative decoding: [#509](https://github.com/xorbitsai/inference/pull/509)
- Incorporate vLLM: [#445](https://github.com/xorbitsai/inference/pull/445)
### New Models
- Built-in support for [OpenHermes 2.5](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B): [#776](https://github.com/xorbitsai/inference/pull/776)
- Built-in support for [Yi](https://huggingface.co/01-ai): [#629](https://github.com/xorbitsai/inference/pull/629)
- Built-in support for [zephyr-7b-alpha](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha) and [zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta): [#597](https://github.com/xorbitsai/inference/pull/597) 
- Built-in support for [chatglm3](https://huggingface.co/THUDM/chatglm3-6b): [#587](https://github.com/xorbitsai/inference/pull/587)
- Built-in support for [mistral-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [mistral-instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1): [#510](https://github.com/xorbitsai/inference/pull/510)
### Integrations
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): an LLMOps platform that enables developers (and even non-developers) to quickly build useful applications based on large language models, ensuring they are visual, operable, and improvable.
- [Chatbox](https://chatboxai.app/): a desktop client for multiple cutting-edge LLM models, available on Windows, Mac and Linux.


## Key Features
🌟 **Model Serving Made Easy**: Simplify the process of serving large language, speech 
recognition, and multimodal models. You can set up and deploy your models
for experimentation and production with a single command.

⚡️ **State-of-the-Art Models**: Experiment with cutting-edge built-in models using a single 
command. Inference provides access to state-of-the-art open-source models!

🖥 **Heterogeneous Hardware Utilization**: Make the most of your hardware resources with
[ggml](https://github.com/ggerganov/ggml). Xorbits Inference intelligently utilizes heterogeneous
hardware, including GPUs and CPUs, to accelerate your model inference tasks.

⚙️ **Flexible API and Interfaces**: Offer multiple interfaces for interacting
with your models, supporting RPC, RESTful API(compatible with OpenAI API), CLI and WebUI
for seamless management and monitoring.

🌐 **Distributed Deployment**: Excel in distributed deployment scenarios, 
allowing the seamless distribution of model inference across multiple devices or machines.

🔌 **Built-in Integration with Third-Party Libraries**: Xorbits Inference seamlessly integrates
with popular third-party libraries including [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window), [Dify](https://docs.dify.ai/advanced/model-configuration/xinference), and [Chatbox](https://chatboxai.app/).

## Why Xinference
| Feature | Xinference | FastChat | OpenLLM | RayLLM |
|---------|------------|----------|---------|--------|
| OpenAI-Compatible Restful API | ✅ | ✅ | ✅ | ✅ |
| vLLM Integrations | ✅ | ✅ | ✅ | ✅ |
| More Inference Engines (GGML, TensorRT) | ✅ | ❌ | ✅ | ✅ |
| More Platforms (CPU, Metal) | ✅ | ✅ | ❌ | ❌ |
| Multi-node Cluster Deployment | ✅ | ❌ | ❌ | ✅ |
| Multimodal Models (Text-to-Image) | ✅ | ✅ | ❌ | ❌ |
| Text Embedding Models | ✅ | ❌ | ❌ | ❌ |

## Getting Started
### Installation
Xinference can be installed with `pip` on Linux, Windows, and macOS. It is highly recommended to create a new virtual environment to avoid conflicts.

To run models using Xinference, you will need to install the backend corresponding to the type of model you intend to serve.

If you aim to serve all supported models, you can install all the necessary dependencies with a single command:
```bash
pip install "xinference[all]"
```

**NOTE**: if you want to serve models in GGML format, it's advised to **install the GGML dependencies manually** based on your hardware specifications to enable acceleration. For more details, see the [GGML Backend](#ggml-backend) section.

#### Transformers Backend
The transformers backend supports most of the state-of-art models. It is the default backend for models in PyTorch format.
```bash
pip install "xinference[transformers]"
```
   
#### vLLM Backend
vLLM is a fast and easy-to-use library for LLM inference and serving. Xinference will choose vLLM as the backend to achieve better throughput when the following conditions are met:

- The model format is PyTorch
- The model is within the list of models supported by vLLM
- The quantization method is `none` (AWQ quantization will be supported soon)
- The system is Linux and has at least one CUDA device

Currently, supported models include:

- ``llama-2``, ``llama-2-chat``
- ``baichuan``, ``baichuan-chat``
- ``internlm``, ``internlm-20b``, ``internlm-chat``, ``internlm-chat-20b``
- ``vicuna-v1.3``, ``vicuna-v1.5``

To install Xinference and vLLM:
```bash
pip install "xinference[vllm]"
```

#### GGML Backend
It's advised to install the GGML dependencies manually based on your hardware specifications to enable acceleration.

Initial setup:
```bash
pip install xinference
pip install ctransformers
```

Hardware-Specific installations:

- Apple Silicon:
```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

- Nvidia cards:
```bash
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

- AMD cards:
```bash
    CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```


### Deployment
You can deploy Xinference locally with a single command or deploy it in a distributed cluster. 

#### Local
To start a local instance of Xinference, run the following command:
```bash
$ xinference-local
```

#### Distributed

To deploy Xinference in a cluster, you need to start a Xinference supervisor on one server and 
Xinference workers on the other servers. Follow the steps below:

**Starting the Supervisor**: On the server where you want to run the Xinference supervisor, run the following command:
```bash
$ xinference-supervisor -H "${supervisor_host}"
```
Replace `${supervisor_host}` with the actual host of your supervisor server.

**Starting the Workers**: On each of the other servers where you want to run Xinference workers, run the following command:
```bash
$ xinference-worker -e "http://${supervisor_host}:9997"
```

Once Xinference is running, an endpoint will be accessible for model management via CLI or
Xinference  client.

- For local deployment, the endpoint will be `http://localhost:9997`.
- For cluster deployment, the endpoint will be `http://${supervisor_host}:9997`, where
`${supervisor_host}` is the hostname or IP address of the server where the supervisor is running.

You can also view a web UI using the Xinference endpoint to chat with all the 
builtin models.

![web UI](assets/index.jpg)

### Xinference CLI
Xinference provides a command line interface (CLI) for model management. Here are some useful 
commands:

- Launch a model (a model UID will be returned): `xinference launch`
- List running models: `xinference list`
- List all the supported models: `xinference registrations`
- Terminate a model: `xinference terminate --model-uid ${model_uid}`

### Xinference Client
Xinference also provides a client for managing and accessing models programmatically:

```python
from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(model_name="chatglm2")
model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history=chat_history,
    generate_config={"max_tokens": 1024}
)
```

Result:
```json
{
  "id": "chatcmpl-8d76b65a-bad0-42ef-912d-4a0533d90d61",
  "model": "56f69622-1e73-11ee-a3bd-9af9f16816c6",
  "object": "chat.completion",
  "created": 1688919187,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The largest animal that has been scientifically measured is the blue whale, which has a maximum length of around 23 meters (75 feet) for adult animals and can weigh up to 150,000 pounds (68,000 kg). However, it is important to note that this is just an estimate and that the largest animal known to science may be larger still. Some scientists believe that the largest animals may not have a clear \"size\" in the same way that humans do, as their size can vary depending on the environment and the stage of their life."
      },
      "finish_reason": "None"
    }
  ],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

See [examples](examples) for more examples.


## Builtin models
To view the builtin models, run the following command:
```bash
$ xinference registrations
```

| Type  | Name                    | Language     | Ability      |
|-------|-------------------------|--------------|--------------|
| LLM   | baichuan                | ['en', 'zh'] | ['generate'] |
| LLM   | baichuan-2              | ['en', 'zh'] | ['generate'] |
| LLM   | baichuan-2-chat         | ['en', 'zh'] | ['chat']     |
| LLM   | baichuan-chat           | ['en', 'zh'] | ['chat']     |
| LLM   | chatglm                 | ['en', 'zh'] | ['chat']     |
| LLM   | chatglm2                | ['en', 'zh'] | ['chat']     |
| LLM   | chatglm2-32k            | ['en', 'zh'] | ['chat']     |
| LLM   | chatglm3                | ['en', 'zh'] | ['chat']     |
| LLM   | chatglm3-32k            | ['en', 'zh'] | ['chat']     |
| LLM   | code-llama              | ['en']       | ['generate'] |
| LLM   | code-llama-instruct     | ['en']       | ['chat']     |
| LLM   | code-llama-python       | ['en']       | ['generate'] |
| LLM   | falcon                  | ['en']       | ['generate'] |
| LLM   | falcon-instruct         | ['en']       | ['chat']     |
| LLM   | glaive-coder            | ['en']       | ['chat']     |
| LLM   | gpt-2                   | ['en']       | ['generate'] |
| LLM   | internlm-20b            | ['en', 'zh'] | ['generate'] |
| LLM   | internlm-7b             | ['en', 'zh'] | ['generate'] |
| LLM   | internlm-chat-20b       | ['en', 'zh'] | ['chat']     |
| LLM   | internlm-chat-7b        | ['en', 'zh'] | ['chat']     |
| LLM   | llama-2                 | ['en']       | ['generate'] |
| LLM   | llama-2-chat            | ['en']       | ['chat']     |
| LLM   | mistral-instruct-v0.1   | ['en']       | ['chat']     |
| LLM   | mistral-v0.1            | ['en']       | ['generate'] |
| LLM   | OpenBuddy               | ['en']       | ['chat']     |
| LLM   | openhermes-2.5          | ['en']       | ['chat']     |
| LLM   | opt                     | ['en']       | ['generate'] |
| LLM   | orca                    | ['en']       | ['chat']     |
| LLM   | qwen-chat               | ['en', 'zh'] | ['chat']     |
| LLM   | starchat-beta           | ['en']       | ['chat']     |
| LLM   | starcoder               | ['en']       | ['generate'] |
| LLM   | starcoderplus           | ['en']       | ['generate'] |
| LLM   | tiny-llama              | ['en']       | ['generate'] |
| LLM   | vicuna-v1.3             | ['en']       | ['chat']     |
| LLM   | vicuna-v1.5             | ['en']       | ['chat']     |
| LLM   | vicuna-v1.5-16k         | ['en']       | ['chat']     |
| LLM   | wizardcoder-python-v1.0 | ['en']       | ['chat']     |
| LLM   | wizardlm-v1.0           | ['en']       | ['chat']     |
| LLM   | wizardmath-v1.0         | ['en']       | ['chat']     |
| LLM   | Yi                      | ['en', 'zh'] | ['generate'] |
| LLM   | Yi-200k                 | ['en', 'zh'] | ['generate'] |
| LLM   | zephyr-7b-alpha         | ['en']       | ['chat']     |
| LLM   | zephyr-7b-beta          | ['en']       | ['chat']     |

For in-depth details on the built-in models, please refer to [built-in models](https://inference.readthedocs.io/en/latest/models/builtin/index.html). 

**NOTE**:
- Xinference will download models automatically for you, and by default the models will be saved under `${USER}/.xinference/cache`. 
- If you have trouble downloading models from the Hugging Face, run `export XINFERENCE_MODEL_SRC=modelscope` to download models from [modelscope](https://modelscope.cn/). Models supported by modelscope:
  - llama-2
  - llama-2-chat
  - baichuan-2
  - baichuan-2-chat
  - chatglm2
  - chatglm2-32k
  - internlm-chat-20b
  - ...
  
  More supported models can be found in the [documentation](https://inference.readthedocs.io/en/latest/models/sources/sources.html#modelscope)

## Custom models
Please refer to [custom models](https://inference.readthedocs.io/en/latest/models/custom.html).
