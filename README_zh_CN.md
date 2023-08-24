<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference：模型推理， 轻而易举 🤖

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

[English](README.md) | 中文介绍 | [日本語](README_ja_JP.md)
</div>
<br />


Xorbits Inference（Xinference）是一个性能强大且功能全面的分布式推理框架。可用于大语言模型（LLM），语音识别模型，多模态模型等各种模型的推理。通过 Xorbits Inference，你可以轻松地一键部署你自己的模型或内置的前沿开源模型。无论你是研究者，开发者，或是数据科学家，都可以通过 Xorbits Inference 与最前沿的 AI 模型，发掘更多可能。


<div align="center">
<i><a href="https://join.slack.com/t/xorbitsio/shared_invite/zt-1z3zsm9ep-87yI9YZ_B79HLB2ccTq4WA">👉 立刻加入我们的 Slack 社区!</a></i>
</div>

## 🔥 近期热点
### 框架增强
- 自定义模型: [#325](https://github.com/xorbitsai/inference/pull/325)
- LoRA 支持: [#271](https://github.com/xorbitsai/inference/issues/271)
- PyTorch 模型多 GPU 支持: [#226](https://github.com/xorbitsai/inference/issues/226)
- Xinference 仪表盘: [#93](https://github.com/xorbitsai/inference/issues/93)
### 新模型
- 内置 GGML 格式的 Starcoder: [#289](https://github.com/xorbitsai/inference/pull/289)
- 内置 [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md): [#313](https://github.com/xorbitsai/inference/issues/313)
- 内置 [SD-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): [#318](https://github.com/xorbitsai/inference/issues/318)
### 工具
- LlamaIndex 插件: [#7151](https://github.com/jerryjliu/llama_index/pull/7151)



## 主要功能
🌟 **模型推理，轻而易举**：大语言模型，语音识别模型，多模态模型的部署流程被大大简化。一个命令即可完成模型的部署工作。 

⚡️ **前沿模型，应有尽有**：框架内置众多中英文的前沿大语言模型，包括 baichuan，chatglm2 等，一键即可体验！内置模型列表还在快速更新中！


🖥 **异构硬件，快如闪电**：通过 [ggml](https://github.com/ggerganov/ggml)，同时使用你的 GPU 与 CPU 进行推理，降低延迟，提高吞吐！

⚙️ **接口调用，灵活多样**：提供多种使用模型的接口，包括 RPC，RESTful API，命令行，web UI 等等。方便模型的管理与监控。

🌐 **集群计算，分布协同**: 支持分布式部署，通过内置的资源调度器，让不同大小的模型按需调度到不同机器，充分使用集群资源。

🔌 **开放生态，无缝对接**: 与流行的三方库无缝对接，包括 [LangChain](https://python.langchain.com/docs/integrations/providers/xinference) 
and [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window)。
让开发者能够快速构建基于 AI 的应用。

## 快速入门
Xinference 可以通过 pip 从 PyPI 安装。我们非常推荐在安装前创建一个新的虚拟环境以避免依赖冲突。

### 安装
```bash
$ pip install "xinference"
```
`xinference` 将会安装所有用于推理的基础依赖。

#### 支持 ggml 推理
想要利用 ggml 推理，可以用以下命令：
```bash
$ pip install "xinference[ggml]"
```
如果你想要获得更高效的加速，请查看下列依赖的安装文档：
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python#installation-from-pypi-recommended) 用于 `baichuan`, `wizardlm-v1.0`, `vicuna-v1.3` 及 `orca`.
- [chatglm-cpp-python](https://github.com/li-plus/chatglm.cpp#getting-started) 用于 `chatglm` 及 `chatglm2`.

#### 支持 PyTorch 推理
想要利用 PyTorch 推理，可以使用以下命令：
```bash
$ pip install "xinference[pytorch]"
```

#### 支持所有类型
如果想要支持推理所有支持的模型，可以安装所有的依赖：
```bash
$ pip install "xinference[all]"
```


### 部署
你可以一键进行本地部署，或按照下面的步骤将 Xinference 部署在计算集群。 

#### 本地部署
运行下面的命令在本地部署 Xinference：
```bash
$ xinference
```

#### 分布式部署
分布式场景下，你需要在一台服务器上部署一个 Xinference supervisor，并在其余服务器上分别部署一个 Xinference worker。 具体步骤如下：

**启动 supervisor**: 执行:
```bash
$ xinference-supervisor -H "${supervisor_host}"
```
替换 `${supervisor_host}` 为 supervisor 所在服务器的实际主机名或 IP 地址。

**启动 workers**: 在其余服务器上，执行：
```bash
$ xinference-worker -e "http://${supervisor_host}:9997"
```

Xinference 启动后，将会打印服务的 endpoint。这个 endpoint 用于通过命令行工具或编程接口进行模型的管理。

- 本地部署下, endpoint 默认为 `http://localhost:9997`.
- 集群部署下, endpoint 默认为 `http://${supervisor_host}:9997`。其中 `${supervisor_host}` 为supervisor 所在服务器的主机名或 IP 地址。

你还可以通过 web UI 与任意内置模型聊天。Xinference 甚至**支持同时与两个最前沿的 AI 模型聊天并比较它们的回复质量**！

![web UI](assets/demo.gif)

### Xinference 命令行
Xinference 提供了命令行工具用于模型管理。支持的命令包括：

- 启动一个模型 (将会返回一个模型 UID)：`xinference launch`
- 查看所有运行中的模型：`xinference list`
- 查看所有内置模型：`xinference list --all`
- 结束模型：`xinference terminate --model-uid ${model_uid}`

### Xinference 编程接口
Xinference 同样提供了编程接口：

```python
from xinference.client import Client

client = Client("http://localhost:9997")
model_uid = client.launch_model(model_name="chatglm2")
model = client.get_model(model_uid)

chat_history = []
prompt = "What is the largest animal?"
model.chat(
    prompt,
    chat_history,
    generate_config={"max_tokens": 1024}
)
```

返回值：
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

请参考 [更多案例](examples)。


## 内置模型
运行以下命令查看内置模型列表：
```bash
$ xinference list --all
```

| Name             | Language      | Ability                |
|------------------|---------------|------------------------|
| baichuan         | ['en', 'zh']  | ['embed', 'generate']  |
| baichuan-chat    | ['en', 'zh']  | ['embed', 'chat']      |
| chatglm          | ['en', 'zh']  | ['embed', 'chat']      |
| chatglm2         | ['en', 'zh']  | ['embed', 'chat']      |
| chatglm2-32k     | ['en', 'zh']  | ['embed', 'chat']      |
| falcon           | ['en']        | ['embed', 'generate']  |
| falcon-instruct  | ['en']        | ['embed', 'chat']      |
| gpt-2            | ['en']        | ['generate']           |
| internlm         | ['en', 'zh']  | ['embed', 'generate']  |
| internlm-chat    | ['en', 'zh']  | ['embed', 'chat']      |
| internlm-chat-8k | ['en', 'zh']  | ['embed', 'chat']      |
| llama-2          | ['en']        | ['embed', 'generate']  |
| llama-2-chat     | ['en']        | ['embed', 'chat']      |
| opt              | ['en']        | ['embed', 'generate']  |
| orca             | ['en']        | ['embed', 'chat']      |
| qwen-chat        | ['en', 'zh']  | ['embed', 'chat']      |
| starchat-beta    | ['en']        | ['embed', 'chat']      |
| starcoder        | ['en']        | ['generate']           |
| starcoderplus    | ['en']        | ['embed', 'generate']  |
| vicuna-v1.3      | ['en']        | ['embed', 'chat']      |
| vicuna-v1.5      | ['en']        | ['embed', 'chat']      |
| vicuna-v1.5-16k  | ['en']        | ['embed', 'chat']      |
| wizardlm-v1.0    | ['en']        | ['embed', 'chat']      |
| wizardmath-v1.0  | ['en']        | ['embed', 'chat']      |

更多信息请参考 [内置模型](https://inference.readthedocs.io/en/latest/models/builtin/index.html)。

**注意**:
- Xinference 会自动为你下载模型，默认的模型存放路径为 `${USER}/.xinference/cache`。

## 自定义模型
请参考 [自定义模型](https://inference.readthedocs.io/en/latest/models/custom.html)。
