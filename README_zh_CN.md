<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference：模型推理， 轻而易举 🤖

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![WeChat](https://img.shields.io/badge/添加微信小助手-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://xorbits.cn/assets/images/wechat_pr.png)
[![Zhihu](https://img.shields.io/static/v1?style=for-the-badge&message=未来速度&color=0084FF&logo=Zhihu&logoColor=FFFFFF&label=)](https://www.zhihu.com/org/xorbits)

[English](README.md) | 中文介绍 | [日本語](README_ja_JP.md)
</div>
<br />


Xorbits Inference（Xinference）是一个性能强大且功能全面的分布式推理框架。可用于大语言模型（LLM），语音识别模型，多模态模型等各种模型的推理。通过 Xorbits Inference，你可以轻松地一键部署你自己的模型或内置的前沿开源模型。无论你是研究者，开发者，或是数据科学家，都可以通过 Xorbits Inference 与最前沿的 AI 模型，发掘更多可能。


<div align="center">
<i><a href="https://xorbits.cn/assets/images/wechat_pr.png">👉 添加微信小助手、加入Xinference社区!</a></i>
</div>

## 🔥 近期热点
### 框架增强
- Docker 镜像支持: [#855](https://github.com/xorbitsai/inference/pull/855)
- 支持多模态模型：[#829](https://github.com/xorbitsai/inference/pull/829)
- 模型自动恢复: [#694](https://github.com/xorbitsai/inference/pull/694)
- 函数调用接口: [#701](https://github.com/xorbitsai/inference/pull/701)，示例代码：https://github.com/xorbitsai/inference/blob/main/examples/FunctionCall.ipynb
- 支持 rerank 模型: [#672](https://github.com/xorbitsai/inference/pull/672)
- 支持指定 grammar 输出: [#525](https://github.com/xorbitsai/inference/pull/525)
- 投机采样: [#509](https://github.com/xorbitsai/inference/pull/509)
### 新模型
- 内置 [qwen-vl](https://huggingface.co/Qwen/Qwen-VL-Chat): [#829](https://github.com/xorbitsai/inference/pull/829)
- 内置 [phi-2](https://huggingface.co/microsoft/phi-2): [#828](https://github.com/xorbitsai/inference/pull/828)
- 内置 [mistral-instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): [#796](https://github.com/xorbitsai/inference/pull/796)
- 内置 [deepseek-llm](https://huggingface.co/deepseek-ai) 与 [deepseek-coder](https://huggingface.co/deepseek-ai): [#786](https://github.com/xorbitsai/inference/pull/786)
- 内置 [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1): [#782](https://github.com/xorbitsai/inference/pull/782)
- 内置 [OpenHermes 2.5](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B): [#776](https://github.com/xorbitsai/inference/pull/776)
### 集成
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): 一个涵盖了大型语言模型开发、部署、维护和优化的 LLMOps 平台。
- [Chatbox](https://chatboxai.app/): 一个支持前沿大语言模型的桌面客户端，支持 Windows，Mac，以及 Linux。

## 主要功能
🌟 **模型推理，轻而易举**：大语言模型，语音识别模型，多模态模型的部署流程被大大简化。一个命令即可完成模型的部署工作。 

⚡️ **前沿模型，应有尽有**：框架内置众多中英文的前沿大语言模型，包括 baichuan，chatglm2 等，一键即可体验！内置模型列表还在快速更新中！

🖥 **异构硬件，快如闪电**：通过 [ggml](https://github.com/ggerganov/ggml)，同时使用你的 GPU 与 CPU 进行推理，降低延迟，提高吞吐！

⚙️ **接口调用，灵活多样**：提供多种使用模型的接口，包括 OpenAI 兼容的 RESTful API（包括 Function Calling），RPC，命令行，web UI 等等。方便模型的管理与交互。

🌐 **集群计算，分布协同**: 支持分布式部署，通过内置的资源调度器，让不同大小的模型按需调度到不同机器，充分使用集群资源。

🔌 **开放生态，无缝对接**: 与流行的三方库无缝对接，包括 [LangChain](https://python.langchain.com/docs/integrations/providers/xinference)，[LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window)，[Dify](https://docs.dify.ai/advanced/model-configuration/xinference)，以及 [Chatbox](https://chatboxai.app/)。

## 为什么选择 Xinference
| 功能特点 | Xinference | FastChat | OpenLLM | RayLLM |
|---------|------------|----------|---------|--------|
| 兼容 OpenAI 的 RESTful API | ✅ | ✅ | ✅ | ✅ |
| vLLM 集成 | ✅ | ✅ | ✅ | ✅ |
| 更多推理引擎（GGML、TensorRT） | ✅ | ❌ | ✅ | ✅ |
| 更多平台支持（CPU、Metal） | ✅ | ✅ | ❌ | ❌ |
| 分布式集群部署 | ✅ | ❌ | ❌ | ✅ |
| 图像模型（文生图） | ✅ | ✅ | ❌ | ❌ |
| 文本嵌入模型 | ✅ | ❌ | ❌ | ❌ |
| 多模态模型 | ✅ | ❌ | ❌ | ❌ |
| 更多 OpenAI 功能 (函数调用) | ✅ | ❌ | ❌ | ❌ |


## 入门指南

**在开始之前，请给我们一个星标，这样你就可以在 GitHub 上及时收到每个新版本的通知！**

* [文档](https://inference.readthedocs.io/zh-cn/latest/index.html)
* [内置模型](https://inference.readthedocs.io/zh-cn/latest/models/builtin/index.html)
* [自定义模型](https://inference.readthedocs.io/zh-cn/latest/models/custom.html)
* [部署文档](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html)
* [示例和教程](https://inference.readthedocs.io/zh-cn/latest/examples/index.html)

### Jupyter Notebook

体验 Xinference 最轻量级的方式是使用我们 [Google Colab 上的 Jupyter Notebook](https://colab.research.google.com/github/xorbitsai/inference/blob/main/examples/Xinference_Quick_Start.ipynb)。

### Docker

Nvidia GPU 用户可以使用[Xinference Docker 镜像](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_docker_image.html) 启动 Xinference 服务器。在执行安装命令之前，确保你的系统中已经安装了 [Docker](https://docs.docker.com/get-docker/) 和 [CUDA](https://developer.nvidia.com/cuda-downloads)。

### 快速开始

使用 pip 安装 Xinference，操作如下。（更多选项，请参阅[安装页面](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)。）

```bash
pip install "xinference[all]"
```

要启动一个本地的 Xinference 实例，请运行以下命令：

```bash
$ xinference-local
```

一旦 Xinference 运行起来，你可以通过多种方式尝试它：通过网络界面、通过 cURL、通过命令行或通过 Xinference 的 Python 客户端。更多指南，请查看我们的[文档](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)。

![网络界面](assets/screenshot.png)

## 参与其中

| 平台                                                                                          | 目的                                              |
|------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [Github 问题](https://github.com/xorbitsai/inference/issues)                                  | 报告错误和提交功能请求。                          |
| [Slack](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)   | 与其他 Xorbits 用户合作。                          |
| [Twitter](https://twitter.com/xorbitsio)                                                     | 及时了解新功能。                                  |
| [微信社群](https://xorbits.cn/assets/images/wechat_pr.png)                                     | 与其他 Xorbits 用户交流。                         |
| [知乎](https://zhihu.com/org/xorbits)                                                         | 了解团队最新的进展。                                  |


