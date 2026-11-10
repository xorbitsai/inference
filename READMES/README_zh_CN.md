# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference（Xorbits Inference）是一个开源分布式推理平台。**

从本地单机到分布式集群，Xinference 在异构硬件上部署和运行模型。内置 300 多个模型，覆盖语言、视觉、文本嵌入（Embedding）、重排（Rerank）、图像、音频、视频和世界模型（实验性）。也可以注册自己的模型，包括通过 Flexible 类型运行传统机器学习模型。通过 Web UI 管理部署，使用 OpenAI 兼容接口、Python SDK 或命令行调用模型。

[快速开始](#快速开始) · [主要功能](#主要功能) · [选型对比](#选型对比) · [自托管安装](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html) · [文档](https://inference.readthedocs.io/zh-cn/latest/) · [模型目录](https://inference.readthedocs.io/zh-cn/latest/models/builtin/index.html) · [发布记录](https://github.com/xorbitsai/inference/releases) · [Xinference 企业版](https://xinference.cn)

[![GitHub release](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![PyPI version](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Docker tag](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Python versions](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![Python CI](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Apache 2.0 license](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Contributors](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Docker pulls](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Discord community](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Telegram community](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![微信社群](https://img.shields.io/badge/%E5%BE%AE%E4%BF%A1-%E5%8A%A0%E5%85%A5%E7%A4%BE%E7%BE%A4-07C160?logo=wechat&logoColor=white&style=flat-square)](https://xinference.cn/images/WeCom.jpg)
[![知乎](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-%E6%9C%AA%E6%9D%A5%E9%80%9F%E5%BA%A6-0084FF?logo=zhihu&logoColor=white&style=flat-square)](https://www.zhihu.com/org/xorbits)

**阅读语言** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · **简体中文**

## 快速开始

### 1. 启动服务

首次部署时，Xinference 会下载模型，并按需安装所选引擎的依赖。

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

打开 [http://127.0.0.1:9997](http://127.0.0.1:9997)。首次使用时，从服务启动日志中取出 Setup Token，创建管理员账号并登录。 [首次启动配置](https://inference.readthedocs.io/zh-cn/latest/user_guide/auth_system.html#initial-admin-account)

### 2. 启动聊天模型

在 **Launch Model（启动模型）** 搜索 `qwen3.5` 并打开启动面板。点击 **Recommend configuration（推荐配置）**，Xinference 会根据当前硬件填好配置。确认后点击 **Deploy（部署）**。

![应用推荐配置后的完整 Qwen3.5 启动面板](../assets/guide-launch-recommend-auto.webp)

### 3. 发出第一条请求

在 **Running Models（运行中的模型）** 打开刚部署的模型，就可以在网页里对话。首次部署时，Xinference 会用 `qwen3.5` 作为 Model UID。如果页面显示了其他 UID，把下面请求里的 `model` 换成页面显示的值。在 **API Key Management（API Key 管理）** 创建 API Key 后，就能从终端发送请求。

```bash
export XINFERENCE_API_KEY="<你的 API Key>"
curl --fail-with-body http://127.0.0.1:9997/v1/chat/completions \
  -H "Authorization: Bearer $XINFERENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Translate this sentence into Spanish: The model is ready."}],
    "enable_thinking": false,
    "max_tokens": 96
  }'
```

![qwen3.5 在 Web UI 中返回了第一条回复](../assets/guide-chat-en.webp)

## 主要功能

| 类别 | Xinference 支持的能力 |
| --- | --- |
| 模型目录 | 内置 300 多个模型，覆盖语言、视觉、文本嵌入、重排、图像、音频、视频和世界模型（实验性）。也可以注册自己的模型，包括用于传统机器学习的 Flexible 模型。 [内置模型](https://inference.readthedocs.io/zh-cn/latest/models/builtin/index.html) |
| 推理引擎与硬件 | 大语言模型可按支持情况选择 vLLM、SGLang、TRT-LLM、Transformers、llama.cpp 或 MLX。不同引擎支持的硬件各有范围，覆盖 CPU、GPU、Apple Silicon 和部分寒武纪 MLU 配置。 [推理后端](https://inference.readthedocs.io/zh-cn/latest/user_guide/backends.html) · [MLU 支持](https://github.com/xorbitsai/inference/pull/3693) |
| 分布式推理 | 支持单机运行、增加 Worker 和模型副本，也可以让支持的大模型跨 Worker 运行。符合条件的部署可进行异构 Prefill/Decode 分离。 [分布式推理](https://inference.readthedocs.io/zh-cn/latest/user_guide/distributed_inference.html) · [P/D 分离](https://inference.readthedocs.io/zh-cn/latest/user_guide/pd_separation.html) |
| 接口与工具 | 提供兼容 OpenAI 和 Anthropic 的接口，支持的模型可使用流式输出和工具调用。可以通过 Web UI、CLI 或 Python SDK 部署和管理模型。 [客户端指南](https://inference.readthedocs.io/zh-cn/latest/user_guide/client_api.html) · [工具调用](https://inference.readthedocs.io/zh-cn/latest/models/model_abilities/tools.html) |
| 性能与运维 | 可以在 Web UI 中对运行中的聊天模型做压测。符合条件的部署支持 Xavier 前缀缓存共享、运行时参数调整和逐副本重载。 [性能测试](https://inference.readthedocs.io/zh-cn/latest/user_guide/benchmarks.html) · [运行时配置](https://inference.readthedocs.io/zh-cn/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| 多语言 | Web UI 支持英语、简体中文、繁体中文、日语和韩语，README 提供十种语言版本。 |

## 选型对比

| 能力 | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **单机启动（无需 Kubernetes）** | ![支持][compare-supported] | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![支持][compare-supported] |
| **多节点推理** | ![支持][compare-supported] | ![支持][compare-supported] | ![支持][compare-supported] | ![支持][compare-supported] |
| **CPU / GPU / Apple Metal** | ![支持][compare-supported] | ![支持][compare-supported] | ![部分支持或需集成][compare-partial] | ![部分支持或需集成][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![支持][compare-supported] | ![支持][compare-supported] | ![部分支持或需集成][compare-partial] | ![部分支持或需集成][compare-partial] |
| **CUDA ↔ Metal 异构 P/D** | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![官方文档未列出支持][compare-undocumented] | ![官方文档未列出支持][compare-undocumented] |
| **同一模型 ID 混用不同引擎的副本** | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![部分支持或需集成][compare-partial] | ![部分支持或需集成][compare-partial] |
| **复用 GPU 权重逐副本更新配置** | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![官方文档未列出支持][compare-undocumented] | ![官方文档未列出支持][compare-undocumented] |
| 模型目录 | ![支持][compare-supported] 300+ | ![支持][compare-supported] | ![部分支持或需集成][compare-partial] | ![部分支持或需集成][compare-partial] |
| 模型部署 Web UI | ![支持][compare-supported] | ![支持][compare-supported] | ![部分支持或需集成][compare-partial] | ![官方文档未列出支持][compare-undocumented] |
| 向量嵌入与重排 | ![支持][compare-supported] | ![支持][compare-supported] | ![支持][compare-supported] | ![支持][compare-supported] |
| 图像与视频生成 | ![支持][compare-supported] | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![支持][compare-supported] |
| 语音识别与合成 | ![支持][compare-supported] | ![支持][compare-supported] | ![官方文档未列出支持][compare-undocumented] | ![部分支持或需集成][compare-partial] |

![][compare-supported] 支持 · ![][compare-partial] 部分支持或需集成 · ![][compare-undocumented] 官方文档未列出支持

<details>
<summary>对比口径与来源</summary>

按开源版本的官方文档对比。具体引擎和硬件支持取决于模型。CUDA ↔ Metal 异构 P/D 需要兼容的模型和引擎组合。

Xinference 可在同一模型 ID 下为新增副本指定引擎。KServe 和 Dynamo 可以在多个独立部署间路由。GPU 权重复用适用于受支持的 vLLM 和 SGLang 配置。

- **Xinference** · [模型目录](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [推理后端](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [分布式部署](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [CUDA 与 Metal 跨引擎 P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [新增副本指定引擎](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [逐副本更新配置](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [模型与部署能力](https://localai.io/docs/features/index.html) · [引擎与硬件](https://localai.io/docs/reference/index.print.html) · [分布式部署](https://localai.io/docs/features/distributed-mode/) · [模型别名与故障转移](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [部署与硬件](https://kserve.github.io/website/docs/intro) · [内置生成式运行时](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Kubeflow 模型目录](https://www.kubeflow.org/docs/components/hub/installation/) · [Kubeflow Web UI](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [灰度发布](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [部署、硬件与引擎](https://docs.dynamo.nvidia.com/dynamo) · [部署配方](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [图像、视频与语音合成](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [嵌入与重排支持](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [多部署路由](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## 近期更新

[查看 v4.2.0 发布说明](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>最近两个版本新增的内置模型（v4.1、v4.2）</summary>

- 内置支持 [Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b): [#6003](https://github.com/xorbitsai/inference/pull/6003)
- 内置支持 [MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0): [#5916](https://github.com/xorbitsai/inference/pull/5916)
- 内置支持 [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3): [#5914](https://github.com/xorbitsai/inference/pull/5914)
- 内置支持 [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2): [#5913](https://github.com/xorbitsai/inference/pull/5913)
- 内置支持 [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein): [#5792](https://github.com/xorbitsai/inference/pull/5792)
- 内置支持 FIBO 2.0 系列（[Dev](https://huggingface.co/briaai/Fibo-2.0-dev)、[Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)）: [#5790](https://github.com/xorbitsai/inference/pull/5790)
- 内置支持 Ming-3.0 TTS 系列（[Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny)、[0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)）: [#5788](https://github.com/xorbitsai/inference/pull/5788)
- 内置支持 [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0): [#5764](https://github.com/xorbitsai/inference/pull/5764)
- 内置支持 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT): [#5755](https://github.com/xorbitsai/inference/pull/5755)
- 内置支持 [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): [#5753](https://github.com/xorbitsai/inference/pull/5753)
- 内置支持 Iris-3B 系列（[Base](https://huggingface.co/speridlabs/iris-3b)、[Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)）: [#5686](https://github.com/xorbitsai/inference/pull/5686)
- 内置支持 MiniCPM-V-4.7 系列（[1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)、[35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)）: [#5676](https://github.com/xorbitsai/inference/pull/5676)
- 内置支持 Qwen3.8-Flash-Next 系列（[Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)）: [#5673](https://github.com/xorbitsai/inference/pull/5673)
- 内置支持 GLM-5.3 系列（[BF16](https://huggingface.co/zai-org/GLM-5.3-BF16)、[FP8](https://huggingface.co/zai-org/GLM-5.3)）: [#5666](https://github.com/xorbitsai/inference/pull/5666)
- 内置支持 [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo): [#5665](https://github.com/xorbitsai/inference/pull/5665)
- 内置支持 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): [#5661](https://github.com/xorbitsai/inference/pull/5661)

</details>

## 模型性能测试

在 **Performance Tests（性能测试）** 中选择一个运行中的聊天模型，设定想测试的并发数。对比吞吐、首 Token 延迟和每个输出 Token 的耗时，再决定怎么配服务。 [压测指南](https://inference.readthedocs.io/zh-cn/latest/user_guide/benchmarks.html)

![Web UI 的压测结果，展示不同并发下的吞吐和延迟](../assets/guide-benchmark-en.webp)

*图中数据来自本地 Qwen3.5-0.8B 的一次运行，实际表现取决于硬件、引擎和负载。*

## 使用 Xinference

| 你想做什么 | 从这里开始 |
| --- | --- |
| 在 Linux、macOS 或 Windows 上安装 | [一键安装](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html#one-line-install) · [安装指南](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html) |
| 部署到容器或 Kubernetes | [Docker](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_kubernetes.html) |
| 注册自己的模型 | [自定义模型](https://inference.readthedocs.io/zh-cn/latest/models/custom.html) |
| 接入应用 | [OpenAI 兼容接口](https://inference.readthedocs.io/zh-cn/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**应用与开发框架** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.cn/zh-CN/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference 企业版

Xinference 企业版支持部署到自己的云环境或数据中心。Xinference Cloud 提供托管的独占部署，Model API 提供可直接调用的托管模型接口。 [企业版官网](https://xinference.cn)

**商务咨询** · [企业微信（扫码联系）](https://xinference.cn/images/WeCom.jpg) · [飞书表单（提交需求）](https://w8v6grm432.feishu.cn/share/base/form/shrcn9u1EBXQxmGMqILEjguuGoh)

## 参与其中

| 平台 | 用途 |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | 报告问题、提交功能建议。 |
| [Discord](https://discord.gg/Xw9tszSkr5) | 提问并和其他 Xinference 用户交流。 |
| [微信社群](https://xinference.cn/images/WeCom.jpg) | 扫码添加微信小助手，加入社群。 |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | 加入群组，与其他 Xinference 用户交流。 |
| [X](https://x.com/Xinference_ai) | 关注项目动态和版本发布。 |
| [知乎](https://www.zhihu.com/org/xorbits) | 查看团队分享和项目进展。 |

## 引用

如果在研究中使用 Xinference，欢迎引用 [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)。

```bibtex
@inproceedings{lu2024xinference,
    title = "Xinference: Making Large Model Serving Easy",
    author = "Lu, Weizheng and Xiong, Lingfeng and Zhang, Feng and Qin, Xuye and Chen, Yueguo",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-demo.30",
    pages = "291--300",
}
```

## 贡献者

感谢每一位参与建设 Xinference 的贡献者。欢迎添加模型、完善文档或修复一个 Bug，提交你的第一个 PR。 [查看贡献指南](https://inference.readthedocs.io/zh-cn/latest/development/contributing_codebase.html)。

[![Xinference 贡献者头像墙](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
