<div align="center">
<img src="./assets/xorbits-logo.png" width="180px" alt="xorbits" />

# Xorbits Inference: Model Serving Made Easy 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-hosting</a> ·
  <a href="https://inference.readthedocs.io/">Documentation</a>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Docker Pulls](https://img.shields.io/docker/pulls/xprobe/xinference?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xprobe/xinference)
[![Discord](https://img.shields.io/badge/join_Discord-5462eb.svg?logo=discord&style=for-the-badge&logoColor=%23f5f5f5)](https://discord.gg/Xw9tszSkr5)
[![Telegram](https://img.shields.io/badge/join_Telegram-26A5E4.svg?logo=telegram&style=for-the-badge&logoColor=white)](https://t.me/+nCNpwmySwk9iYmI1)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

<p align="center">
  <a href="./README.md"><img alt="English" src="https://img.shields.io/badge/English-454545?style=for-the-badge"></a>
  <a href="./READMES/README_ja_JP.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9?style=for-the-badge"></a>
  <br>
  <a href="./READMES/README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./READMES/README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />


Xorbits Inference(Xinference) is a powerful and versatile library designed to serve language, 
speech recognition, and multimodal models. With Xorbits Inference, you can effortlessly deploy 
and serve your or state-of-the-art built-in models using just a single command. Whether you are a 
researcher, developer, or data scientist, Xorbits Inference empowers you to unleash the full 
potential of cutting-edge AI models.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Join our Discord community!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Join our Telegram group!</a></i>
</div>

## 🔥 Hot Topics
### Framework Enhancements
- Xinference 3.0.0 is available with migration notes and breaking changes: [Release Notes](https://xinference.co/release_notes/v3.0.0.html)
- Agent-native Serving: Xinference integrates with [Xagent](https://github.com/xorbitsai/xagent) to enable dynamic planning, tool use, and autonomous multi-step reasoning — moving beyond static pipelines.
- Auto batch: Multiple concurrent requests are automatically batched, significantly improving throughput: [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): New llama.cpp Python binding, maintained by Xinference team, supports continuous batching and is more production-ready.: [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Distributed inference: running models across workers: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- VLLM enhancement: Shared KV cache across multiple replicas: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### New Models
- Built-in support for [Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2): [#5437](https://github.com/xorbitsai/inference/pull/5437)
- Built-in support for WeMM-Embedding series ([2B](https://huggingface.co/tencent/WeMM-Embedding-2B), [4B](https://huggingface.co/tencent/WeMM-Embedding-4B), [9B](https://huggingface.co/tencent/WeMM-Embedding-9B)): [#5439](https://github.com/xorbitsai/inference/pull/5439)
- Built-in support for [NaviDC-OCR](https://huggingface.co/StarDoc-AI/NaviDC-OCR): [#5431](https://github.com/xorbitsai/inference/pull/5431)
- Built-in support for [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3): [#5417](https://github.com/xorbitsai/inference/pull/5417)
- Built-in support for world models ([Matrix-Game-3.0-5B](https://huggingface.co/Skywork/Matrix-Game-3.0), [HY-WorldPlay-5B](https://huggingface.co/tencent/HY-WorldPlay), [Astra](https://huggingface.co/EvanEternal/Astra)): [#5414](https://github.com/xorbitsai/inference/pull/5414)
- Built-in support for Krea 2 series ([Raw](https://huggingface.co/krea/Krea-2-Raw), [Turbo](https://huggingface.co/krea/Krea-2-Turbo)): [#5419](https://github.com/xorbitsai/inference/pull/5419)
- Built-in support for [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5): [#5413](https://github.com/xorbitsai/inference/pull/5413)
- Built-in support for Ornith 1.5 series ([35B-A3B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-35B-A3B), [397B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-397B)): [#5406](https://github.com/xorbitsai/inference/pull/5406), [#5405](https://github.com/xorbitsai/inference/pull/5405)
- Built-in support for [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2): [#5404](https://github.com/xorbitsai/inference/pull/5404)
- Built-in support for [GLM-Image](https://huggingface.co/zai-org/GLM-Image): [#5394](https://github.com/xorbitsai/inference/pull/5394)
- Built-in support for HiDream-O1 series ([Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image), [Image-Dev](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev), [Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604)): [#5370](https://github.com/xorbitsai/inference/pull/5370)
- Built-in support for [SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT): [#5369](https://github.com/xorbitsai/inference/pull/5369)
- Built-in support for [Ideogram4](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers): [#5367](https://github.com/xorbitsai/inference/pull/5367)
- Built-in support for [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731): [#5371](https://github.com/xorbitsai/inference/pull/5371)
- Built-in support for [FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3): [#5352](https://github.com/xorbitsai/inference/pull/5352)
- Built-in support for [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): [#5345](https://github.com/xorbitsai/inference/pull/5345)
- Built-in support for Qwen3.8 series ([27B](https://huggingface.co/Qwen/Qwen3.8-27B), [2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)): [#5337](https://github.com/xorbitsai/inference/pull/5337), [#5339](https://github.com/xorbitsai/inference/pull/5339)
- Built-in support for [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0): [#5327](https://github.com/xorbitsai/inference/pull/5327)
- Built-in support for [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2): [#5322](https://github.com/xorbitsai/inference/pull/5322)
- Built-in support for [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5): [#5319](https://github.com/xorbitsai/inference/pull/5319)
- Built-in support for Ling-3.0 series ([tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny), [flash](https://huggingface.co/inclusionAI/Ling-3.0-flash)): [#5311](https://github.com/xorbitsai/inference/pull/5311)
- Built-in support for [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and [Lightning LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo): [#5321](https://github.com/xorbitsai/inference/pull/5321), [#5338](https://github.com/xorbitsai/inference/pull/5338)
- Built-in support for Wan2.2 Animate 2 series ([14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers), [14B Distilled](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers)): [#5309](https://github.com/xorbitsai/inference/pull/5309)
- Built-in support for [FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1): [#5306](https://github.com/xorbitsai/inference/pull/5306)
- Built-in support for CAMPPlus speaker embedding series ([Chinese](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common), [Chinese-English advanced](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)): [#5298](https://github.com/xorbitsai/inference/pull/5298)
- Built-in support for [DeepDoc](https://huggingface.co/Xorbits/deepdoc): [#5230](https://github.com/xorbitsai/inference/pull/5230)
- Built-in support for [jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5): [#5269](https://github.com/xorbitsai/inference/pull/5269)
- Built-in support for R3 series ([embedding](https://huggingface.co/tencent/R3-embedding-0.6b), [rerank](https://huggingface.co/tencent/R3-rerank-0.6b)): [#5272](https://github.com/xorbitsai/inference/pull/5272)
### Integrations
- [Xagent](https://github.com/xorbitsai/xagent): an enterprise agent platform for building and running AI agents with planning, memory, and tool use — not limited to rigid workflows.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): an LLMOps platform that enables developers (and even non-developers) to quickly build useful applications based on large language models, ensuring they are visual, operable, and improvable.
- [FastGPT](https://github.com/labring/FastGPT): a knowledge-based platform built on the LLM, offers out-of-the-box data processing and model invocation capabilities, allows for workflow orchestration through Flow visualization.
- [RAGFlow](https://github.com/infiniflow/ragflow): is an open-source RAG engine based on deep document understanding.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): MaxKB = Max Knowledge Brain, it is a powerful and easy-to-use AI assistant that integrates Retrieval-Augmented Generation (RAG) pipelines, supports robust workflows, and provides advanced MCP tool-use capabilities.


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
with your models, supporting OpenAI compatible RESTful API (including Function Calling API), RPC, CLI 
and WebUI for seamless model management and interaction.

🌐 **Distributed Deployment**: Excel in distributed deployment scenarios, 
allowing the seamless distribution of model inference across multiple devices or machines.

🔌 **Built-in Integration with Third-Party Libraries**: Xorbits Inference seamlessly integrates
with popular third-party libraries including [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex](https://gpt-index.readthedocs.io/en/stable/examples/llm/XinferenceLocalDeployment.html#i-run-pip-install-xinference-all-in-a-terminal-window), [Dify](https://docs.dify.ai/advanced/model-configuration/xinference), and [Chatbox](https://chatboxai.app/).

## Why Xinference
| Feature                                        | Xinference | FastChat | OpenLLM | RayLLM |
|------------------------------------------------|------------|----------|---------|--------|
| OpenAI-Compatible RESTful API                  | ✅ | ✅ | ✅ | ✅ |
| vLLM Integrations                              | ✅ | ✅ | ✅ | ✅ |
| More Inference Engines (GGML, TensorRT)        | ✅ | ❌ | ✅ | ✅ |
| More Platforms (CPU, Metal)                    | ✅ | ✅ | ❌ | ❌ |
| Multi-node Cluster Deployment                  | ✅ | ❌ | ❌ | ✅ |
| Image Models (Text-to-Image)                   | ✅ | ✅ | ❌ | ❌ |
| Text Embedding Models                          | ✅ | ❌ | ❌ | ❌ |
| Multimodal Models                              | ✅ | ❌ | ❌ | ❌ |
| Audio Models                                   | ✅ | ❌ | ❌ | ❌ |
| More OpenAI Functionalities (Function Calling) | ✅ | ❌ | ❌ | ❌ |

## Using Xinference

- **Self-hosting Xinference Community Edition</br>**
Quickly get Xinference running in your environment with this [starter guide](#getting-started).
Use our [documentation](https://inference.readthedocs.io/) for further references and more in-depth instructions.

- **Xinference for enterprise / organizations</br>**
We provide additional enterprise-centric features. [send us an email](mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry) to discuss enterprise needs. </br>

## Staying Ahead

Star Xinference on GitHub and be instantly notified of new releases.

![star-us](assets/stay_ahead.gif)

## Getting Started

* [Docs](https://inference.readthedocs.io/en/latest/index.html)
* [Built-in Models](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Custom Models](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Deployment Docs](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Docker 

Nvidia GPU users can start Xinference server using [Xinference Docker Image](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html). Prior to executing the installation command, ensure that both [Docker](https://docs.docker.com/get-docker/) and [CUDA](https://developer.nvidia.com/cuda-downloads) are set up on your system.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s via helm

Ensure that you have GPU support in your Kubernetes cluster, then install as follows.

```
# add repo
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# update indexes and query xinference versions
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# install xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

For more customized installation methods on K8s, please refer to the [documentation](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html).

### Quick Start

Install Xinference by using pip as follows. (For more options, see [Installation page](https://inference.readthedocs.io/en/latest/getting_started/installation.html).)

```bash
pip install "xinference[all]"
```

To start a local instance of Xinference, run the following command:

```bash
$ xinference-local
```

Once Xinference is running, there are multiple ways you can try it: via the web UI, via cURL,
 via the command line, or via the Xinference’s python client. Check out our [docs]( https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally) for the guide.

![web UI](assets/screenshot.png)

## Getting involved

| Platform                                                                                        | Purpose                                     |
|-------------------------------------------------------------------------------------------------|---------------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)                                  | Reporting bugs and filing feature requests. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Collaborating with other Xinference users.  |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                                       | Chatting with other Xinference users.       |
| [Twitter](https://twitter.com/xorbitsio)                                                        | Staying up-to-date on new features.         |

## Citation

If this work is helpful, please kindly cite as:

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

## Contributors

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Star History

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
