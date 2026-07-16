<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference：讓模型部署變得簡單 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">自我託管（Self-Hosting）</a> ·
  <a href="https://inference.readthedocs.io/">文件</a>
</p>

[![PyPI Latest Release](https://img.shields.io/pypi/v/xinference.svg?style=for-the-badge)](https://pypi.org/project/xinference/)
[![License](https://img.shields.io/pypi/l/xinference.svg?style=for-the-badge)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/inference/goto?ref=main)
[![Docker Pulls](https://img.shields.io/docker/pulls/xprobe/xinference?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xprobe/xinference)
[![Discord](https://img.shields.io/badge/join_Discord-5462eb.svg?logo=discord&style=for-the-badge&logoColor=%23f5f5f5)](https://discord.gg/Xw9tszSkr5)
[![Telegram](https://img.shields.io/badge/join_Telegram-26A5E4.svg?logo=telegram&style=for-the-badge&logoColor=white)](https://t.me/+nCNpwmySwk9iYmI1)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=x&style=for-the-badge)](https://twitter.com/xorbitsio)

<p align="center">
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9?style=for-the-badge"></a>
  <a href="./README_ja_JP.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9?style=for-the-badge"></a>
  <a href="./README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9?style=for-the-badge"></a>
  <a href="./README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9?style=for-the-badge"></a>
  <a href="./README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9?style=for-the-badge"></a>
  <br>
  <a href="./README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9?style=for-the-badge"></a>
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-454545?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference（Xinference）是一個強大且通用的函式庫，適用於語言模型、語音識別與多模態模型。使用 Xinference，您可以只用一條指令部署自有模型或整合的尖端模型並將其提供為服務。研究者、開發者與資料科學家皆能充分發揮現代 AI 模型的能力。

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 加入我們的 Discord 社群！</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">加入我們的 Telegram 群組</a></i>
</div>

## 🔥 重點功能與更新
### 框架強化
- 原生 Agent 部署：Xinference 與 [Xagent](https://github.com/xorbitsai/xagent) 整合，支援動態規劃、工具使用與自動化多步驟推論，突破靜態 pipeline 的限制。
- 自動批次（Batching）：多個併發請求會自動合併以大幅提升吞吐量。 : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp)：Xinference 團隊維護的新一代 llama.cpp Python binding，支援連續批次並更適合生產環境。 : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- 分散式推論：模型可在多個 worker 之間分散執行： [#2877](https://github.com/xorbitsai/inference/pull/2877)
- vLLM 改進：在多個複本之間共享 KV-cache： [#2732](https://github.com/xorbitsai/inference/pull/2732)
### 新增模型
- 整合 VibeThinker 系列（[1.5B](https://huggingface.co/WeiboAI/VibeThinker-1.5B)、[3B](https://huggingface.co/WeiboAI/VibeThinker-3B)）： [#5085](https://github.com/xorbitsai/inference/pull/5085)
- 整合 Nex-N2 系列（[mini](https://huggingface.co/nex-agi/Nex-N2-mini)、[Pro](https://huggingface.co/nex-agi/Nex-N2-Pro)、[Pro-fp8](https://huggingface.co/nex-agi/Nex-N2-Pro-fp8)）： [#5094](https://github.com/xorbitsai/inference/pull/5094)
- 整合 [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR)： [#5103](https://github.com/xorbitsai/inference/pull/5103)
- 整合 [Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B)： [#5119](https://github.com/xorbitsai/inference/pull/5119)
- 整合 [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)： [#5010](https://github.com/xorbitsai/inference/pull/5010)
- 整合 jina-embeddings-v5 系列（[text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano)、[text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)、[omni-nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano)、[omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)）： [#5018](https://github.com/xorbitsai/inference/pull/5018)
- 整合 MiniCPM-V-4.6 系列（[MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6)、[MiniCPM-V-4.6-Thinking](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)）： [#5025](https://github.com/xorbitsai/inference/pull/5025)
- 整合 Tencent Hy-MT2 系列（[1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B)、[7B](https://huggingface.co/tencent/Hy-MT2-7B)、[30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B)）： [#5029](https://github.com/xorbitsai/inference/pull/5029)
### 整合項目
- [Xagent](https://github.com/xorbitsai/xagent)：企業級 Agent 平台，具規劃、記憶與工具整合。
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference)：LLMOps 平台，快速建立可視化與控制的應用。
- [FastGPT](https://github.com/labring/FastGPT)：基於 LLM 的知識平台，用於資料處理與模型呼叫。
- [RAGFlow](https://github.com/infiniflow/ragflow)：用於深度文件理解的開源 RAG 引擎。
- [MaxKB](https://github.com/1Panel-dev/MaxKB)：具有 RAG 整合的開源知識庫助理。

## 主要功能
🌟 模型部署更簡單：簡化 LLM、語音識別與多模態模型的上線流程。實驗與生產模型都能透過單一命令設定與部署。

⚡️ 方便使用的尖端模型：只需一條指令即可嘗試整合的最新模型。Xinference 提供對最先進開源模型的存取。

🖥 異構硬體支援：有效利用 GPU 與 CPU（例如透過 [ggml](https://github.com/ggerganov/ggml)），以加速推論。

⚙️ 彈性的 API 與介面：OpenAI 相容的 RESTful API（包含 Function Calling）、RPC、CLI、Web UI 等。

🌐 分散式部署：方便在多台設備與機器間分散推論工作。

🔌 第三方整合：支援與 [LangChain](https://python.langchain.com/docs/integrations/providers/xinference)、[LlamaIndex]、[Dify]、[Chatbox] 等整合。

## 為何選擇 Xinference
| 功能                                         | Xinference | FastChat | OpenLLM | RayLLM |
|----------------------------------------------|------------|----------|---------|--------|
| OpenAI 相容的 RESTful API                      | ✅         | ✅        | ✅       | ✅      |
| vLLM 整合                                    | ✅         | ✅        | ✅       | ✅      |
| 多種推論引擎（GGML、TensorRT）                 | ✅         | ❌        | ✅       | ✅      |
| 支援多種平台（CPU、Metal）                    | ✅         | ✅        | ❌       | ❌      |
| 多節點集群部署                                | ✅         | ❌        | ❌       | ✅      |
| 影像模型（文字→影像）                         | ✅         | ✅        | ❌       | ❌      |
| 文字嵌入模型                                  | ✅         | ❌        | ❌       | ❌      |
| 多模態模型                                    | ✅         | ❌        | ❌       | ❌      |
| 語音模型                                      | ✅         | ❌        | ❌       | ❌      |
| OpenAI 類 Function Calling 支援               | ✅         | ❌        | ❌       | ❌      |

## 使用 Xinference

- **Self-Hosting Xinference Community Edition**
  請依照 [快速上手指南](#getting-started) 在本地啟動 Xinference。詳細說明請參閱文件：https://inference.readthedocs.io/。

- **Xinference for Enterprise**
  若需企業功能與支援，請聯絡： mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry

## 保持更新

在 GitHub 為 Xinference 加星，以取得發布更新通知。

![star-us](../assets/stay_ahead.gif)

## 開始使用

* [文件](https://inference.readthedocs.io/en/latest/index.html)
* [內建模型](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [自訂模型](https://inference.readthedocs.io/en/latest/models/custom.html)
* [部署文件](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Docker

具有 NVIDIA GPU 的使用者可以使用 Xinference Docker 映像。請在安裝前確認系統已安裝 Docker 與 CUDA。

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

在已啟用 GPU 的 Kubernetes 叢集中，執行下列指令安裝：

```
# 新增 Helm 倉庫
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# 更新索引並檢視版本
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# 安裝 Xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

更多 K8s 選項請參閱文件。

### 快速上手

使用 pip 安裝 Xinference：

```bash
pip install "xinference[all]"
```

啟動本地實例：

```bash
$ xinference-local
```

啟動後可透過 Web UI、cURL、CLI 或 Python 客戶端來使用。

![web UI](../assets/screenshot.png)

## 參與專案

| 平台                                                                     | 目的                                     |
|--------------------------------------------------------------------------|-----------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)            | 報告錯誤與功能請求                       |
| [Discord](https://discord.gg/Xw9tszSkr5)                                 | 與其他使用者協作                         |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                               | 社群討論                                 |
| [Twitter](https://twitter.com/xorbitsio)                                 | 新聞與公告                               |

## 引用

若本專案對您有幫助，請以以下格式引用：

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

## 貢獻者

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## 星星歷史

[![Star History Chart](https://api.star-history.com/svg?repos=xorbitsai/inference&type=Date)](https://star-history.com/#xorbitsai/inference&Date)
