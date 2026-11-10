# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference（Xorbits Inference）是開放原始碼的分散式推論平台。**

從本機到分散式叢集，都能在異質硬體上部署和執行模型。Xinference 內建 300 多個模型，涵蓋語言、視覺、嵌入、重排序、影像、音訊、影片與世界生成（實驗性）。也可註冊自己的模型，並透過 Flexible 類型執行傳統機器學習模型。使用 Web UI 管理部署，並透過相容 OpenAI 的 API、Python SDK 或 CLI 呼叫模型。

[快速入門](#快速入門) · [主要功能](#主要功能) · [平台比較](#平台比較) · [自行託管](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [文件](https://inference.readthedocs.io/en/latest/) · [模型目錄](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [版本紀錄](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![GitHub 發行版本](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![PyPI 版本](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Docker 標籤](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Python 版本](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![Python CI](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Apache 2.0 授權](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![GitHub 星星數](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![GitHub Fork 數](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![貢獻者](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Docker 下載次數](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Discord 社群](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Telegram 社群](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![在 X 追蹤 Xinference](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**閱讀語言** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · **繁體中文** · [简体中文](./README_zh_CN.md)

## 快速入門

### 1. 啟動服務

首次部署時，Xinference 會下載模型，並視需要安裝所選引擎的相依套件。

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

開啟 [http://127.0.0.1:9997](http://127.0.0.1:9997)。首次使用時，以伺服器日誌中的設定權杖建立管理員帳號，再登入。 [初次設定](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. 部署聊天模型

在 **Launch Model（啟動模型）** 中搜尋 `qwen3.5` 並開啟部署表單。點選 **Recommend configuration（推薦配置）**，讓 Xinference 依據硬體填入建議設定。確認後點選 **Deploy（部署）**。

![套用建議設定後的完整 Qwen3.5 部署表單](../assets/guide-launch-recommend-auto.webp)

### 3. 送出第一個請求

在 **Running Models（執行中的模型）** 開啟已部署的模型，即可在 Web UI 試用。首個執行個體以 `qwen3.5` 作為模型 UID。若畫面顯示其他 UID，請將下方請求中的 `model` 換成該值。在 **API Key Management（API 金鑰管理）** 建立金鑰後，即可呼叫 API。

```bash
export XINFERENCE_API_KEY="<your-api-key>"
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

![Qwen3.5 聊天介面與完整回覆](../assets/guide-chat-en.webp)

## 主要功能

| 類別 | Xinference 支援的功能 |
| --- | --- |
| 模型目錄 | 內建 300 多個模型，涵蓋語言、視覺、嵌入、重排序、影像、音訊、影片與世界生成（實驗性）。支援註冊自訂模型，包括透過 Flexible 類型執行的傳統機器學習模型。 [模型目錄](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| 引擎與硬體 | 大型語言模型可依支援情況使用 vLLM、SGLang、TRT-LLM、Transformers、llama.cpp 或 MLX。CPU、GPU、Apple Silicon 與部分 MLU 組態的支援範圍依引擎而異。 [推論後端](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [MLU 支援](https://github.com/xorbitsai/inference/pull/3693) |
| 分散式推論 | 可從單機開始，加入 Worker 和模型副本，或讓大型模型透過支援的引擎跨 Worker 執行。相容的部署可在異質硬體上分離 Prefill 和 Decode。 [分散式推論](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D 分離](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API 與工具 | 提供相容 OpenAI 和 Anthropic 的端點，依模型支援串流和工具呼叫。透過 Web UI、CLI 或 Python SDK 部署和管理模型。 [用戶端指南](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [工具呼叫](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| 效能與維運 | 在 Web UI 測試執行中聊天模型的效能。相容的部署可透過 Xavier 共用前綴快取、調整部分執行階段設定，並逐一重新載入副本。 [效能測試](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [執行階段設定](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| 多語言 | Web UI 支援英語、簡體中文、繁體中文、日語和韓語。README 提供十種語言版本。 |

## 平台比較

| 功能 | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **單機啟動（無須 Kubernetes）** | ![支援][compare-supported] | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![支援][compare-supported] |
| **多節點推論** | ![支援][compare-supported] | ![支援][compare-supported] | ![支援][compare-supported] | ![支援][compare-supported] |
| **CPU / GPU / Apple Metal** | ![支援][compare-supported] | ![支援][compare-supported] | ![部分支援或需要整合][compare-partial] | ![部分支援或需要整合][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![支援][compare-supported] | ![支援][compare-supported] | ![部分支援或需要整合][compare-partial] | ![部分支援或需要整合][compare-partial] |
| **CUDA ↔ Metal 異質 P/D** | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![官方文件未列出支援][compare-undocumented] | ![官方文件未列出支援][compare-undocumented] |
| **同一模型 ID 混用不同引擎的副本** | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![部分支援或需要整合][compare-partial] | ![部分支援或需要整合][compare-partial] |
| **重複使用 GPU 權重並逐一更新副本設定** | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![官方文件未列出支援][compare-undocumented] | ![官方文件未列出支援][compare-undocumented] |
| 模型目錄 | ![支援][compare-supported] 300+ | ![支援][compare-supported] | ![部分支援或需要整合][compare-partial] | ![部分支援或需要整合][compare-partial] |
| 模型部署 Web UI | ![支援][compare-supported] | ![支援][compare-supported] | ![部分支援或需要整合][compare-partial] | ![官方文件未列出支援][compare-undocumented] |
| 嵌入與重排序 | ![支援][compare-supported] | ![支援][compare-supported] | ![支援][compare-supported] | ![支援][compare-supported] |
| 影像與影片生成 | ![支援][compare-supported] | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![支援][compare-supported] |
| 語音辨識與合成 | ![支援][compare-supported] | ![支援][compare-supported] | ![官方文件未列出支援][compare-undocumented] | ![部分支援或需要整合][compare-partial] |

![][compare-supported] 支援 · ![][compare-partial] 部分支援或需要整合 · ![][compare-undocumented] 官方文件未列出支援

<details>
<summary>比較範圍與來源</summary>

以開放原始碼版本的官方文件為準。引擎與硬體支援情況依模型而異。CUDA ↔ Metal 異質 P/D 需要相容的模型與引擎組合。

Xinference 可為同一模型 ID 下的新增副本指定引擎。KServe 和 Dynamo 可在獨立部署之間路由。GPU 權重重複使用適用於支援的 vLLM 和 SGLang 組態。

- **Xinference** · [模型目錄](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [推論後端](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [分散式推論](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [CUDA 與 Metal 跨引擎 P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [為副本指定引擎](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [逐一更新副本設定](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [模型與部署功能](https://localai.io/docs/features/index.html) · [引擎與硬體](https://localai.io/docs/reference/index.print.html) · [分散式模式](https://localai.io/docs/features/distributed-mode/) · [別名與容錯移轉](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [部署與硬體](https://kserve.github.io/website/docs/intro) · [內建生成式執行環境](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Kubeflow 模型目錄](https://www.kubeflow.org/docs/components/hub/installation/) · [Kubeflow Web UI](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [漸進式部署](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [部署、硬體與引擎](https://docs.dynamo.nvidia.com/dynamo) · [部署範例](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [影像、影片與語音生成](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [嵌入與重排序](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [跨部署路由](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## 近期更新

[閱讀 v4.2.0 版本說明](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>最近兩個版本新增的內建模型（v4.1、v4.2）</summary>

- [Audio8-TTS-Realistic-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b) · [#6003](https://github.com/xorbitsai/inference/pull/6003)
- [MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) · [#5916](https://github.com/xorbitsai/inference/pull/5916)
- [OvisOCR3](https://huggingface.co/ATH-MaaS/OvisOCR3) · [#5914](https://github.com/xorbitsai/inference/pull/5914)
- [Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) · [#5913](https://github.com/xorbitsai/inference/pull/5913)
- [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) · [#5792](https://github.com/xorbitsai/inference/pull/5792)
- FIBO 2.0 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)) · [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Ming-3.0 TTS ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) · [#5788](https://github.com/xorbitsai/inference/pull/5788)
- [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) · [#5764](https://github.com/xorbitsai/inference/pull/5764)
- [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [#5755](https://github.com/xorbitsai/inference/pull/5755)
- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) · [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Iris-3B ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)) · [#5686](https://github.com/xorbitsai/inference/pull/5686)
- MiniCPM-V-4.7 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)) · [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Qwen3.8-Flash-Next ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)) · [#5673](https://github.com/xorbitsai/inference/pull/5673)
- GLM-5.3 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)) · [#5666](https://github.com/xorbitsai/inference/pull/5666)
- [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) · [#5665](https://github.com/xorbitsai/inference/pull/5665)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [#5661](https://github.com/xorbitsai/inference/pull/5661)

</details>

## 模型效能測試

開啟 **Performance Tests（效能測試）**，選擇執行中的聊天模型，設定要測試的並行程度。比較吞吐量、首 Token 延遲與每個輸出 Token 的耗時，再選擇服務組態。 [效能測試指南](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![比較不同並行程度的吞吐量與延遲圖表](../assets/guide-benchmark-en.webp)

*圖中資料來自本機執行 Qwen3.5-0.8B 的一次測試。實際效能取決於硬體、引擎與負載。*

## 使用 Xinference

| 你想做什麼 | 從這裡開始 |
| --- | --- |
| 在 Linux、macOS 或 Windows 上安裝 | [單一指令安裝](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [安裝指南](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| 在容器或 Kubernetes 中執行 | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| 註冊自己的模型 | [自訂模型](https://inference.readthedocs.io/en/latest/models/custom.html) |
| 串接應用程式 | [相容 OpenAI 的 API](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**應用程式與開發框架** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Xinference Enterprise 可部署於自己的雲端環境或資料中心。Xinference Cloud 提供專用託管部署，Model API 提供託管模型端點。 [企業版總覽](https://xinference.co/enterprise.html)

**商務洽詢** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## 參與專案

| 平台 | 用途 |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | 回報問題並提出功能建議。 |
| [Discord](https://discord.gg/Xw9tszSkr5) | 提問並和其他使用者交流。 |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | 加入 Xinference 社群討論。 |
| [X](https://x.com/Xinference_ai) | 追蹤專案消息與版本發布。 |

## 引用

若在研究中使用 Xinference，請引用 [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

感謝每位參與 Xinference 開發的人。歡迎提交第一個 Pull Request，新增模型、改進文件或修正問題。 [貢獻指南](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Xinference 貢獻者頭像牆](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
