<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: 모델 서빙을 쉽게 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-Hosting</a> ·
  <a href="https://inference.readthedocs.io/">문서</a>
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
  <a href="./README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-454545?style=for-the-badge"></a>
  <a href="./README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9?style=for-the-badge"></a>
  <a href="./README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-d9d9d9?style=for-the-badge"></a>
  <br>
  <a href="./README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9?style=for-the-badge"></a>
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference (Xinference)는 언어, 음성 인식, 멀티모달 모델을 위한 강력하고 다목적의 라이브러리입니다. Xinference를 사용하면 단 한 줄의 명령으로 자체 모델이나 통합된 최첨단 모델을 배포하고 서비스로 제공할 수 있습니다. 연구자, 개발자, 데이터 과학자 모두 최신 AI 모델의 기능을 최대한 활용할 수 있습니다.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 우리 Discord 커뮤니티에 참여하세요!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Telegram 그룹에 참여하기</a></i>
</div>

## 🔥 주요 하이라이트
### 프레임워크 개선
- Xinference 4.0.0은 다중 엔진 Xavier, 이기종 P/D 분리, 런타임 모델 재구성을 지원합니다: [릴리스 노트](https://github.com/xorbitsai/inference/releases/tag/v4.0.0)
- 에이전트 네이티브 배포: Xinference는 [Xagent](https://github.com/xorbitsai/xagent)와 통합되어 동적 플래닝, 도구 사용 및 자율적 다단계 추론을 지원하며 정적 파이프라인의 한계를 넘어섭니다.
- 자동 배칭: 여러 동시 요청을 자동으로 묶어 처리량을 크게 향상시킵니다. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): Xinference 팀이 관리하는 새로운 llama.cpp Python 바인딩은 연속 배칭을 지원하며 프로덕션에 더 적합합니다. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- 분산 추론: 모델을 여러 워커에 걸쳐 실행할 수 있습니다: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- vLLM 개선: 여러 복제본 간 KV 캐시 공유: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### 신규 모델
- [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) 기본 지원: [#5792](https://github.com/xorbitsai/inference/pull/5792)
- FIBO 2.0 시리즈 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)) 기본 지원: [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Ming-3.0 TTS 시리즈 ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) 기본 지원: [#5788](https://github.com/xorbitsai/inference/pull/5788)
- [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) 기본 지원: [#5764](https://github.com/xorbitsai/inference/pull/5764)
- [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) 기본 지원: [#5755](https://github.com/xorbitsai/inference/pull/5755)
- [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) 기본 지원: [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Iris-3B 시리즈 ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)) 기본 지원: [#5686](https://github.com/xorbitsai/inference/pull/5686)
- MiniCPM-V-4.7 시리즈 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)) 기본 지원: [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Qwen3.8-Flash-Next 시리즈 ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)) 기본 지원: [#5673](https://github.com/xorbitsai/inference/pull/5673)
- GLM-5.3 시리즈 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)) 기본 지원: [#5666](https://github.com/xorbitsai/inference/pull/5666)
- [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) 기본 지원: [#5665](https://github.com/xorbitsai/inference/pull/5665)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 기본 지원: [#5661](https://github.com/xorbitsai/inference/pull/5661)
- [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) 기본 지원: [#5644](https://github.com/xorbitsai/inference/pull/5644)
- [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) 기본 지원: [#5591](https://github.com/xorbitsai/inference/pull/5591)
- [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) 기본 지원: [#5588](https://github.com/xorbitsai/inference/pull/5588)
### 통합
- [Xagent](https://github.com/xorbitsai/xagent): 플래닝, 메모리, 툴 통합을 제공하는 엔터프라이즈 에이전트 플랫폼.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): 시각화와 제어가 가능한 LLMOps 플랫폼.
- [FastGPT](https://github.com/labring/FastGPT): 데이터 처리와 모델 호출을 위한 LLM 기반 지식 플랫폼.
- [RAGFlow](https://github.com/infiniflow/ragflow): 문서의 심층 이해를 위한 오픈소스 RAG 엔진.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): RAG 통합형 오픈소스 지식베이스 어시스턴트.

## 주요 기능
🌟 모델 배포 간소화: LLM, 음성 인식, 멀티모달 모델의 배포를 단순화합니다. 실험용 및 프로덕션 모델을 단일 명령으로 설정하고 배포할 수 있습니다.

⚡️ 최첨단 모델에 대한 쉬운 접근성: 통합된 최신 모델을 단 한 명령으로 시험해볼 수 있습니다. Xinference는 오픈소스 최첨단 모델에 대한 접근을 제공합니다.

🖥 이기종 하드웨어 지원: GPU와 CPU를 효율적으로 활용(예: [ggml](https://github.com/ggerganov/ggml))하여 추론 속도를 높입니다.

⚙️ 유연한 API 및 인터페이스: OpenAI 호환 RESTful API(함수 호출 포함), RPC, CLI, Web UI 등 다양한 인터페이스를 제공합니다.

🌐 분산 배포: 여러 장치와 머신에 걸친 분산 추론을 용이하게 합니다.

🔌 서드파티 통합: [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox] 등과의 통합을 지원합니다.

## 왜 Xinference인가
| 기능                                      | Xinference | FastChat | OpenLLM | RayLLM |
|-------------------------------------------|------------|----------|---------|--------|
| OpenAI 호환 RESTful API                     | ✅         | ✅        | ✅       | ✅      |
| vLLM 통합                                   | ✅         | ✅        | ✅       | ✅      |
| 다양한 추론 엔진(GGML, TensorRT)            | ✅         | ❌        | ✅       | ✅      |
| 다양한 플랫폼(CPU, Metal)                  | ✅         | ✅        | ❌       | ❌      |
| 멀티노드 클러스터 배포                      | ✅         | ❌        | ❌       | ✅      |
| 이미지 모델(텍스트→이미지)                  | ✅         | ✅        | ❌       | ❌      |
| 텍스트 임베딩 모델                          | ✅         | ❌        | ❌       | ❌      |
| 멀티모달 모델                               | ✅         | ❌        | ❌       | ❌      |
| 음성 모델                                   | ✅         | ❌        | ❌       | ❌      |
| OpenAI 기능(함수 호출)                      | ✅         | ❌        | ❌       | ❌      |

## Xinference 사용법

- **Self-Hosting Xinference Community Edition**
  스타터 가이드를 따라 로컬에서 Xinference를 시작하세요. 자세한 내용은 문서(https://inference.readthedocs.io/) 참조.

- **기업용 Xinference**
  **Xinference Enterprise** 는 프로덕션 환경에서 모델을 운영하는 팀을 위한 상용 버전입니다. 자체 클라우드나 데이터 센터에 배포하거나 **Xinference Cloud** 의 관리형 전용 배포를 선택할 수 있습니다. 호스팅되는 **Model API** 를 통해 추론 인프라를 관리하지 않고도 OpenAI 호환 API로 지원 모델에 접근할 수 있습니다. 자세한 내용은 [xinference.co](https://xinference.co) 에서 확인하세요.
  기업 관련 문의는 [이메일](mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry)로 연락해 주세요.

## 최신 상태 유지

GitHub에서 Xinference에 별을 눌러 릴리스 알림을 받으세요.

![star-us](../assets/stay_ahead.gif)

## 시작하기

* [문서](https://inference.readthedocs.io/en/latest/index.html)
* [내장 모델](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [커스텀 모델](https://inference.readthedocs.io/en/latest/models/custom.html)
* [배포 문서](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Quickstart

**pip**

```bash
pip install xinference
xinference-local
```

**원커맨드 설치**

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
```

[설치 가이드](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [사용 가이드](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally)

### Docker

NVIDIA GPU 사용자는 [Xinference Docker 이미지](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html)를 사용할 수 있습니다. 설치 전에 Docker 및 CUDA가 설치되어 있는지 확인하세요.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

GPU가 활성화된 Kubernetes 클러스터에서 다음과 같이 설치하세요:

```
# 레포지토리 추가
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# 인덱스 업데이트 및 버전 확인
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Xinference 설치
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

자세한 K8s 옵션은 문서를 참조하세요.

![web UI](../assets/screenshot.png)

## 기여하기

| 플랫폼                                                                    | 목적                                  |
|---------------------------------------------------------------------------|---------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)            | 버그 리포트 및 기능 요청               |
| [Discord](https://discord.gg/Xw9tszSkr5)                                 | 다른 사용자와 협업                     |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                               | 커뮤니티 토론                          |
| [Twitter](https://twitter.com/xorbitsio)                                 | 공지 및 업데이트                       |

## 인용

이 프로젝트가 유용했다면 다음과 같이 인용해 주세요:

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

## 기여자

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## 스타 히스토리

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
