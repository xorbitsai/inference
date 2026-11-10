# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference(Xorbits Inference)는 오픈 소스 분산 추론 플랫폼입니다.**

로컬 머신부터 분산 클러스터까지 이기종 하드웨어에서 모델을 배포하고 실행할 수 있습니다. Xinference는 언어, 비전, 임베딩, 리랭킹, 이미지, 오디오, 비디오, 월드 생성(실험적 기능)을 위한 300개 이상의 내장 모델을 제공합니다. 자체 모델도 등록할 수 있으며 Flexible 유형으로 전통적인 머신러닝 모델도 실행할 수 있습니다. Web UI에서 배포를 관리하고 OpenAI 호환 API, Python SDK 또는 CLI로 모델을 사용하세요.

[빠른 시작](#빠른-시작) · [주요 기능](#주요-기능) · [플랫폼 비교](#플랫폼-비교) · [자체 호스팅](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [문서](https://inference.readthedocs.io/en/latest/) · [모델 카탈로그](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [릴리스](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![GitHub 릴리스](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![PyPI 버전](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Docker 태그](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Python 버전](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![Python CI](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Apache 2.0 라이선스](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![GitHub 스타](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![GitHub 포크](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![기여자](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Docker 다운로드](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Discord 커뮤니티](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Telegram 커뮤니티](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![X에서 Xinference 팔로우](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**언어** · [English](../README.md) · [日本語](./README_ja_JP.md) · **한국어** · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## 빠른 시작

### 1. 서버 시작

처음 배포할 때 Xinference가 모델을 다운로드하고 엔진에 필요한 종속성을 설치합니다.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

[http://127.0.0.1:9997](http://127.0.0.1:9997)을 엽니다. 새 인스턴스에서는 서버 로그에 표시되는 설정 토큰으로 관리자 계정을 만든 다음 로그인합니다. [최초 설정](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. 채팅 모델 배포

**Launch Model**에서 `qwen3.5`를 검색하고 배포 화면을 엽니다. **Recommend configuration**을 클릭하면 하드웨어에 맞는 설정이 채워집니다. 추천 내용을 확인한 뒤 **Deploy**를 클릭합니다.

![추천 설정을 적용한 Qwen3.5 전체 배포 화면](../assets/guide-launch-recommend-auto.webp)

### 3. 첫 요청 보내기

**Running Models**에서 배포된 모델을 열어 Web UI로 프롬프트를 시험할 수 있습니다. 첫 인스턴스의 모델 UID는 `qwen3.5`입니다. 다른 UID가 표시되면 아래 요청의 `model`에 그 값을 사용하세요. API를 호출하려면 **API Key Management**에서 키를 만듭니다.

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

![Qwen3.5 채팅 화면과 응답 결과](../assets/guide-chat-en.webp)

## 주요 기능

| 영역 | Xinference 지원 기능 |
| --- | --- |
| 모델 카탈로그 | 언어, 비전, 임베딩, 리랭킹, 이미지, 오디오, 비디오, 월드 생성(실험적 기능)을 위한 300개 이상의 내장 모델을 제공합니다. Flexible 유형의 전통적인 머신러닝 모델을 포함해 자체 모델도 등록할 수 있습니다. [모델 카탈로그](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| 엔진과 하드웨어 | 지원되는 LLM에는 vLLM, SGLang, TRT-LLM, Transformers, llama.cpp 또는 MLX를 선택할 수 있습니다. CPU, GPU, Apple Silicon 및 일부 MLU 구성의 지원 범위는 엔진에 따라 다릅니다. [추론 백엔드](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [MLU 지원](https://github.com/xorbitsai/inference/pull/3693) |
| 분산 추론 | 한 대에서 실행하거나 워커와 레플리카를 추가할 수 있습니다. 지원 엔진을 사용하면 대형 모델을 여러 워커에서 실행할 수 있고, 호환 구성에서는 이기종 하드웨어에 프리필과 디코드를 나눌 수 있습니다. [분산 추론](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D 분리](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API와 도구 | OpenAI 및 Anthropic 호환 엔드포인트를 제공하며, 지원 모델에서 스트리밍과 도구 호출을 사용할 수 있습니다. Web UI, CLI 또는 Python SDK로 모델을 배포하고 관리합니다. [클라이언트 가이드](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [도구 호출](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| 성능과 운영 | Web UI에서 실행 중인 채팅 모델의 성능을 측정합니다. 지원되는 구성은 Xavier를 통한 접두사 캐시 공유, 일부 런타임 설정 변경, 레플리카별 순차 재로딩을 지원합니다. [성능 테스트](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [런타임 설정](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| 다국어 지원 | Web UI는 영어, 중국어 간체와 번체, 일본어, 한국어를 지원합니다. README는 10개 언어로 제공됩니다. |

## 플랫폼 비교

| 기능 | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Kubernetes 없이 단독 실행** | ![지원][compare-supported] | ![지원][compare-supported] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![지원][compare-supported] |
| **멀티 노드 추론** | ![지원][compare-supported] | ![지원][compare-supported] | ![지원][compare-supported] | ![지원][compare-supported] |
| **CPU / GPU / Apple Metal** | ![지원][compare-supported] | ![지원][compare-supported] | ![일부 지원 또는 통합 필요][compare-partial] | ![일부 지원 또는 통합 필요][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![지원][compare-supported] | ![지원][compare-supported] | ![일부 지원 또는 통합 필요][compare-partial] | ![일부 지원 또는 통합 필요][compare-partial] |
| **CUDA와 Metal 간 프리필 및 디코드** | ![지원][compare-supported] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![공식 문서에 지원 내용 없음][compare-undocumented] |
| **하나의 모델 ID에서 서로 다른 엔진의 레플리카 혼용** | ![지원][compare-supported] | ![일부 지원 또는 통합 필요][compare-partial] | ![일부 지원 또는 통합 필요][compare-partial] | ![일부 지원 또는 통합 필요][compare-partial] |
| **GPU 가중치를 재사용하며 레플리카 설정을 순차 변경** | ![지원][compare-supported] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![공식 문서에 지원 내용 없음][compare-undocumented] |
| 모델 카탈로그 | ![지원][compare-supported] 300+ | ![지원][compare-supported] | ![일부 지원 또는 통합 필요][compare-partial] | ![일부 지원 또는 통합 필요][compare-partial] |
| Web UI에서 모델 배포 | ![지원][compare-supported] | ![지원][compare-supported] | ![일부 지원 또는 통합 필요][compare-partial] | ![공식 문서에 지원 내용 없음][compare-undocumented] |
| 임베딩 및 리랭킹 | ![지원][compare-supported] | ![지원][compare-supported] | ![지원][compare-supported] | ![지원][compare-supported] |
| 이미지 및 비디오 생성 | ![지원][compare-supported] | ![지원][compare-supported] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![지원][compare-supported] |
| 음성 인식 및 합성 | ![지원][compare-supported] | ![지원][compare-supported] | ![공식 문서에 지원 내용 없음][compare-undocumented] | ![일부 지원 또는 통합 필요][compare-partial] |

![][compare-supported] 지원 · ![][compare-partial] 일부 지원 또는 통합 필요 · ![][compare-undocumented] 공식 문서에 지원 내용 없음

<details>
<summary>비교 범위와 출처</summary>

오픈 소스 릴리스의 공식 문서에 명시된 기능을 비교합니다. 엔진과 하드웨어 지원은 모델에 따라 다릅니다. CUDA와 Metal 간 P/D 분리에는 호환되는 모델과 엔진 조합이 필요합니다.

Xinference는 같은 모델 ID에서 서로 다른 엔진의 레플리카를 동시에 운영할 수 있습니다. LocalAI는 하나의 모델 이름에 우선순위가 있는 장애 조치 체인을 구성하고 첫 번째로 사용 가능한 대상으로 요청을 보냅니다. KServe와 Dynamo는 별도 배포 사이에서 요청을 라우팅할 수 있습니다. GPU 가중치 재사용은 지원되는 vLLM 및 SGLang 구성에 적용됩니다.

- **Xinference** · [모델 카탈로그](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [백엔드](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [분산 추론](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [CUDA와 Metal 간 교차 엔진 P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [레플리카 엔진 지정](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [순차 재설정](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [모델 및 배포 기능](https://localai.io/docs/features/index.html) · [엔진 및 하드웨어](https://localai.io/docs/reference/index.print.html) · [분산 모드](https://localai.io/docs/features/distributed-mode/) · [별칭 및 장애 조치](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [배포 및 하드웨어](https://kserve.github.io/website/docs/intro) · [내장 생성형 런타임](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Kubeflow 모델 카탈로그](https://www.kubeflow.org/docs/components/hub/installation/) · [Kubeflow Web UI](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [카나리 배포](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [배포, 하드웨어 및 엔진](https://docs.dynamo.nvidia.com/dynamo) · [배포 레시피](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [이미지, 비디오 및 음성 생성](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [임베딩 및 리랭킹](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [배포 간 라우팅](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## 최근 업데이트

[v4.2.0 릴리스 노트](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>최근 두 릴리스(v4.1, v4.2)에 추가된 내장 모델</summary>

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

## 모델 성능 테스트

**Performance Tests**를 열고 실행 중인 채팅 모델과 테스트할 동시 요청 수를 선택합니다. 처리량, 첫 토큰까지 걸리는 시간, 출력 토큰당 시간을 비교해 서빙 구성을 선택하세요. [성능 테스트 가이드](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![동시 요청 수에 따른 처리량과 지연 시간 그래프](../assets/guide-benchmark-en.webp)

*Qwen3.5-0.8B를 로컬에서 실행한 예입니다. 성능은 하드웨어, 엔진 및 워크로드에 따라 달라집니다.*

## Xinference 사용하기

| 하려는 작업 | 시작하기 |
| --- | --- |
| Linux, macOS 또는 Windows에 설치 | [한 줄 명령으로 설치](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [설치 가이드](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| 컨테이너 또는 Kubernetes에서 실행 | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| 자체 모델 등록 | [사용자 정의 모델](https://inference.readthedocs.io/en/latest/models/custom.html) |
| 애플리케이션 연결 | [OpenAI 호환 API](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**앱 및 프레임워크** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Xinference Enterprise를 자체 클라우드 또는 데이터 센터에 배포할 수 있습니다. Xinference Cloud는 관리형 전용 배포를 제공하고, Model API는 호스팅 모델 엔드포인트를 제공합니다. [Enterprise 개요](https://xinference.co/enterprise.html)

**비즈니스 문의** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## 프로젝트 참여

| 플랫폼 | 용도 |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | 버그를 보고하고 기능을 제안합니다. |
| [Discord](https://discord.gg/Xw9tszSkr5) | 질문하거나 다른 사용자와 의견을 나눕니다. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Xinference 커뮤니티와 대화합니다. |
| [X](https://x.com/Xinference_ai) | 프로젝트 소식과 릴리스를 확인합니다. |

## 인용

연구에 Xinference를 사용한 경우 다음 논문을 인용해 주세요. [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

Xinference에 기여하는 모든 분께 감사드립니다. 모델을 추가하거나 문서를 개선하고 버그를 수정하는 첫 Pull Request를 보내 주세요. [기여 가이드](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Xinference 기여자 아바타 모음](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
