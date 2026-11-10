# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference (Xorbits Inference) ist eine quelloffene Plattform für verteilte Inferenz.**

Modelle lassen sich auf heterogener Hardware bereitstellen und ausführen, vom einzelnen Rechner bis zum verteilten Cluster. Xinference enthält über 300 integrierte Modelle für Sprache, Bildverständnis, Embeddings, Reranking, Bild, Audio, Video und experimentelle Weltgenerierung. Registrieren Sie eigene Modelle, auch für klassisches maschinelles Lernen über den Typ Flexible. Verwalten Sie Bereitstellungen in der Weboberfläche und greifen Sie über OpenAI-kompatible APIs, das Python-SDK oder die CLI auf Modelle zu.

[Schnellstart](#schnellstart) · [Funktionen](#funktionen) · [Plattformvergleich](#plattformvergleich) · [Selbst hosten](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [Dokumentation](https://inference.readthedocs.io/en/latest/) · [Modellkatalog](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Releases](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![GitHub-Release](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![PyPI-Version](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Docker-Tag](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Python-Versionen](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![Python-CI](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Apache-2.0-Lizenz](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![GitHub-Sterne](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![GitHub-Forks](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Mitwirkende](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Docker-Downloads](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Discord-Community](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Telegram-Community](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![Xinference auf X folgen](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**Sprachen** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · **Deutsch** · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## Schnellstart

### 1. Server starten

Beim ersten Start lädt Xinference das Modell herunter und installiert die benötigten Abhängigkeiten der Engine.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

Öffnen Sie [http://127.0.0.1:9997](http://127.0.0.1:9997). Bei einer neuen Instanz erstellen Sie mit dem Setup-Token aus dem Serverprotokoll ein Administratorkonto und melden sich an. [Ersteinrichtung](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. Chatmodell bereitstellen

Suchen Sie unter **Launch Model** nach `qwen3.5` und öffnen Sie das Bereitstellungsformular. Klicken Sie auf **Recommend configuration**, um Einstellungen für Ihre Hardware zu erhalten. Prüfen Sie den Vorschlag und klicken Sie auf **Deploy**.

![Vollständiges Bereitstellungsformular für Qwen3.5 nach Übernahme der empfohlenen Konfiguration](../assets/guide-launch-recommend-auto.webp)

### 3. Erste Anfrage senden

Öffnen Sie das Modell unter **Running Models**, um eine Anfrage in der Weboberfläche zu testen. Die erste Instanz verwendet `qwen3.5` als Modell-UID. Falls eine andere UID angezeigt wird, tragen Sie diese unten unter `model` ein. Erstellen Sie unter **API Key Management** einen Schlüssel für den API-Aufruf.

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

![Chatoberfläche mit einer fertigen Antwort von Qwen3.5](../assets/guide-chat-en.webp)

## Funktionen

| Bereich | Funktionsumfang von Xinference |
| --- | --- |
| Modellkatalog | Über 300 integrierte Modelle für Sprache, Bildverständnis, Embeddings, Reranking, Bild, Audio, Video und experimentelle Weltgenerierung. Registrieren Sie eigene Modelle, auch für klassisches ML über den Typ Flexible. [Modellkatalog](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| Engines und Hardware | Je nach LLM können Sie vLLM, SGLang, TRT-LLM, Transformers, llama.cpp oder MLX wählen. Unterstützung für CPUs, GPUs, Apple Silicon und ausgewählte MLU-Konfigurationen hängt von der Engine ab. [Inferenz-Backends](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [MLU-Unterstützung](https://github.com/xorbitsai/inference/pull/3693) |
| Verteilte Inferenz | Nutzen Sie einen einzelnen Rechner, fügen Sie Worker und Replikate hinzu oder verteilen Sie ein großes Modell mit einer geeigneten Engine über mehrere Worker. Kompatible Bereitstellungen trennen Prefill und Decode auf heterogener Hardware. [Verteilte Inferenz](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D-Trennung](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| APIs und Werkzeuge | OpenAI- und Anthropic-kompatible Endpunkte mit Streaming und Tool-Aufrufen, soweit unterstützt. Stellen Sie Modelle über Weboberfläche, CLI oder Python-SDK bereit und verwalten Sie sie dort. [Client-Dokumentation](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [Tool-Aufrufe](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| Leistung und Betrieb | Testen Sie laufende Chatmodelle in der Weboberfläche. Bei unterstützten Konfigurationen können Replikate über Xavier Präfix-Caches teilen, ausgewählte Laufzeitparameter ändern und einzeln neu geladen werden. [Leistungstests](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [Laufzeitkonfiguration](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| Sprachen | Die Weboberfläche ist auf Englisch, vereinfachtem Chinesisch, traditionellem Chinesisch, Japanisch und Koreanisch verfügbar. Das README gibt es in zehn Sprachen. |

## Plattformvergleich

| Funktion | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Einzelbetrieb ohne Kubernetes** | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Unterstützt][compare-supported] |
| **Inferenz über mehrere Knoten** | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] |
| **CPU / GPU / Apple Metal** | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Teilweise oder per Integration][compare-partial] | ![Teilweise oder per Integration][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Teilweise oder per Integration][compare-partial] | ![Teilweise oder per Integration][compare-partial] |
| **Prefill und Decode über CUDA und Metal** | ![Unterstützt][compare-supported] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Keine dokumentierte Unterstützung][compare-undocumented] |
| **Unterschiedliche Engines unter derselben Modell-ID** | ![Unterstützt][compare-supported] | ![Teilweise oder per Integration][compare-partial] | ![Teilweise oder per Integration][compare-partial] | ![Teilweise oder per Integration][compare-partial] |
| **Schrittweise Neukonfiguration mit Wiederverwendung der GPU-Gewichte** | ![Unterstützt][compare-supported] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Keine dokumentierte Unterstützung][compare-undocumented] |
| Modellkatalog | ![Unterstützt][compare-supported] 300+ | ![Unterstützt][compare-supported] | ![Teilweise oder per Integration][compare-partial] | ![Teilweise oder per Integration][compare-partial] |
| Modellbereitstellung per Weboberfläche | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Teilweise oder per Integration][compare-partial] | ![Keine dokumentierte Unterstützung][compare-undocumented] |
| Embeddings und Reranking | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] |
| Bild- und Videogenerierung | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Unterstützt][compare-supported] |
| Spracherkennung und Sprachsynthese | ![Unterstützt][compare-supported] | ![Unterstützt][compare-supported] | ![Keine dokumentierte Unterstützung][compare-undocumented] | ![Teilweise oder per Integration][compare-partial] |

![][compare-supported] Unterstützt · ![][compare-partial] Teilweise oder per Integration · ![][compare-undocumented] Keine dokumentierte Unterstützung

<details>
<summary>Umfang und Quellen</summary>

Der Vergleich stützt sich auf die dokumentierten Funktionen der Open-Source-Releases. Die Unterstützung von Engines und Hardware variiert je nach Modell. Prefill und Decode über CUDA und Metal setzen ein kompatibles Modell und ein geeignetes Engine-Paar voraus.

Xinference kann Replikate mit verschiedenen Engines gleichzeitig unter derselben Modell-ID betreiben. LocalAI verwendet unter einem Modellnamen eine geordnete Ausweichkette und leitet Anfragen an das erste verfügbare Ziel weiter. KServe und Dynamo können Anfragen über getrennte Bereitstellungen verteilen. Die Wiederverwendung von GPU-Gewichten gilt für kompatible vLLM- und SGLang-Konfigurationen.

- **Xinference** · [Modellkatalog](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Backends](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Verteilte Inferenz](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D über CUDA und Metal](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [Engine-Wahl für Replikate](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [Schrittweise Neukonfiguration](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [Modelle und Bereitstellung](https://localai.io/docs/features/index.html) · [Engines und Hardware](https://localai.io/docs/reference/index.print.html) · [Verteilter Betrieb](https://localai.io/docs/features/distributed-mode/) · [Aliasse und Ausweichmodelle](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [Bereitstellung und Hardware](https://kserve.github.io/website/docs/intro) · [Integrierte generative Laufzeiten](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Kubeflow-Modellkatalog](https://www.kubeflow.org/docs/components/hub/installation/) · [Kubeflow-Weboberfläche](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [Canary-Bereitstellungen](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [Bereitstellung, Hardware und Engines](https://docs.dynamo.nvidia.com/dynamo) · [Bereitstellungsvorlagen](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [Bild-, Video- und Sprachgenerierung](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [Embeddings und Reranking](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [Routing über Bereitstellungen](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## Neuerungen

[Versionshinweise zu v4.2.0](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>Neue integrierte Modelle der letzten beiden Releases (v4.1 und v4.2)</summary>

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

## Leistungstests

Öffnen Sie **Performance Tests**, wählen Sie ein laufendes Chatmodell und legen Sie fest, wie viele Anfragen gleichzeitig getestet werden sollen. Vergleichen Sie Durchsatz, Zeit bis zum ersten Token und Zeit je Ausgabetoken. [Anleitung zu Leistungstests](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![Durchsatz und Latenz bei unterschiedlich vielen gleichzeitigen Anfragen](../assets/guide-benchmark-en.webp)

*Beispiel eines lokalen Laufs mit Qwen3.5-0.8B. Die Leistung hängt von Hardware, Engine und Auslastung ab.*

## Xinference verwenden

| Vorhaben | Einstieg |
| --- | --- |
| Unter Linux, macOS oder Windows installieren | [Installation mit einem Befehl](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Installationsanleitung](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| Im Container oder auf Kubernetes ausführen | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| Ein eigenes Modell registrieren | [Benutzerdefinierte Modelle](https://inference.readthedocs.io/en/latest/models/custom.html) |
| Eine Anwendung anbinden | [OpenAI-kompatible APIs](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**Anwendungen und Frameworks** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Betreiben Sie Xinference Enterprise in Ihrer eigenen Cloud oder Ihrem Rechenzentrum. Xinference Cloud bietet verwaltete dedizierte Bereitstellungen. Die Model API stellt gehostete Modellendpunkte bereit. [Enterprise-Übersicht](https://xinference.co/enterprise.html)

**Geschäftliche Anfragen** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## Mitmachen

| Plattform | Zweck |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | Fehler melden und Funktionen vorschlagen. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Fragen stellen und sich mit anderen Nutzern austauschen. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Mit der Xinference-Community chatten. |
| [X](https://x.com/Xinference_ai) | Neuigkeiten und Releases verfolgen. |

## Zitieren

Wenn Sie Xinference für Ihre Forschung verwenden, zitieren Sie bitte [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

## Mitwirkende

Vielen Dank an alle, die zu Xinference beitragen. Öffnen Sie Ihren ersten Pull Request, um ein Modell hinzuzufügen, die Dokumentation zu verbessern oder einen Fehler zu beheben. [Leitfaden für Beiträge](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Avatare der Mitwirkenden an Xinference](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
