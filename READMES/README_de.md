<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: Modell-Serving leicht gemacht 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-Hosting</a> ·
  <a href="https://inference.readthedocs.io/">Dokumentation</a>
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
  <a href="./README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-454545?style=for-the-badge"></a>
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

Xorbits Inference (Xinference) ist eine leistungsfähige und vielseitige Bibliothek für Sprach-, Spracherkennungs- und multimodale Modelle. Mit Xorbits Inference können Sie Ihr eigenes Modell oder integrierte Spitzenmodelle mit nur einem Befehl bereitstellen und als Dienst veröffentlichen. Forschende, Entwickler und Data Scientists können so das Potenzial moderner KI-Modelle vollständig ausschöpfen.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Treten Sie unserer Discord-Community bei!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Treten Sie unserer Telegram-Gruppe bei!</a></i>
</div>

## 🔥 Aktuelle Highlights
### Verbesserungen im Framework
- Agent-native Bereitstellung: Xinference ist in [Xagent](https://github.com/xorbitsai/xagent) integriert und ermöglicht dynamische Planung, Tool-Nutzung und eigenständige mehrstufige Inferenz, wodurch die Grenzen statischer Pipelines überwunden werden.
- Automatisches Batching: Mehrere gleichzeitige Anfragen werden automatisch gebündelt, was den Durchsatz deutlich erhöht.: [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): Neue Python-Bindings für llama.cpp, verwaltet vom Xinference-Team, unterstützen fortlaufendes Batching und sind produktionsfreundlicher.: [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Verteilte Inferenz: Modelle können über Worker verteilt ausgeführt werden: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Verbesserungen für vLLM: Gemeinsame KV-Cache-Nutzung über mehrere Replikate: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Neue Modelle
- Eingebautes [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B): [#5010](https://github.com/xorbitsai/inference/pull/5010)
- Eingebaute jina-embeddings-v5 Serie (z. B. [text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), [text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)): [#5018](https://github.com/xorbitsai/inference/pull/5018)
- Eingebaute MiniCPM-V-4.6 Serie: [#5025](https://github.com/xorbitsai/inference/pull/5025)
- Eingebaute Tencent Hy-MT2 Serie (1.8B, 7B, 30B-A3B): [#5029](https://github.com/xorbitsai/inference/pull/5029)
- Eingebaute [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6): [#5033](https://github.com/xorbitsai/inference/pull/5033)
- Eingebaute [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2): [#5045](https://github.com/xorbitsai/inference/pull/5045)
- Eingebaute [DeepSeek V4]: [#4938](https://github.com/xorbitsai/inference/pull/4938)
- Eingebaute [MiniMax-M2.7]: [#4843](https://github.com/xorbitsai/inference/pull/4843)
### Integrationen
- [Xagent](https://github.com/xorbitsai/xagent): Enterprise-Agentenplattform mit Planung, Speicher und Tool-Integration.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): LLMOps-Plattform zur schnellen Anwendungsentwicklung mit Visualisierung und Bedienoberfläche.
- [FastGPT](https://github.com/labring/FastGPT): LLM-basierte Knowledge-Plattform für Datenverarbeitung und Modellaufrufe.
- [RAGFlow](https://github.com/infiniflow/ragflow): Open-Source-RAG-Engine für tiefes Dokumentenverständnis.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): Open-Source-Wissensdatenbank-Assistent mit RAG-Integration.

## Hauptfunktionen
🌟 **Modell-Serving leicht gemacht**: Vereinfachen Sie die Bereitstellung von LLMs, Spracherkennung und multimodalen Modellen. Tests und Produktionsmodelle lassen sich mit einem einzigen Befehl einrichten und deployen.

⚡️ **State-of-the-art-Modelle einfach nutzbar**: Probieren Sie integrierte Spitzenmodelle mit nur einem Befehl aus. Xinference bietet Zugang zu modernen Open-Source-Modellen.

🖥 **Heterogene Hardware-Unterstützung**: Nutzen Sie GPU und CPU effizient (z. B. über [ggml](https://github.com/ggerganov/ggml)), um Inferenz zu beschleunigen.

⚙️ **Flexible APIs und Schnittstellen**: OpenAI-kompatible RESTful API (inkl. Function Calling), RPC, CLI, Web UI und mehr.

🌐 **Verteiltes Deployment**: Einfache Verteilung von Inferenz über mehrere Geräte und Maschinen.

🔌 **Third-Party-Integrationen**: Nahtlose Integration mit [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox] u. a.

## Warum Xinference
| Funktion                                     | Xinference | FastChat | OpenLLM | RayLLM |
|----------------------------------------------|------------|----------|---------|--------|
| OpenAI-kompatible RESTful API                 | ✅         | ✅        | ✅       | ✅      |
| vLLM-Integration                              | ✅         | ✅        | ✅       | ✅      |
| Verschiedene Inferenz-Engines (GGML, TensorRT)| ✅         | ❌        | ✅       | ✅      |
| Verschiedene Plattformen (CPU, Metal)         | ✅         | ✅        | ❌       | ❌      |
| Multi-Node Cluster Deployment                 | ✅         | ❌        | ❌       | ✅      |
| Bildmodelle (Text→Bild)                       | ✅         | ✅        | ❌       | ❌      |
| Text-Embedding-Modelle                        | ✅         | ❌        | ❌       | ❌      |
| Multimodale Modelle                            | ✅         | ❌        | ❌       | ❌      |
| Sprachmodelle                                  | ✅         | ❌        | ❌       | ❌      |
| OpenAI-Funktionalität (Function Calling)      | ✅         | ❌        | ❌       | ❌      |

## Wie man Xinference verwendet

- **Self-Hosting Xinference Community Edition**
  Folgen Sie dem [Starter-Guide](#getting-started), um Xinference lokal zu starten. Details in der Dokumentation: https://inference.readthedocs.io/.

- **Xinference für Unternehmen**
  Es gibt zusätzliche Enterprise-Funktionen; für Anfragen kontaktieren Sie bitte: mailto:business@xprobe.io?subject=[GitHub]Business%20License%20Inquiry

## Bleiben Sie vorne dabei

Sternen Sie Xinference auf GitHub, um Release-Updates zu erhalten.

![star-us](../assets/stay_ahead.gif)

## Einstieg

* [Dokumentation](https://inference.readthedocs.io/en/latest/index.html)
* [Integrierte Modelle](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Custom-Modelle](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Deployment-Dokumentation](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)
* [Beispiele und Tutorials](https://inference.readthedocs.io/en/latest/examples/index.html)

### Jupyter Notebook

Die einfachste Möglichkeit, Xinference auszuprobieren, ist das [Google Colab Jupyter Notebook](https://colab.research.google.com/github/xorbitsai/inference/blob/main/examples/Xinference_Quick_Start.ipynb).

### Docker

NVIDIA-GPU-Benutzer können das [Xinference Docker Image](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) verwenden. Stellen Sie sicher, dass Docker und CUDA vor der Installation vorhanden sind.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

Nach Aktivierung von GPU im Kubernetes-Cluster installieren Sie wie folgt:

```
# Repository hinzufügen
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# Index aktualisieren und Version prüfen
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Xinference installieren
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

Weitere K8s-Optionen finden Sie in der Dokumentation.

### Quickstart

Installieren Sie Xinference per pip:

```bash
pip install "xinference[all]"
```

Starten Sie eine lokale Instanz mit:

```bash
$ xinference-local
```

Anschließend können Sie Web UI, cURL, CLI oder den Python-Client verwenden.

![web UI](../assets/screenshot.png)

## Mitmachen

| Plattform                                                                 | Zweck                                   |
|---------------------------------------------------------------------------|-----------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)             | Bug-Reports und Feature-Anfragen        |
| [Discord](https://discord.gg/Xw9tszSkr5)                                  | Zusammenarbeit mit anderen Anwendern    |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                | Diskussionen mit der Community          |
| [Twitter](https://twitter.com/xorbitsio)                                  | Neuigkeiten und Ankündigungen           |

## Zitation

Wenn dieses Projekt hilfreich war, zitieren Sie bitte wie folgt:

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

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Star-Verlauf

[![Star History Chart](https://api.star-history.com/svg?repos=xorbitsai/inference&type=Date)](https://star-history.com/#xorbitsai/inference&Date)
