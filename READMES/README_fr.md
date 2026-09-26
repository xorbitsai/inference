<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference : Simplifier le déploiement de modèles 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Auto-hébergement</a> ·
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
  <a href="../README.md"><img alt="English" src="https://img.shields.io/badge/English-d9d9d9?style=for-the-badge"></a>
  <a href="./README_ja_JP.md"><img alt="日本語" src="https://img.shields.io/badge/日本語-d9d9d9?style=for-the-badge"></a>
  <a href="./README_ko.md"><img alt="한국어" src="https://img.shields.io/badge/한국어-d9d9d9?style=for-the-badge"></a>
  <a href="./README_de.md"><img alt="Deutsch" src="https://img.shields.io/badge/Deutsch-d9d9d9?style=for-the-badge"></a>
  <a href="./README_fr.md"><img alt="Français" src="https://img.shields.io/badge/Français-454545?style=for-the-badge"></a>
  <br>
  <a href="./README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-d9d9d9?style=for-the-badge"></a>
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference (Xinference) est une bibliothèque puissante et polyvalente pour les modèles de langage, la reconnaissance vocale et les modèles multimodaux. Avec Xorbits Inference, vous pouvez déployer votre propre modèle ou des modèles avancés intégrés en une seule commande et les proposer en tant que service. Chercheurs, développeurs et data scientists peuvent exploiter pleinement les capacités des modèles IA modernes.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Rejoignez notre communauté Discord !</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Rejoignez notre groupe Telegram</a></i>
</div>

## 🔥 Sujets phares
### Améliorations du framework
- Xinference 3.0.0 est disponible avec des notes de migration et des changements incompatibles : [Notes de version](https://xinference.co/release_notes/v3.0.0.html)
- Déploiement natif d'agents : Xinference s'intègre à [Xagent](https://github.com/xorbitsai/xagent) et permet la planification dynamique, l'utilisation d'outils et des inférences multi-étapes autonomes, dépassant les limites des pipelines statiques.
- Batching automatique : plusieurs requêtes simultanées sont automatiquement groupées pour augmenter significativement le débit. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp) : les nouvelles liaisons Python pour llama.cpp, maintenues par l'équipe Xinference, prennent en charge le batching continu et conviennent mieux à la production. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inférence distribuée : les modèles peuvent être exécutés entre plusieurs workers : [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Améliorations de vLLM : partage du KV-cache entre plusieurs réplicas : [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nouveaux modèles
- Prise en charge intégrée de [TeleOCR](https://huggingface.co/StarDoc-AI/TeleOCR) : [#5583](https://github.com/xorbitsai/inference/pull/5583)
- Prise en charge intégrée de la série Ming-Image ([Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), [Design Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)) : [#5582](https://github.com/xorbitsai/inference/pull/5582)
- Prise en charge intégrée de [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) : [#5571](https://github.com/xorbitsai/inference/pull/5571)
- Prise en charge intégrée de [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) : [#5550](https://github.com/xorbitsai/inference/pull/5550)
- Prise en charge intégrée de la série Spark-X2.5 ([1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B), [4B](https://huggingface.co/XHToken/Spark-X2.5-4B), [1.7B Base](https://huggingface.co/XHToken/Spark-X2.5-1.7B-Base), [4B Base](https://huggingface.co/XHToken/Spark-X2.5-4B-Base)) : [#5538](https://github.com/xorbitsai/inference/pull/5538)
- Prise en charge intégrée de la série LingBot-World-V2 ([14B Fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast), [14B Pretrain](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-pretrain), [1.3B Fast](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast)) : [#5536](https://github.com/xorbitsai/inference/pull/5536)
- Prise en charge intégrée de la série Irodori-TTS v4.1 ([Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime), [Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)) : [#5527](https://github.com/xorbitsai/inference/pull/5527)
- Prise en charge intégrée de [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) : [#5526](https://github.com/xorbitsai/inference/pull/5526)
- Prise en charge intégrée de la série AuK ([AuK](https://huggingface.co/tencent/AuK), [AuK-Flash](https://huggingface.co/tencent/AuK-Flash)) : [#5525](https://github.com/xorbitsai/inference/pull/5525)
- Prise en charge intégrée de [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) : [#5506](https://github.com/xorbitsai/inference/pull/5506)
- Prise en charge intégrée de la série Fish Audio ([S1-mini](https://huggingface.co/fishaudio/s1-mini), [S2-Pro](https://huggingface.co/fishaudio/s2-pro)) : [#5490](https://github.com/xorbitsai/inference/pull/5490)
- Prise en charge intégrée de [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR) : [#5475](https://github.com/xorbitsai/inference/pull/5475)
- Prise en charge intégrée de [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) : [#5468](https://github.com/xorbitsai/inference/pull/5468)
- Prise en charge intégrée de la série JoyAI d'édition d'images ([Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers), [Edit Plus](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers)) : [#5458](https://github.com/xorbitsai/inference/pull/5458)
### Intégrations
- [Xagent](https://github.com/xorbitsai/xagent) : plateforme d'agents pour entreprises avec planification, mémoire et intégration d'outils.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference) : plateforme LLMOps pour construire rapidement des applications avec visualisation et contrôle.
- [FastGPT](https://github.com/labring/FastGPT) : plateforme de connaissances basée sur LLM pour le traitement des données et les appels de modèles.
- [RAGFlow](https://github.com/infiniflow/ragflow) : moteur RAG open-source pour la compréhension approfondie des documents.
- [MaxKB](https://github.com/1Panel-dev/MaxKB) : assistant de base de connaissances open-source avec intégration RAG.

## Principales fonctionnalités
🌟 Déploiement de modèles simplifié : simplifie la mise à disposition de LLMs, modèles de reconnaissance vocale et modèles multimodaux. Les modèles d'expérimentation et de production peuvent être configurés et déployés en une seule commande.

⚡️ Modèles de pointe faciles d'accès : testez des modèles intégrés en une seule commande. Xinference offre l'accès à des modèles open source de pointe.

🖥 Utilisation de matériel hétérogène : exploitez GPU et CPU efficacement (par ex. via [ggml](https://github.com/ggerganov/ggml)) pour accélérer l'inférence.

⚙️ API et interfaces flexibles : API RESTful compatible OpenAI (incl. Function Calling), RPC, CLI, Web UI, etc.

🌐 Déploiement distribué : facilite la distribution de l'inférence sur plusieurs dispositifs et machines.

🔌 Intégrations tierces : intégration avec [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox], etc.

## Pourquoi Xinference
| Fonction                                      | Xinference | FastChat | OpenLLM | RayLLM |
|-----------------------------------------------|------------|----------|---------|--------|
| API RESTful compatible OpenAI                  | ✅         | ✅        | ✅       | ✅      |
| Intégration vLLM                                | ✅         | ✅        | ✅       | ✅      |
| Divers moteurs d'inférence (GGML, TensorRT)     | ✅         | ❌        | ✅       | ✅      |
| Diverses plateformes (CPU, Metal)               | ✅         | ✅        | ❌       | ❌      |
| Déploiement en cluster multi-nœud               | ✅         | ❌        | ❌       | ✅      |
| Modèles d'images (Texte→Image)                  | ✅         | ✅        | ❌       | ❌      |
| Modèles d'embeddings texte                       | ✅         | ❌        | ❌       | ❌      |
| Modèles multimodaux                               | ✅         | ❌        | ❌       | ❌      |
| Modèles vocaux                                    | ✅         | ❌        | ❌       | ❌      |
| Fonctionnalité OpenAI (Function Calling)         | ✅         | ❌        | ❌       | ❌      |

## Utilisation de Xinference

- **Auto-hébergement Xinference Community Edition**
  Suivez le [guide de démarrage](#getting-started) pour lancer Xinference localement. Détails dans la documentation : https://inference.readthedocs.io/.

- **Xinference pour entreprises**
  Des fonctionnalités enterprise sont disponibles ; pour les demandes, contactez : mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry

## Restez à la pointe

Ajoutez une étoile à Xinference sur GitHub pour recevoir des notifications sur les nouvelles versions.

![star-us](../assets/stay_ahead.gif)

## Commencer

* [Documentation](https://inference.readthedocs.io/en/latest/index.html)
* [Modèles intégrés](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modèles personnalisés](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentation de déploiement](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Docker

Les utilisateurs de GPU NVIDIA peuvent utiliser l'[image Docker de Xinference](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html). Assurez-vous que Docker et CUDA sont installés avant de procéder à l'installation.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

Après avoir activé le support GPU dans le cluster Kubernetes, installez Xinference comme suit :

```
# Ajouter le dépôt
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# Mettre à jour l'index et vérifier la version
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Installer Xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

Pour plus d'options K8s, veuillez consulter la documentation.

### Démarrage rapide

Installez Xinference via pip :

```bash
pip install "xinference[all]"
```

Démarrez une instance locale avec :

```bash
$ xinference-local
```

Ensuite, vous pouvez utiliser l'interface Web, cURL, la CLI ou le client Python.

![web UI](../assets/screenshot.png)

## Participer

| Plateforme                                                                | Objectif                                      |
|---------------------------------------------------------------------------|-----------------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)           | Signaler des bugs et demander des fonctionnalités |
| [Discord](https://discord.gg/Xw9tszSkr5)                                 | Collaborer avec d'autres utilisateurs de Xinference |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                               | Échanger avec d'autres utilisateurs de Xinference |
| [Twitter](https://twitter.com/xorbitsio)                                 | Suivre les nouveautés et annonces              |

## Citation

Si ce projet vous a été utile, merci de citer :

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

## Contributeurs

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Historique des étoiles

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
