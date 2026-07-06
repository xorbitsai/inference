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
- Déploiement natif d'agents : Xinference s'intègre à [Xagent](https://github.com/xorbitsai/xagent) et permet la planification dynamique, l'utilisation d'outils et des inférences multi-étapes autonomes, dépassant les limites des pipelines statiques.
- Batching automatique : plusieurs requêtes simultanées sont automatiquement groupées pour augmenter significativement le débit. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp) : les nouvelles liaisons Python pour llama.cpp, maintenues par l'équipe Xinference, prennent en charge le batching continu et conviennent mieux à la production. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inférence distribuée : les modèles peuvent être exécutés entre plusieurs workers : [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Améliorations de vLLM : partage du KV-cache entre plusieurs réplicas : [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nouveaux modèles
- Intégration de la série VibeThinker ([1.5B](https://huggingface.co/WeiboAI/VibeThinker-1.5B), [3B](https://huggingface.co/WeiboAI/VibeThinker-3B)) : [#5085](https://github.com/xorbitsai/inference/pull/5085)
- Intégration de la série Nex-N2 ([mini](https://huggingface.co/nex-agi/Nex-N2-mini), [Pro](https://huggingface.co/nex-agi/Nex-N2-Pro), [Pro-fp8](https://huggingface.co/nex-agi/Nex-N2-Pro-fp8)) : [#5094](https://github.com/xorbitsai/inference/pull/5094)
- Intégration de [Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) : [#5103](https://github.com/xorbitsai/inference/pull/5103)
- Intégration de [Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) : [#5119](https://github.com/xorbitsai/inference/pull/5119)
- Intégration de [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) : [#5010](https://github.com/xorbitsai/inference/pull/5010)
- Intégration de la série jina-embeddings-v5 ([text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), [text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small), [omni-nano](https://huggingface.co/jinaai/jina-embeddings-v5-omni-nano), [omni-small](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small)) : [#5018](https://github.com/xorbitsai/inference/pull/5018)
- Intégration de la série MiniCPM-V-4.6 ([MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6), [MiniCPM-V-4.6-Thinking](https://huggingface.co/openbmb/MiniCPM-V-4.6-Thinking)) : [#5025](https://github.com/xorbitsai/inference/pull/5025)
- Intégration de la série Tencent Hy-MT2 ([1.8B](https://huggingface.co/tencent/Hy-MT2-1.8B), [7B](https://huggingface.co/tencent/Hy-MT2-7B), [30B-A3B](https://huggingface.co/tencent/Hy-MT2-30B-A3B)) : [#5029](https://github.com/xorbitsai/inference/pull/5029)
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
* [Exemples et tutoriels](https://inference.readthedocs.io/en/latest/examples/index.html)

### Jupyter Notebook

Le moyen le plus simple d'essayer Xinference est d'utiliser le [Jupyter Notebook Google Colab](https://colab.research.google.com/github/xorbitsai/inference/blob/main/examples/Xinference_Quick_Start.ipynb).

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

[![Star History Chart](https://api.star-history.com/svg?repos=xorbitsai/inference&type=Date)](https://star-history.com/#xorbitsai/inference&Date)
