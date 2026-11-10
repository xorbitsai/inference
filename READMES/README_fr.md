# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference (Xorbits Inference) est une plateforme open source d’inférence distribuée.**

Déployez et exécutez des modèles sur du matériel hétérogène, d’une machine locale à un cluster distribué. Xinference intègre plus de 300 modèles pour le langage, la vision, les embeddings, le reclassement, les images, l’audio, la vidéo et la génération de mondes (expérimentale). Enregistrez vos propres modèles, y compris des modèles de ML classique avec le type Flexible. Gérez les déploiements dans l’interface Web et utilisez les modèles via les API compatibles OpenAI, le SDK Python ou la CLI.

[Démarrage rapide](#démarrage-rapide) · [Fonctionnalités](#fonctionnalités) · [Comparaison des plateformes](#comparaison-des-plateformes) · [Auto-hébergement](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [Documentation](https://inference.readthedocs.io/en/latest/) · [Catalogue de modèles](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Versions](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![Version GitHub](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![Version PyPI](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Tag Docker](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Versions de Python](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![CI Python](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Licence Apache 2.0](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Étoiles GitHub](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![Forks GitHub](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Contributeurs](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Téléchargements Docker](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Communauté Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Communauté Telegram](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![Suivre Xinference sur X](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**Langues** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · **Français** · [Español](./README_es.md) · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## Démarrage rapide

### 1. Démarrer le serveur

Au premier lancement, Xinference télécharge le modèle et installe les dépendances nécessaires au moteur.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

Ouvrez [http://127.0.0.1:9997](http://127.0.0.1:9997). Sur une nouvelle instance, utilisez le jeton de configuration affiché dans les journaux du serveur pour créer un compte administrateur, puis connectez-vous. [Configuration initiale](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. Déployer un modèle de chat

Dans **Launch Model**, recherchez `qwen3.5` et ouvrez son formulaire de déploiement. Cliquez sur **Recommend configuration** pour obtenir une configuration adaptée à votre matériel. Vérifiez-la, puis cliquez sur **Deploy**.

![Formulaire complet de Qwen3.5 après application de la configuration recommandée](../assets/guide-launch-recommend-auto.webp)

### 3. Envoyer une première requête

Ouvrez le modèle dans **Running Models** pour tester une requête dans l’interface Web. La première instance utilise `qwen3.5` comme UID de modèle. Si votre instance affiche un autre UID, indiquez-le dans le champ `model` ci-dessous. Créez une clé dans **API Key Management** pour appeler l’API.

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

![Interface de chat de Qwen3.5 avec une réponse complète](../assets/guide-chat-en.webp)

## Fonctionnalités

| Domaine | Fonctionnalités de Xinference |
| --- | --- |
| Catalogue de modèles | Plus de 300 modèles intégrés couvrent le langage, la vision, les embeddings, le reclassement, les images, l’audio, la vidéo et la génération de mondes (expérimentale). Enregistrez vos propres modèles, y compris des modèles de ML classique via le type Flexible. [Catalogue de modèles](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| Moteurs et matériel | Selon le LLM, choisissez vLLM, SGLang, TRT-LLM, Transformers, llama.cpp ou MLX. La prise en charge des CPU, GPU, Apple Silicon et de certaines configurations MLU varie selon le moteur. [Moteurs d’inférence](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Prise en charge des MLU](https://github.com/xorbitsai/inference/pull/3693) |
| Inférence distribuée | Exécutez les modèles sur une machine, ajoutez des workers et des répliques ou répartissez un grand modèle entre plusieurs workers avec un moteur compatible. Les déploiements compatibles séparent le préremplissage et le décodage sur du matériel hétérogène. [Inférence distribuée](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Séparation P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API et outils | Les points de terminaison sont compatibles avec OpenAI et Anthropic. Le streaming et l’appel d’outils dépendent du modèle. Déployez et gérez les modèles via l’interface Web, la CLI ou le SDK Python. [Guide client](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [Appel d’outils](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| Performances et exploitation | Testez les modèles de chat en service dans l’interface Web. Les déploiements compatibles permettent de partager les caches de préfixes avec Xavier, d’ajuster certains paramètres et de recharger les répliques une par une. [Tests de performance](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [Configuration à chaud](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| Langues | L’interface Web est disponible en anglais, chinois simplifié, chinois traditionnel, japonais et coréen. Le README est disponible en dix langues. |

## Comparaison des plateformes

| Fonctionnalité | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Exécution autonome sans Kubernetes** | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Pris en charge][compare-supported] |
| **Inférence multinœud** | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] |
| **CPU / GPU / Apple Metal** | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Partiel ou via une intégration][compare-partial] | ![Partiel ou via une intégration][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Partiel ou via une intégration][compare-partial] | ![Partiel ou via une intégration][compare-partial] |
| **Préremplissage/décodage entre CUDA et Metal** | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Prise en charge non documentée][compare-undocumented] | ![Prise en charge non documentée][compare-undocumented] |
| **Plusieurs moteurs sous un même ID de modèle** | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Partiel ou via une intégration][compare-partial] | ![Partiel ou via une intégration][compare-partial] |
| **Reconfiguration progressive avec réutilisation des poids en GPU** | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Prise en charge non documentée][compare-undocumented] | ![Prise en charge non documentée][compare-undocumented] |
| Catalogue de modèles | ![Pris en charge][compare-supported] 300+ | ![Pris en charge][compare-supported] | ![Partiel ou via une intégration][compare-partial] | ![Partiel ou via une intégration][compare-partial] |
| Déploiement de modèles via l’interface Web | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Partiel ou via une intégration][compare-partial] | ![Prise en charge non documentée][compare-undocumented] |
| Embeddings et reclassement | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] |
| Génération d’images et de vidéos | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Pris en charge][compare-supported] |
| Reconnaissance et synthèse vocales | ![Pris en charge][compare-supported] | ![Pris en charge][compare-supported] | ![Prise en charge non documentée][compare-undocumented] | ![Partiel ou via une intégration][compare-partial] |

![][compare-supported] Pris en charge · ![][compare-partial] Partiel ou via une intégration · ![][compare-undocumented] Prise en charge non documentée

<details>
<summary>Périmètre et sources</summary>

La comparaison s’appuie sur les fonctionnalités documentées des versions open source. Les moteurs et le matériel pris en charge varient selon le modèle. La séparation CUDA ↔ Metal requiert un modèle et une paire de moteurs compatibles.

Xinference permet de choisir le moteur des nouvelles répliques sous le même ID de modèle. KServe et Dynamo peuvent acheminer les requêtes entre plusieurs déploiements. La réutilisation des poids en GPU s’applique aux configurations vLLM et SGLang compatibles.

- **Xinference** · [Catalogue](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Moteurs](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Inférence distribuée](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D entre moteurs CUDA et Metal](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [Choix du moteur des répliques](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [Reconfiguration progressive](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [Modèles et déploiement](https://localai.io/docs/features/index.html) · [Moteurs et matériel](https://localai.io/docs/reference/index.print.html) · [Mode distribué](https://localai.io/docs/features/distributed-mode/) · [Alias et basculement](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [Déploiement et matériel](https://kserve.github.io/website/docs/intro) · [Moteurs génératifs intégrés](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Catalogue Kubeflow](https://www.kubeflow.org/docs/components/hub/installation/) · [Interface Web Kubeflow](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [Déploiements canary](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [Déploiement, matériel et moteurs](https://docs.dynamo.nvidia.com/dynamo) · [Recettes de déploiement](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [Génération d’images, de vidéos et de voix](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [Embeddings et reclassement](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [Routage entre déploiements](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## Nouveautés

[Notes de version v4.2.0](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>Modèles intégrés lors des deux dernières versions (v4.1 et v4.2)</summary>

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

## Tests de performance

Dans **Performance Tests**, choisissez un modèle de chat en service et les niveaux de concurrence à tester. Comparez le débit, le délai jusqu’au premier token et le temps par token généré avant de choisir une configuration. [Guide des tests de performance](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![Graphiques de débit et de latence à plusieurs niveaux de concurrence](../assets/guide-benchmark-en.webp)

*Exemple obtenu avec Qwen3.5-0.8B en local. Les résultats dépendent du matériel, du moteur et de la charge.*

## Utiliser Xinference

| Objectif | Par où commencer |
| --- | --- |
| Installer sur Linux, macOS ou Windows | [Installation en une commande](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guide d’installation](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| Exécuter dans un conteneur ou sur Kubernetes | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| Enregistrer un modèle personnalisé | [Modèles personnalisés](https://inference.readthedocs.io/en/latest/models/custom.html) |
| Connecter une application | [API compatibles OpenAI](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**Applications et frameworks** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Déployez Xinference Enterprise dans votre propre cloud ou centre de données. Xinference Cloud propose des déploiements dédiés et gérés, et la Model API fournit des points de terminaison hébergés. [Présentation de l’offre Enterprise](https://xinference.co/enterprise.html)

**Contact commercial** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## Participer au projet

| Plateforme | Usage |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | Signaler des problèmes et proposer des fonctionnalités. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Poser des questions et échanger avec d’autres utilisateurs. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Discuter avec la communauté Xinference. |
| [X](https://x.com/Xinference_ai) | Suivre l’actualité et les nouvelles versions. |

## Citation

Si vous utilisez Xinference dans vos travaux de recherche, veuillez citer [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

Merci à toutes les personnes qui contribuent à Xinference. Ouvrez votre première pull request pour ajouter un modèle, améliorer la documentation ou corriger un problème. [Guide de contribution](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Mosaïque des avatars des contributeurs de Xinference](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
