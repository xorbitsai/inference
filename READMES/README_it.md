# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference (Xorbits Inference) è una piattaforma open source per l’inferenza distribuita.**

Distribuisci ed esegui modelli su hardware eterogeneo, da una macchina locale a un cluster distribuito. Xinference include più di 300 modelli integrati per linguaggio, visione, embedding, reranking, immagini, audio, video e generazione di mondi (sperimentale). Registra modelli personalizzati, anche di machine learning tradizionale tramite il tipo Flexible. Gestisci le distribuzioni dall’interfaccia web e usa i modelli tramite API compatibili con OpenAI, SDK Python o CLI.

[Avvio rapido](#avvio-rapido) · [Funzionalità principali](#funzionalità-principali) · [Confronto tra piattaforme](#confronto-tra-piattaforme) · [Self-hosting](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [Documentazione](https://inference.readthedocs.io/en/latest/) · [Catalogo dei modelli](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Versioni](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![Versione GitHub](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![Versione PyPI](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Tag Docker](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Versioni di Python](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![CI Python](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Licenza Apache 2.0](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Stelle GitHub](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![Fork GitHub](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Collaboratori](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Download Docker](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Comunità Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Comunità Telegram](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![Segui Xinference su X](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**Lingue** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · **Italiano** · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## Avvio rapido

### 1. Avviare il server

Al primo avvio, Xinference scarica il modello e installa le dipendenze necessarie per il motore.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

Apri [http://127.0.0.1:9997](http://127.0.0.1:9997). Su una nuova istanza, usa il token di configurazione mostrato nei log del server per creare un account amministratore. Poi accedi. [Configurazione iniziale](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. Distribuire un modello di chat

In **Launch Model**, cerca `qwen3.5` e apri il modulo di distribuzione. Seleziona **Recommend configuration** per ottenere impostazioni adatte al tuo hardware. Controllale e fai clic su **Deploy**.

![Modulo di distribuzione completo di Qwen3.5 con la configurazione consigliata](../assets/guide-launch-recommend-auto.webp)

### 3. Inviare la prima richiesta

Apri il modello distribuito in **Running Models** per provarlo nell’interfaccia web. La prima istanza usa `qwen3.5` come UID del modello. Se l’istanza mostra un UID diverso, usalo nel campo `model` della richiesta. Crea una chiave in **API Key Management** per chiamare l’API.

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

![Interfaccia di chat di Qwen3.5 con una risposta completa](../assets/guide-chat-en.webp)

## Funzionalità principali

| Area | Funzionalità di Xinference |
| --- | --- |
| Catalogo dei modelli | Più di 300 modelli integrati per linguaggio, visione, embedding, reranking, immagini, audio, video e generazione di mondi (sperimentale). Registra modelli personalizzati, anche di machine learning tradizionale tramite il tipo Flexible. [Catalogo dei modelli](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| Motori e hardware | A seconda del modello, scegli vLLM, SGLang, TRT-LLM, Transformers, llama.cpp o MLX. Il supporto per CPU, GPU, Apple Silicon e alcune configurazioni MLU dipende dal motore. [Backend di inferenza](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Supporto MLU](https://github.com/xorbitsai/inference/pull/3693) |
| Inferenza distribuita | Usa una sola macchina, aggiungi worker e repliche oppure esegui un modello di grandi dimensioni su più worker con un motore compatibile. Le configurazioni compatibili separano prefill e decodifica su hardware eterogeneo. [Inferenza distribuita](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Separazione P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API e strumenti | Endpoint compatibili con OpenAI e Anthropic, con streaming e chiamate a strumenti dove supportati. Distribuisci e gestisci i modelli tramite interfaccia web, CLI o SDK Python. [Guida al client](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [Chiamate a strumenti](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| Prestazioni e gestione | Esegui benchmark dei modelli di chat attivi nell’interfaccia web. Le configurazioni compatibili consentono di condividere la cache dei prefissi con Xavier, modificare alcuni parametri in esecuzione e ricaricare le repliche una alla volta. [Test delle prestazioni](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [Configurazione in esecuzione](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| Lingue | L’interfaccia web è disponibile in inglese, cinese semplificato, cinese tradizionale, giapponese e coreano. Il README è disponibile in dieci lingue. |

## Confronto tra piattaforme

| Funzionalità | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Esecuzione autonoma senza Kubernetes** | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supporto non documentato][compare-undocumented] | ![Supportato][compare-supported] |
| **Inferenza su più nodi** | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supportato][compare-supported] |
| **CPU / GPU / Apple Metal** | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Parziale o tramite integrazione][compare-partial] | ![Parziale o tramite integrazione][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Parziale o tramite integrazione][compare-partial] | ![Parziale o tramite integrazione][compare-partial] |
| **Prefill e decodifica tra CUDA e Metal** | ![Supportato][compare-supported] | ![Supporto non documentato][compare-undocumented] | ![Supporto non documentato][compare-undocumented] | ![Supporto non documentato][compare-undocumented] |
| **Motori diversi per lo stesso ID modello** | ![Supportato][compare-supported] | ![Parziale o tramite integrazione][compare-partial] | ![Parziale o tramite integrazione][compare-partial] | ![Parziale o tramite integrazione][compare-partial] |
| **Riconfigurazione progressiva con riutilizzo dei pesi in GPU** | ![Supportato][compare-supported] | ![Supporto non documentato][compare-undocumented] | ![Supporto non documentato][compare-undocumented] | ![Supporto non documentato][compare-undocumented] |
| Catalogo dei modelli | ![Supportato][compare-supported] 300+ | ![Supportato][compare-supported] | ![Parziale o tramite integrazione][compare-partial] | ![Parziale o tramite integrazione][compare-partial] |
| Distribuzione dei modelli tramite interfaccia web | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Parziale o tramite integrazione][compare-partial] | ![Supporto non documentato][compare-undocumented] |
| Embedding e reranking | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supportato][compare-supported] |
| Generazione di immagini e video | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supporto non documentato][compare-undocumented] | ![Supportato][compare-supported] |
| Riconoscimento e sintesi vocale | ![Supportato][compare-supported] | ![Supportato][compare-supported] | ![Supporto non documentato][compare-undocumented] | ![Parziale o tramite integrazione][compare-partial] |

![][compare-supported] Supportato · ![][compare-partial] Parziale o tramite integrazione · ![][compare-undocumented] Supporto non documentato

<details>
<summary>Ambito e fonti</summary>

Il confronto si basa sulle funzionalità documentate delle versioni open source. I motori e l’hardware supportati variano in base al modello. Prefill e decodifica tra CUDA e Metal richiedono un modello e una coppia di motori compatibili.

Xinference può servire in parallelo repliche con motori diversi sotto lo stesso ID modello. LocalAI usa una catena di failover ordinata sotto un unico nome di modello e invia le richieste alla prima destinazione disponibile. KServe e Dynamo possono instradare richieste tra distribuzioni distinte. Il riutilizzo dei pesi in GPU si applica alle configurazioni vLLM e SGLang supportate.

- **Xinference** · [Catalogo](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Backend](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Inferenza distribuita](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [P/D tra CUDA e Metal](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [Scelta del motore per le repliche](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [Riconfigurazione progressiva](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [Modelli e distribuzione](https://localai.io/docs/features/index.html) · [Motori e hardware](https://localai.io/docs/reference/index.print.html) · [Modalità distribuita](https://localai.io/docs/features/distributed-mode/) · [Alias e failover](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [Distribuzione e hardware](https://kserve.github.io/website/docs/intro) · [Runtime generativi integrati](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Catalogo di Kubeflow](https://www.kubeflow.org/docs/components/hub/installation/) · [Interfaccia web di Kubeflow](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [Distribuzioni canary](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [Distribuzione, hardware e motori](https://docs.dynamo.nvidia.com/dynamo) · [Ricette di distribuzione](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [Generazione di immagini, video e voce](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [Embedding e reranking](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [Instradamento tra distribuzioni](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## Novità

[Note della versione v4.2.0](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>Modelli integrati nelle ultime due versioni (v4.1 e v4.2)</summary>

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

## Test delle prestazioni

Apri **Performance Tests**, seleziona un modello di chat attivo e imposta i livelli di concorrenza da provare. Confronta throughput, tempo al primo token e tempo per token generato prima di scegliere una configurazione. [Guida ai benchmark](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![Grafici di throughput e latenza a diversi livelli di concorrenza](../assets/guide-benchmark-en.webp)

*Esempio di esecuzione locale con Qwen3.5-0.8B. Le prestazioni dipendono da hardware, motore e carico di lavoro.*

## Usare Xinference

| Obiettivo | Da dove iniziare |
| --- | --- |
| Installare su Linux, macOS o Windows | [Installazione con un comando](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guida all’installazione](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| Eseguire in un container o su Kubernetes | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| Registrare un modello personalizzato | [Modelli personalizzati](https://inference.readthedocs.io/en/latest/models/custom.html) |
| Collegare un’applicazione | [API compatibili con OpenAI](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**Applicazioni e framework** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Distribuisci Xinference Enterprise nel tuo cloud o data center. Xinference Cloud offre distribuzioni dedicate gestite e Model API fornisce endpoint per modelli ospitati. [Panoramica di Enterprise](https://xinference.co/enterprise.html)

**Contatti commerciali** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## Partecipare al progetto

| Piattaforma | Utilizzo |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | Segnalare problemi e proporre funzionalità. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Fare domande e confrontarsi con altri utenti. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Chattare con la comunità Xinference. |
| [X](https://x.com/Xinference_ai) | Seguire notizie e nuove versioni. |

## Citazione

Se usi Xinference nella tua ricerca, cita [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

## Collaboratori

Grazie a chi contribuisce a Xinference. Apri la tua prima pull request per aggiungere un modello, migliorare la documentazione o correggere un bug. [Guida ai contributi](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Mosaico degli avatar di chi contribuisce a Xinference](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
