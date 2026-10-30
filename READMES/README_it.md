<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: rendere semplice il deploy dei modelli 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-Hosting</a> ·
  <a href="https://inference.readthedocs.io/">Documentazione</a>
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
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-454545?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference (Xinference) è una libreria potente e versatile per modelli di linguaggio, riconoscimento vocale e modelli multimodali. Con Xinference puoi distribuire il tuo modello o modelli integrati di ultima generazione con un solo comando e servirli come servizio. Ricercatori, sviluppatori e data scientist possono sfruttare appieno le capacità dei moderni modelli di IA.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Unisciti alla nostra community Discord!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Unisciti al nostro gruppo Telegram</a></i>
</div>

## 🔥 Novità in evidenza
### Miglioramenti del framework
- Xinference 4.0.0 è disponibile con Xavier multi-engine, separazione P/D eterogenea e riconfigurazione dei modelli in fase di esecuzione: [Note di rilascio](https://github.com/xorbitsai/inference/releases/tag/v4.0.0)
- Deploy nativo per agenti: Xinference si integra con [Xagent](https://github.com/xorbitsai/xagent) fornendo pianificazione dinamica, utilizzo di tool e inferenze multi-step autonome, superando i limiti delle pipeline statiche.
- Batching automatico: più richieste concorrenti vengono raggruppate automaticamente per aumentare significativamente il throughput. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): nuove binding Python per llama.cpp, mantenute dal team Xinference, che supportano il batching continuo e sono più adatte alla produzione. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferenza distribuita: i modelli possono essere eseguiti attraverso più worker: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Miglioramenti per vLLM: condivisione del KV-cache tra più repliche: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nuovi modelli
- Supporto integrato per [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) : [#5792](https://github.com/xorbitsai/inference/pull/5792)
- Supporto integrato per la serie FIBO 2.0 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)) : [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Supporto integrato per la serie Ming-3.0 TTS ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) : [#5788](https://github.com/xorbitsai/inference/pull/5788)
- Supporto integrato per [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) : [#5764](https://github.com/xorbitsai/inference/pull/5764)
- Supporto integrato per [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) : [#5755](https://github.com/xorbitsai/inference/pull/5755)
- Supporto integrato per [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) : [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Supporto integrato per la serie Iris-3B ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)) : [#5686](https://github.com/xorbitsai/inference/pull/5686)
- Supporto integrato per la serie MiniCPM-V-4.7 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)) : [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Supporto integrato per la serie Qwen3.8-Flash-Next ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)) : [#5673](https://github.com/xorbitsai/inference/pull/5673)
- Supporto integrato per la serie GLM-5.3 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)) : [#5666](https://github.com/xorbitsai/inference/pull/5666)
- Supporto integrato per [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) : [#5665](https://github.com/xorbitsai/inference/pull/5665)
- Supporto integrato per [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) : [#5661](https://github.com/xorbitsai/inference/pull/5661)
- Supporto integrato per [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) : [#5644](https://github.com/xorbitsai/inference/pull/5644)
- Supporto integrato per [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) : [#5591](https://github.com/xorbitsai/inference/pull/5591)
- Supporto integrato per [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) : [#5588](https://github.com/xorbitsai/inference/pull/5588)
### Integrazioni
- [Xagent](https://github.com/xorbitsai/xagent): piattaforma enterprise per agenti con pianificazione, memoria e integrazione di tool.
- [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference): piattaforma LLMOps per costruire rapidamente applicazioni con visualizzazione e controllo.
- [FastGPT](https://github.com/labring/FastGPT): piattaforma di conoscenza basata su LLM per l'elaborazione dei dati e le chiamate ai modelli.
- [RAGFlow](https://github.com/infiniflow/ragflow): motore RAG open-source per una comprensione profonda dei documenti.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): assistente open-source per basi di conoscenza con integrazione RAG.

## Funzionalità principali
🌟 Deploy di modelli semplificato: semplifica l'esposizione di LLM, modelli di riconoscimento vocale e modelli multimodali. I modelli di sperimentazione e produzione possono essere configurati e distribuiti con un unico comando.

⚡️ Modelli all'avanguardia facilmente accessibili: prova i modelli integrati con un solo comando. Xinference offre accesso a modelli open source di ultima generazione.

🖥 Supporto per hardware eterogeneo: sfrutta GPU e CPU in modo efficiente (es. tramite [ggml](https://github.com/ggerganov/ggml)) per accelerare l'inferenza.

⚙️ API e interfacce flessibili: API RESTful compatibile OpenAI (incluso Function Calling), RPC, CLI, Web UI, ecc.

🌐 Deploy distribuito: facilita la distribuzione dell'inferenza su più dispositivi e macchine.

🔌 Integrazioni di terze parti: integrazione con [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference), [LlamaIndex], [Dify], [Chatbox], ecc.

## Perché Xinference
| Funzionalità                                  | Xinference | FastChat | OpenLLM | RayLLM |
|-----------------------------------------------|------------|----------|---------|--------|
| API RESTful compatibile OpenAI                 | ✅         | ✅        | ✅       | ✅      |
| Integrazione vLLM                               | ✅         | ✅        | ✅       | ✅      |
| Diversi motori di inferenza (GGML, TensorRT)    | ✅         | ❌        | ✅       | ✅      |
| Diverse piattaforme (CPU, Metal)                | ✅         | ✅        | ❌       | ❌      |
| Deploy in cluster multi-nodo                    | ✅         | ❌        | ❌       | ✅      |
| Modelli immagine (Testo→Immagine)               | ✅         | ✅        | ❌       | ❌      |
| Modelli di embedding testo                      | ✅         | ❌        | ❌       | ❌      |
| Modelli multimodali                              | ✅         | ❌        | ❌       | ❌      |
| Modelli vocali                                   | ✅         | ❌        | ❌       | ❌      |
| Funzionalità OpenAI (Function Calling)           | ✅         | ❌        | ❌       | ❌      |

## Come usare Xinference

- **Self-Hosting Xinference Community Edition**
  Segui la [guida di avvio](#getting-started) per lanciare Xinference localmente. Dettagli nella documentazione: https://inference.readthedocs.io/.

- **Xinference per le aziende**
  **Xinference Enterprise** è l’edizione commerciale per i team che eseguono modelli in produzione. È possibile distribuirla nel proprio cloud o data center, oppure scegliere una distribuzione dedicata gestita con **Xinference Cloud**. La **Model API** ospitata consente di accedere ai modelli supportati tramite un’API compatibile con OpenAI senza gestire l’infrastruttura di inferenza. Maggiori informazioni su [xinference.co](https://xinference.co).
  Per richieste aziendali, [contattateci via e-mail](mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry).

## Rimani aggiornato

Dai una stella a Xinference su GitHub per ricevere aggiornamenti sulle release.

![star-us](../assets/stay_ahead.gif)

## Getting started

* [Documentazione](https://inference.readthedocs.io/en/latest/index.html)
* [Modelli integrati](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelli custom](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentazione sul deployment](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Quickstart

**pip**

```bash
pip install xinference
xinference-local
```

**Installazione con un comando**

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
```

[Guida all’installazione](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guida all’uso](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally)

### Docker

Gli utenti con GPU NVIDIA possono usare l'[immagine Docker di Xinference](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html). Verifica che Docker e CUDA siano installati prima dell'uso.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

Dopo aver abilitato le GPU nel cluster Kubernetes, installa con:

```
# Aggiungi repository
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# Aggiorna indice e controlla le versioni
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Installa Xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

Ulteriori opzioni K8s nella documentazione.

![web UI](../assets/screenshot.png)

## Contribuire

| Piattaforma                                                                 | Scopo                                    |
|-----------------------------------------------------------------------------|------------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)              | Segnalazione bug e richieste di feature  |
| [Discord](https://discord.gg/Xw9tszSkr5)                                   | Collaborazione con altri utenti          |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                 | Discussioni con la community             |
| [Twitter](https://twitter.com/xorbitsio)                                   | Novità e annunci                         |

## Citazione

Se questo progetto ti è stato utile, citane il lavoro così:

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

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Storico stelle

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
