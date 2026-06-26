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
- Deploy nativo per agenti: Xinference si integra con [Xagent](https://github.com/xorbitsai/xagent) fornendo pianificazione dinamica, utilizzo di tool e inferenze multi-step autonome, superando i limiti delle pipeline statiche.
- Batching automatico: più richieste concorrenti vengono raggruppate automaticamente per aumentare significativamente il throughput. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): nuove binding Python per llama.cpp, mantenute dal team Xinference, che supportano il batching continuo e sono più adatte alla produzione. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferenza distribuita: i modelli possono essere eseguiti attraverso più worker: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Miglioramenti per vLLM: condivisione del KV-cache tra più repliche: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nuovi modelli
- Integrazione di [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) : [#5010](https://github.com/xorbitsai/inference/pull/5010)
- Serie integrata jina-embeddings-v5 (ad es. [text-nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano), [text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)) : [#5018](https://github.com/xorbitsai/inference/pull/5018)
- Serie integrata MiniCPM-V-4.6 : [#5025](https://github.com/xorbitsai/inference/pull/5025)
- Serie integrata Tencent Hy-MT2 (1.8B, 7B, 30B-A3B) : [#5029](https://github.com/xorbitsai/inference/pull/5029)
- Integrazione di [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) : [#5033](https://github.com/xorbitsai/inference/pull/5033)
- Integrazione di [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) : [#5045](https://github.com/xorbitsai/inference/pull/5045)
- Integrazione di [DeepSeek V4] : [#4938](https://github.com/xorbitsai/inference/pull/4938)
- Integrazione di [MiniMax-M2.7] : [#4843](https://github.com/xorbitsai/inference/pull/4843)
### Integrazioni
- [Xagent](https://github.com/xorbitsai/xagent): piattaforma enterprise per agenti con pianificazione, memoria e integrazione di tool.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): piattaforma LLMOps per costruire rapidamente applicazioni con visualizzazione e controllo.
- [FastGPT](https://github.com/labring/FastGPT): piattaforma di conoscenza basata su LLM per l'elaborazione dei dati e le chiamate ai modelli.
- [RAGFlow](https://github.com/infiniflow/ragflow): motore RAG open-source per una comprensione profonda dei documenti.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): assistente open-source per basi di conoscenza con integrazione RAG.

## Funzionalità principali
🌟 Deploy di modelli semplificato: semplifica l'esposizione di LLM, modelli di riconoscimento vocale e modelli multimodali. I modelli di sperimentazione e produzione possono essere configurati e distribuiti con un unico comando.

⚡️ Modelli all'avanguardia facilmente accessibili: prova i modelli integrati con un solo comando. Xinference offre accesso a modelli open source di ultima generazione.

🖥 Supporto per hardware eterogeneo: sfrutta GPU e CPU in modo efficiente (es. tramite [ggml](https://github.com/ggerganov/ggml)) per accelerare l'inferenza.

⚙️ API e interfacce flessibili: API RESTful compatibile OpenAI (incluso Function Calling), RPC, CLI, Web UI, ecc.

🌐 Deploy distribuito: facilita la distribuzione dell'inferenza su più dispositivi e macchine.

🔌 Integrazioni di terze parti: integrazione con [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox], ecc.

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
  Sono disponibili funzionalità enterprise; per richieste contatta: mailto:business@xprobe.io?subject=[GitHub]Business%20License%20Inquiry

## Rimani aggiornato

Dai una stella a Xinference su GitHub per ricevere aggiornamenti sulle release.

![star-us](../assets/stay_ahead.gif)

## Getting started

* [Documentazione](https://inference.readthedocs.io/en/latest/index.html)
* [Modelli integrati](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelli custom](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentazione sul deployment](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)
* [Esempi e tutorial](https://inference.readthedocs.io/en/latest/examples/index.html)

### Jupyter Notebook

Il modo più semplice per provare Xinference è il [notebook Google Colab](https://colab.research.google.com/github/xorbitsai/inference/blob/main/examples/Xinference_Quick_Start.ipynb).

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

### Quickstart

Installa Xinference con pip:

```bash
pip install "xinference[all]"
```

Avvia un'istanza locale con:

```bash
$ xinference-local
```

Poi puoi usare la Web UI, cURL, la CLI o il client Python.

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

[![Star History Chart](https://api.star-history.com/svg?repos=xorbitsai/inference&type=Date)](https://star-history.com/#xorbitsai/inference&Date)
