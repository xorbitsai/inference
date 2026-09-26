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
- Xinference 3.0.0 è disponibile con note di migrazione e modifiche incompatibili: [Note di rilascio](https://xinference.co/release_notes/v3.0.0.html)
- Deploy nativo per agenti: Xinference si integra con [Xagent](https://github.com/xorbitsai/xagent) fornendo pianificazione dinamica, utilizzo di tool e inferenze multi-step autonome, superando i limiti delle pipeline statiche.
- Batching automatico: più richieste concorrenti vengono raggruppate automaticamente per aumentare significativamente il throughput. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): nuove binding Python per llama.cpp, mantenute dal team Xinference, che supportano il batching continuo e sono più adatte alla produzione. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferenza distribuita: i modelli possono essere eseguiti attraverso più worker: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Miglioramenti per vLLM: condivisione del KV-cache tra più repliche: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nuovi modelli
- Supporto integrato per [TeleOCR](https://huggingface.co/StarDoc-AI/TeleOCR) : [#5583](https://github.com/xorbitsai/inference/pull/5583)
- Supporto integrato per la serie Ming-Image ([Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), [Design Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)) : [#5582](https://github.com/xorbitsai/inference/pull/5582)
- Supporto integrato per [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) : [#5571](https://github.com/xorbitsai/inference/pull/5571)
- Supporto integrato per [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) : [#5550](https://github.com/xorbitsai/inference/pull/5550)
- Supporto integrato per la serie Spark-X2.5 ([1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B), [4B](https://huggingface.co/XHToken/Spark-X2.5-4B), [1.7B Base](https://huggingface.co/XHToken/Spark-X2.5-1.7B-Base), [4B Base](https://huggingface.co/XHToken/Spark-X2.5-4B-Base)) : [#5538](https://github.com/xorbitsai/inference/pull/5538)
- Supporto integrato per la serie LingBot-World-V2 ([14B Fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast), [14B Pretrain](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-pretrain), [1.3B Fast](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast)) : [#5536](https://github.com/xorbitsai/inference/pull/5536)
- Supporto integrato per la serie Irodori-TTS v4.1 ([Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime), [Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)) : [#5527](https://github.com/xorbitsai/inference/pull/5527)
- Supporto integrato per [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) : [#5526](https://github.com/xorbitsai/inference/pull/5526)
- Supporto integrato per la serie AuK ([AuK](https://huggingface.co/tencent/AuK), [AuK-Flash](https://huggingface.co/tencent/AuK-Flash)) : [#5525](https://github.com/xorbitsai/inference/pull/5525)
- Supporto integrato per [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) : [#5506](https://github.com/xorbitsai/inference/pull/5506)
- Supporto integrato per la serie Fish Audio ([S1-mini](https://huggingface.co/fishaudio/s1-mini), [S2-Pro](https://huggingface.co/fishaudio/s2-pro)) : [#5490](https://github.com/xorbitsai/inference/pull/5490)
- Supporto integrato per [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR) : [#5475](https://github.com/xorbitsai/inference/pull/5475)
- Supporto integrato per [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) : [#5468](https://github.com/xorbitsai/inference/pull/5468)
- Supporto integrato per la serie JoyAI per la modifica delle immagini ([Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers), [Edit Plus](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers)) : [#5458](https://github.com/xorbitsai/inference/pull/5458)
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
  Sono disponibili funzionalità enterprise; per richieste contatta: mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry

## Rimani aggiornato

Dai una stella a Xinference su GitHub per ricevere aggiornamenti sulle release.

![star-us](../assets/stay_ahead.gif)

## Getting started

* [Documentazione](https://inference.readthedocs.io/en/latest/index.html)
* [Modelli integrati](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelli custom](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentazione sul deployment](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

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

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
