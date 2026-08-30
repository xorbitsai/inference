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
- Supporto integrato per [Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) : [#5437](https://github.com/xorbitsai/inference/pull/5437)
- Supporto integrato per la serie WeMM-Embedding ([2B](https://huggingface.co/tencent/WeMM-Embedding-2B), [4B](https://huggingface.co/tencent/WeMM-Embedding-4B), [9B](https://huggingface.co/tencent/WeMM-Embedding-9B)) : [#5439](https://github.com/xorbitsai/inference/pull/5439)
- Supporto integrato per [NaviDC-OCR](https://huggingface.co/StarDoc-AI/NaviDC-OCR) : [#5431](https://github.com/xorbitsai/inference/pull/5431)
- Supporto integrato per [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) : [#5417](https://github.com/xorbitsai/inference/pull/5417)
- Supporto integrato per i modelli world ([Matrix-Game-3.0-5B](https://huggingface.co/Skywork/Matrix-Game-3.0), [HY-WorldPlay-5B](https://huggingface.co/tencent/HY-WorldPlay), [Astra](https://huggingface.co/EvanEternal/Astra)) : [#5414](https://github.com/xorbitsai/inference/pull/5414)
- Supporto integrato per la serie Krea 2 ([Raw](https://huggingface.co/krea/Krea-2-Raw), [Turbo](https://huggingface.co/krea/Krea-2-Turbo)) : [#5419](https://github.com/xorbitsai/inference/pull/5419)
- Supporto integrato per [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) : [#5413](https://github.com/xorbitsai/inference/pull/5413)
- Supporto integrato per la serie Ornith 1.5 ([35B-A3B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-35B-A3B), [397B](https://modelscope.cn/models/ornith-ai/Ornith-1.5-397B)) : [#5406](https://github.com/xorbitsai/inference/pull/5406), [#5405](https://github.com/xorbitsai/inference/pull/5405)
- Supporto integrato per [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) : [#5404](https://github.com/xorbitsai/inference/pull/5404)
- Supporto integrato per [GLM-Image](https://huggingface.co/zai-org/GLM-Image) : [#5394](https://github.com/xorbitsai/inference/pull/5394)
- Supporto integrato per la serie HiDream-O1 ([Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image), [Image-Dev](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev), [Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604)) : [#5370](https://github.com/xorbitsai/inference/pull/5370)
- Supporto integrato per [SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) : [#5369](https://github.com/xorbitsai/inference/pull/5369)
- Supporto integrato per [Ideogram4](https://huggingface.co/ideogram-ai/ideogram-4-nf4-diffusers) : [#5367](https://github.com/xorbitsai/inference/pull/5367)
- Supporto integrato per [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) : [#5371](https://github.com/xorbitsai/inference/pull/5371)
- Supporto integrato per [FireRedTTS3](https://huggingface.co/FireRedTeam/FireRedTTS3) : [#5352](https://github.com/xorbitsai/inference/pull/5352)
- Supporto integrato per [MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) : [#5345](https://github.com/xorbitsai/inference/pull/5345)
- Supporto integrato per la serie Qwen3.8 ([27B](https://huggingface.co/Qwen/Qwen3.8-27B), [2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)) : [#5337](https://github.com/xorbitsai/inference/pull/5337), [#5339](https://github.com/xorbitsai/inference/pull/5339)
- Supporto integrato per [jina-reranker-m0](https://huggingface.co/jinaai/jina-reranker-m0) : [#5327](https://github.com/xorbitsai/inference/pull/5327)
- Supporto integrato per [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) : [#5322](https://github.com/xorbitsai/inference/pull/5322)
- Supporto integrato per [IndexTTS-2.5](https://huggingface.co/IndexTeam/IndexTTS-2.5) : [#5319](https://github.com/xorbitsai/inference/pull/5319)
- Supporto integrato per la serie Ling-3.0 ([tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny), [flash](https://huggingface.co/inclusionAI/Ling-3.0-flash)) : [#5311](https://github.com/xorbitsai/inference/pull/5311)
- Supporto integrato per [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) e [Lightning LoRA](https://huggingface.co/lightx2v/Minimax-h3-Turbo) : [#5321](https://github.com/xorbitsai/inference/pull/5321), [#5338](https://github.com/xorbitsai/inference/pull/5338)
- Supporto integrato per la serie Wan2.2 Animate 2 ([14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers), [14B Distilled](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers)) : [#5309](https://github.com/xorbitsai/inference/pull/5309)
- Supporto integrato per [FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1) : [#5306](https://github.com/xorbitsai/inference/pull/5306)
- Supporto integrato per la serie CAMPPlus di embedding vocali ([cinese](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common), [cinese-inglese avanzato](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced)) : [#5298](https://github.com/xorbitsai/inference/pull/5298)
- Supporto integrato per [DeepDoc](https://huggingface.co/Xorbits/deepdoc) : [#5230](https://github.com/xorbitsai/inference/pull/5230)
- Supporto integrato per [jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) : [#5269](https://github.com/xorbitsai/inference/pull/5269)
- Supporto integrato per la serie R3 ([embedding](https://huggingface.co/tencent/R3-embedding-0.6b), [reranking](https://huggingface.co/tencent/R3-rerank-0.6b)) : [#5272](https://github.com/xorbitsai/inference/pull/5272)
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
