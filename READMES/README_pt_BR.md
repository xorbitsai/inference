<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: Simplificando o deploy de modelos 🤖

<p align="center">
  <a href="https://xinference.co">Xinference Enterprise</a> ·
  <a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-Hosting</a> ·
  <a href="https://inference.readthedocs.io/">Documentação</a>
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
  <a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
  <a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-454545?style=for-the-badge"></a>
  <a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
  <a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />

Xorbits Inference (Xinference) é uma biblioteca poderosa e versátil para modelos de linguagem, reconhecimento de voz e modelos multimodais. Com o Xinference você pode implantar seu próprio modelo ou modelos integrados de ponta com um único comando e oferecê-los como um serviço. Pesquisadores, desenvolvedores e cientistas de dados podem explorar totalmente as capacidades dos modelos de IA modernos.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 Junte-se à nossa comunidade no Discord!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Participe do nosso grupo no Telegram</a></i>
</div>

## 🔥 Destaques
### Melhorias no framework
- O Xinference 4.0.0 está disponível com Xavier multi-engine, separação P/D heterogênea e reconfiguração de modelos em tempo de execução: [Notas da versão](https://github.com/xorbitsai/inference/releases/tag/v4.0.0)
- Deploy nativo para agentes: o Xinference integra-se ao [Xagent](https://github.com/xorbitsai/xagent), permitindo planejamento dinâmico, uso de ferramentas e inferências multi-step autônomas, ultrapassando os limites de pipelines estáticos.
- Batching automático: múltiplas requisições simultâneas são agrupadas automaticamente para aumentar significativamente o throughput. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): novos bindings Python para llama.cpp mantidos pela equipe Xinference, suportam batching contínuo e são mais adequados para produção. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferência distribuída: modelos podem ser executados entre vários workers: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Melhorias no vLLM: compartilhamento do KV-cache entre réplicas: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Novos modelos
- Suporte integrado para [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) : [#5792](https://github.com/xorbitsai/inference/pull/5792)
- Suporte integrado para a série FIBO 2.0 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)) : [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Suporte integrado para a série Ming-3.0 TTS ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)) : [#5788](https://github.com/xorbitsai/inference/pull/5788)
- Suporte integrado para [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) : [#5764](https://github.com/xorbitsai/inference/pull/5764)
- Suporte integrado para [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) : [#5755](https://github.com/xorbitsai/inference/pull/5755)
- Suporte integrado para [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) : [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Suporte integrado para a série Iris-3B ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)) : [#5686](https://github.com/xorbitsai/inference/pull/5686)
- Suporte integrado para a série MiniCPM-V-4.7 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)) : [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Suporte integrado para a série Qwen3.8-Flash-Next ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)) : [#5673](https://github.com/xorbitsai/inference/pull/5673)
- Suporte integrado para a série GLM-5.3 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)) : [#5666](https://github.com/xorbitsai/inference/pull/5666)
- Suporte integrado para [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) : [#5665](https://github.com/xorbitsai/inference/pull/5665)
- Suporte integrado para [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) : [#5661](https://github.com/xorbitsai/inference/pull/5661)
- Suporte integrado para [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) : [#5644](https://github.com/xorbitsai/inference/pull/5644)
- Suporte integrado para [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS) : [#5591](https://github.com/xorbitsai/inference/pull/5591)
- Suporte integrado para [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) : [#5588](https://github.com/xorbitsai/inference/pull/5588)
### Integrações
- [Xagent](https://github.com/xorbitsai/xagent): plataforma de agentes enterprise com planejamento, memória e integração de ferramentas.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): plataforma LLMOps para construir aplicações rapidamente com visualização e controle.
- [FastGPT](https://github.com/labring/FastGPT): plataforma de conhecimento baseada em LLM para processamento de dados e chamadas de modelo.
- [RAGFlow](https://github.com/infiniflow/ragflow): motor RAG open-source para compreensão profunda de documentos.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): assistente open-source de base de conhecimento com integração RAG.

## Principais funcionalidades
🌟 Deploy de modelos simplificado: simplifica a disponibilização de LLMs, modelos de reconhecimento de voz e modelos multimodais. Modelos experimentais e de produção podem ser configurados e implantados com um único comando.

⚡️ Modelos de ponta acessíveis: experimente modelos integrados com um único comando. O Xinference fornece acesso a modelos open source de última geração.

🖥 Suporte para hardware heterogêneo: aproveite GPUs e CPUs eficientemente (ex.: via [ggml](https://github.com/ggerganov/ggml)) para acelerar a inferência.

⚙️ APIs e interfaces flexíveis: API RESTful compatível com OpenAI (incluindo Function Calling), RPC, CLI, Web UI e mais.

🌐 Deploy distribuído: facilita a distribuição de inferência através de múltiplos dispositivos e máquinas.

🔌 Integrações de terceiros: integração com [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox], etc.

## Por que Xinference
| Função                                      | Xinference | FastChat | OpenLLM | RayLLM |
|---------------------------------------------|------------|----------|---------|--------|
| API RESTful compatível com OpenAI            | ✅         | ✅        | ✅       | ✅      |
| Integração vLLM                              | ✅         | ✅        | ✅       | ✅      |
| Diversos motores de inferência (GGML, TensorRT)| ✅         | ❌        | ✅       | ✅      |
| Diversas plataformas (CPU, Metal)            | ✅         | ✅        | ❌       | ❌      |
| Deploy em cluster multi-nó                    | ✅         | ❌        | ❌       | ✅      |
| Modelos de imagem (Texto→Imagem)             | ✅         | ✅        | ❌       | ❌      |
| Modelos de embeddings de texto               | ✅         | ❌        | ❌       | ❌      |
| Modelos multimodais                           | ✅         | ❌        | ❌       | ❌      |
| Modelos de voz                                | ✅         | ❌        | ❌       | ❌      |
| Function Calling (OpenAI-like)               | ✅         | ❌        | ❌       | ❌      |

## Como usar o Xinference

- **Self-Hosting Xinference Community Edition**
  Siga o [guia de início](#começando) para executar o Xinference localmente. Detalhes na documentação: https://inference.readthedocs.io/.

- **Xinference para empresas**
  **Xinference Enterprise** é a edição comercial para equipes que executam modelos em produção. Implante-a na sua própria nuvem ou no seu data center, ou escolha uma implantação dedicada gerenciada com **Xinference Cloud**. A **Model API** hospedada permite acessar os modelos compatíveis por meio de uma API compatível com OpenAI sem gerenciar a infraestrutura de inferência. Saiba mais em [xinference.co](https://xinference.co).
  Para consultas empresariais, [entre em contato por e-mail](mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry).

## Fique atualizado

Dê uma estrela no Xinference no GitHub para receber atualizações de release.

![star-us](../assets/stay_ahead.gif)

## Começando

* [Documentação](https://inference.readthedocs.io/en/latest/index.html)
* [Modelos integrados](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelos customizados](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentação de deploy](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Quickstart

**pip**

```bash
pip install xinference
xinference-local
```

**Instalação com um comando**

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
```

[Guia de instalação](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guia de uso](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally)

### Docker

Usuários com GPU NVIDIA podem usar a [imagem Docker do Xinference](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html). Certifique-se de ter Docker e CUDA antes de instalar.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

Após habilitar GPUs no cluster Kubernetes, instale com:

```
# Adicionar repositório
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# Atualizar índice e verificar versões
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Instalar Xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

Mais opções de K8s na documentação.

![web UI](../assets/screenshot.png)

## Contribuir

| Plataforma                                                                 | Propósito                                |
|---------------------------------------------------------------------------|------------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)             | Reportar bugs e solicitar features        |
| [Discord](https://discord.gg/Xw9tszSkr5)                                  | Colaboração com outros usuários          |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                | Discussões com a comunidade              |
| [Twitter](https://twitter.com/xorbitsio)                                  | Notícias e anúncios                      |

## Citação

Se este projeto foi útil, cite-o assim:

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

## Colaboradores

<a href="https://github.com/xorbitsai/inference/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=xorbitsai/inference" />
</a>

## Histórico de estrelas

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
