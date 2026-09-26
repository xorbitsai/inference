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
- O Xinference 3.0.0 está disponível com notas de migração e mudanças incompatíveis: [Notas da versão](https://xinference.co/release_notes/v3.0.0.html)
- Deploy nativo para agentes: o Xinference integra-se ao [Xagent](https://github.com/xorbitsai/xagent), permitindo planejamento dinâmico, uso de ferramentas e inferências multi-step autônomas, ultrapassando os limites de pipelines estáticos.
- Batching automático: múltiplas requisições simultâneas são agrupadas automaticamente para aumentar significativamente o throughput. : [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): novos bindings Python para llama.cpp mantidos pela equipe Xinference, suportam batching contínuo e são mais adequados para produção. : [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferência distribuída: modelos podem ser executados entre vários workers: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Melhorias no vLLM: compartilhamento do KV-cache entre réplicas: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Novos modelos
- Suporte integrado para [TeleOCR](https://huggingface.co/StarDoc-AI/TeleOCR) : [#5583](https://github.com/xorbitsai/inference/pull/5583)
- Suporte integrado para a série Ming-Image ([Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design), [Design Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer)) : [#5582](https://github.com/xorbitsai/inference/pull/5582)
- Suporte integrado para [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) : [#5571](https://github.com/xorbitsai/inference/pull/5571)
- Suporte integrado para [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) : [#5550](https://github.com/xorbitsai/inference/pull/5550)
- Suporte integrado para a série Spark-X2.5 ([1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B), [4B](https://huggingface.co/XHToken/Spark-X2.5-4B), [1.7B Base](https://huggingface.co/XHToken/Spark-X2.5-1.7B-Base), [4B Base](https://huggingface.co/XHToken/Spark-X2.5-4B-Base)) : [#5538](https://github.com/xorbitsai/inference/pull/5538)
- Suporte integrado para a série LingBot-World-V2 ([14B Fast](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast), [14B Pretrain](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-pretrain), [1.3B Fast](https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast)) : [#5536](https://github.com/xorbitsai/inference/pull/5536)
- Suporte integrado para a série Irodori-TTS v4.1 ([Anime](https://huggingface.co/phasefield-audio/Irodori-TTS-v4.1-Anime), [Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small)) : [#5527](https://github.com/xorbitsai/inference/pull/5527)
- Suporte integrado para [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) : [#5526](https://github.com/xorbitsai/inference/pull/5526)
- Suporte integrado para a série AuK ([AuK](https://huggingface.co/tencent/AuK), [AuK-Flash](https://huggingface.co/tencent/AuK-Flash)) : [#5525](https://github.com/xorbitsai/inference/pull/5525)
- Suporte integrado para [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) : [#5506](https://github.com/xorbitsai/inference/pull/5506)
- Suporte integrado para a série Fish Audio ([S1-mini](https://huggingface.co/fishaudio/s1-mini), [S2-Pro](https://huggingface.co/fishaudio/s2-pro)) : [#5490](https://github.com/xorbitsai/inference/pull/5490)
- Suporte integrado para [MonkeyOCR](https://huggingface.co/echo840/MonkeyOCR) : [#5475](https://github.com/xorbitsai/inference/pull/5475)
- Suporte integrado para [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) : [#5468](https://github.com/xorbitsai/inference/pull/5468)
- Suporte integrado para a série JoyAI de edição de imagens ([Edit](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Diffusers), [Edit Plus](https://huggingface.co/jdopensource/JoyAI-Image-Edit-Plus-Diffusers)) : [#5458](https://github.com/xorbitsai/inference/pull/5458)
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
  Siga o [guia de início](#getting-started) para executar o Xinference localmente. Detalhes na documentação: https://inference.readthedocs.io/.

- **Xinference para empresas**
  Recursos enterprise estão disponíveis; para solicitações, entre em contato: mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry

## Fique atualizado

Dê uma estrela no Xinference no GitHub para receber atualizações de release.

![star-us](../assets/stay_ahead.gif)

## Começando

* [Documentação](https://inference.readthedocs.io/en/latest/index.html)
* [Modelos integrados](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelos customizados](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentação de deploy](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

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

### Quickstart

Instale o Xinference via pip:

```bash
pip install "xinference[all]"
```

Inicie uma instância local com:

```bash
$ xinference-local
```

Depois, você pode usar a Web UI, cURL, CLI ou o cliente Python.

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
