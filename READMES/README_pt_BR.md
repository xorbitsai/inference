# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference (Xorbits Inference) é uma plataforma de inferência distribuída de código aberto.**

Implante e execute modelos em hardware heterogêneo, de uma máquina local a um cluster distribuído. O Xinference inclui mais de 300 modelos integrados para linguagem, visão, embeddings, reranking, imagens, áudio, vídeo e geração de mundos (experimental). Cadastre seus próprios modelos, incluindo modelos de aprendizado de máquina tradicional por meio do tipo Flexible. Gerencie as implantações na interface web e use os modelos via APIs compatíveis com a OpenAI, SDK Python ou CLI.

[Início rápido](#início-rápido) · [Principais recursos](#principais-recursos) · [Comparação de plataformas](#comparação-de-plataformas) · [Hospedagem própria](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [Documentação](https://inference.readthedocs.io/en/latest/) · [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Versões](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![Versão no GitHub](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![Versão no PyPI](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Tag do Docker](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Versões do Python](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![CI do Python](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Licença Apache 2.0](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Estrelas no GitHub](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![Forks no GitHub](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Colaboradores](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Downloads do Docker](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Comunidade no Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Comunidade no Telegram](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![Siga o Xinference no X](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**Idiomas** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · [Español](./README_es.md) · [Italiano](./README_it.md) · **Português** · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## Início rápido

### 1. Inicie o servidor

Na primeira execução, o Xinference baixa o modelo e instala as dependências necessárias do motor de inferência.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

Abra [http://127.0.0.1:9997](http://127.0.0.1:9997). Em uma nova instância, use o token de configuração exibido no log do servidor para criar uma conta de administrador. Em seguida, faça login. [Configuração inicial](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. Implante um modelo de chat

Em **Launch Model**, pesquise por `qwen3.5` e abra o formulário de implantação. Clique em **Recommend configuration** para preencher as opções de acordo com o seu hardware. Revise a recomendação e clique em **Deploy**.

![Formulário completo do Qwen3.5 com a configuração recomendada aplicada](../assets/guide-launch-recommend-auto.webp)

### 3. Envie a primeira requisição

Abra o modelo em **Running Models** para testar um prompt na interface web. A primeira instância usa `qwen3.5` como UID do modelo. Se a sua instância mostrar outro UID, use esse valor no campo `model` abaixo. Crie uma chave em **API Key Management** para chamar a API.

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

![Interface de chat do Qwen3.5 com uma resposta completa](../assets/guide-chat-en.webp)

## Principais recursos

| Área | Recursos do Xinference |
| --- | --- |
| Catálogo de modelos | Mais de 300 modelos integrados para linguagem, visão, embeddings, reranking, imagens, áudio, vídeo e geração de mundos (experimental). Cadastre modelos próprios, incluindo modelos de ML tradicional por meio do tipo Flexible. [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| Motores e hardware | Para LLMs compatíveis, escolha vLLM, SGLang, TRT-LLM, Transformers, llama.cpp ou MLX. O suporte a CPUs, GPUs, Apple Silicon e determinadas configurações MLU varia conforme o motor. [Backends de inferência](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Suporte a MLU](https://github.com/xorbitsai/inference/pull/3693) |
| Inferência distribuída | Execute em uma máquina, adicione workers e réplicas ou execute um modelo grande em vários workers com um motor compatível. Implantações compatíveis permitem separar prefill e decodificação em hardware heterogêneo. [Inferência distribuída](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Separação P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| APIs e ferramentas | Endpoints compatíveis com OpenAI e Anthropic, com streaming e chamadas de ferramentas quando disponíveis. Implante e gerencie modelos pela interface web, CLI ou SDK Python. [Guia do cliente](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [Chamadas de ferramentas](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| Desempenho e operação | Teste modelos de chat em execução pela interface web. Implantações compatíveis podem compartilhar cache de prefixos com o Xavier, ajustar parâmetros em tempo de execução e recarregar réplicas uma por vez. [Testes de desempenho](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [Configuração em tempo de execução](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| Idiomas | A interface web está disponível em inglês, chinês simplificado, chinês tradicional, japonês e coreano. O README está disponível em dez idiomas. |

## Comparação de plataformas

| Recurso | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Execução local sem Kubernetes** | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Suporte não documentado][compare-undocumented] | ![Compatível][compare-supported] |
| **Inferência em vários nós** | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Compatível][compare-supported] |
| **CPU / GPU / Apple Metal** | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Parcial ou via integração][compare-partial] | ![Parcial ou via integração][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Parcial ou via integração][compare-partial] | ![Parcial ou via integração][compare-partial] |
| **Prefill e decodificação entre CUDA e Metal** | ![Compatível][compare-supported] | ![Suporte não documentado][compare-undocumented] | ![Suporte não documentado][compare-undocumented] | ![Suporte não documentado][compare-undocumented] |
| **Motores diferentes sob o mesmo ID de modelo** | ![Compatível][compare-supported] | ![Parcial ou via integração][compare-partial] | ![Parcial ou via integração][compare-partial] | ![Parcial ou via integração][compare-partial] |
| **Reconfiguração gradual com reaproveitamento de pesos na GPU** | ![Compatível][compare-supported] | ![Suporte não documentado][compare-undocumented] | ![Suporte não documentado][compare-undocumented] | ![Suporte não documentado][compare-undocumented] |
| Catálogo de modelos | ![Compatível][compare-supported] 300+ | ![Compatível][compare-supported] | ![Parcial ou via integração][compare-partial] | ![Parcial ou via integração][compare-partial] |
| Implantação de modelos pela interface web | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Parcial ou via integração][compare-partial] | ![Suporte não documentado][compare-undocumented] |
| Embeddings e reranking | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Compatível][compare-supported] |
| Geração de imagens e vídeos | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Suporte não documentado][compare-undocumented] | ![Compatível][compare-supported] |
| Reconhecimento e síntese de voz | ![Compatível][compare-supported] | ![Compatível][compare-supported] | ![Suporte não documentado][compare-undocumented] | ![Parcial ou via integração][compare-partial] |

![][compare-supported] Compatível · ![][compare-partial] Parcial ou via integração · ![][compare-undocumented] Suporte não documentado

<details>
<summary>Escopo e fontes</summary>

A comparação considera os recursos documentados das versões de código aberto. A compatibilidade com motores e hardware varia de acordo com o modelo. A separação entre CUDA e Metal requer um modelo e um par de motores compatíveis.

O Xinference pode atender requisições com réplicas de motores diferentes em paralelo sob o mesmo ID de modelo. O LocalAI usa uma cadeia ordenada de failover sob um único nome de modelo e envia as requisições ao primeiro destino disponível. KServe e Dynamo podem encaminhar requisições entre implantações separadas. O reaproveitamento dos pesos na GPU se aplica a configurações compatíveis de vLLM e SGLang.

- **Xinference** · [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Backends](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Inferência distribuída](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Separação P/D entre CUDA e Metal](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [Motor por réplica](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [Reconfiguração gradual](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [Modelos e implantação](https://localai.io/docs/features/index.html) · [Backends e hardware](https://localai.io/docs/reference/index.print.html) · [Modo distribuído](https://localai.io/docs/features/distributed-mode/) · [Aliases e failover](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [Implantação e hardware](https://kserve.github.io/website/docs/intro) · [Runtimes generativos integrados](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Catálogo do Kubeflow](https://www.kubeflow.org/docs/components/hub/installation/) · [Interface web do Kubeflow](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [Implantações canário](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [Implantação, hardware e motores](https://docs.dynamo.nvidia.com/dynamo) · [Receitas de implantação](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [Geração de imagens, vídeos e voz](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [Embeddings e reranking](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [Roteamento entre implantações](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## Novidades

[Notas da versão v4.2.0](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>Modelos integrados nas duas últimas versões (v4.1 e v4.2)</summary>

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

## Testes de desempenho

Em **Performance Tests**, selecione um modelo de chat em execução e defina os níveis de concorrência para o teste. Compare a vazão, o tempo até o primeiro token e o tempo por token gerado antes de escolher a configuração. [Guia de testes de desempenho](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![Gráficos de vazão e latência em diferentes níveis de concorrência](../assets/guide-benchmark-en.webp)

*Exemplo de uma execução local do Qwen3.5-0.8B. O desempenho depende do hardware, do motor e da carga.*

## Como usar o Xinference

| Objetivo | Por onde começar |
| --- | --- |
| Instalar no Linux, macOS ou Windows | [Instalação com um comando](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guia de instalação](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| Executar em um contêiner ou no Kubernetes | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| Cadastrar um modelo próprio | [Modelos personalizados](https://inference.readthedocs.io/en/latest/models/custom.html) |
| Conectar uma aplicação | [APIs compatíveis com a OpenAI](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**Aplicações e frameworks** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Implante o Xinference Enterprise na sua própria nuvem ou no seu data center. O Xinference Cloud oferece implantações dedicadas e gerenciadas, e a Model API disponibiliza endpoints para modelos hospedados. [Visão geral do Enterprise](https://xinference.co/enterprise.html)

**Contato comercial** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## Participe do projeto

| Plataforma | Para que serve |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | Relatar problemas e sugerir recursos. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Fazer perguntas e trocar ideias com outros usuários. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Conversar com a comunidade do Xinference. |
| [X](https://x.com/Xinference_ai) | Acompanhar notícias e lançamentos. |

## Citação

Se você usar o Xinference na sua pesquisa, cite [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

Agradecemos a todas as pessoas que contribuem para o Xinference. Abra seu primeiro pull request para adicionar um modelo, melhorar a documentação ou corrigir um problema. [Guia de contribuição](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Mosaico de avatares dos colaboradores do Xinference](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
