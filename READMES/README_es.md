# ![Xinference](../assets/xinference-banner-platform-types.webp)

**Xinference (Xorbits Inference) es una plataforma de inferencia distribuida de código abierto.**

Despliega y ejecuta modelos en hardware heterogéneo, desde un equipo local hasta un clúster distribuido. Xinference incluye más de 300 modelos integrados de lenguaje, visión, embeddings, reranking, imagen, audio, vídeo y generación de mundos (experimental). Registra tus propios modelos, incluidos los de aprendizaje automático clásico mediante el tipo Flexible. Gestiona los despliegues en la interfaz web y usa los modelos con las API compatibles con OpenAI, el SDK de Python o la CLI.

[Inicio rápido](#inicio-rápido) · [Funciones principales](#funciones-principales) · [Comparación de plataformas](#comparación-de-plataformas) · [Autoalojamiento](https://inference.readthedocs.io/en/latest/getting_started/installation.html) · [Documentación](https://inference.readthedocs.io/en/latest/) · [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Versiones](https://github.com/xorbitsai/inference/releases) · [Xinference Enterprise](https://xinference.co/enterprise.html)

[![Versión de GitHub](https://img.shields.io/github/v/release/xorbitsai/inference?logo=github&label=release&style=flat-square)](https://github.com/xorbitsai/inference/releases)
[![Versión de PyPI](https://img.shields.io/pypi/v/xinference?logo=pypi&style=flat-square)](https://pypi.org/project/xinference/)
[![Etiqueta de Docker](https://img.shields.io/docker/v/xprobe/xinference?sort=semver&logo=docker&label=docker%20tag&style=flat-square)](https://hub.docker.com/r/xprobe/xinference/tags)
[![Versiones de Python](https://img.shields.io/pypi/pyversions/xinference?logo=python&style=flat-square)](https://pypi.org/project/xinference/)
[![CI de Python](https://img.shields.io/github/actions/workflow/status/xorbitsai/inference/python.yaml?branch=main&label=CI&style=flat-square)](https://github.com/xorbitsai/inference/actions/workflows/python.yaml)

[![Licencia Apache 2.0](https://img.shields.io/github/license/xorbitsai/inference?label=license&style=flat-square)](https://github.com/xorbitsai/inference/blob/main/LICENSE)
[![Estrellas de GitHub](https://img.shields.io/github/stars/xorbitsai/inference?logo=github&label=stars&style=flat-square)](https://github.com/xorbitsai/inference/stargazers)
[![Forks de GitHub](https://img.shields.io/github/forks/xorbitsai/inference?logo=github&label=forks&style=flat-square)](https://github.com/xorbitsai/inference/forks)
[![Colaboradores](https://img.shields.io/github/contributors/xorbitsai/inference?style=flat-square)](https://github.com/xorbitsai/inference/graphs/contributors)
[![Descargas de Docker](https://img.shields.io/docker/pulls/xprobe/xinference?logo=docker&label=docker%20pulls&style=flat-square)](https://hub.docker.com/r/xprobe/xinference)
[![Comunidad en Discord](https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/Xw9tszSkr5)
[![Comunidad en Telegram](https://img.shields.io/badge/Telegram-join-26A5E4?logo=telegram&logoColor=white&style=flat-square)](https://t.me/+nCNpwmySwk9iYmI1)
[![Sigue a Xinference en X](https://img.shields.io/badge/X-follow-171717?logo=x&logoColor=white&style=flat-square)](https://x.com/Xinference_ai)

**Idiomas** · [English](../README.md) · [日本語](./README_ja_JP.md) · [한국어](./README_ko.md) · [Deutsch](./README_de.md) · [Français](./README_fr.md) · **Español** · [Italiano](./README_it.md) · [Português](./README_pt_BR.md) · [繁體中文](./README_zh_TW.md) · [简体中文](./README_zh_CN.md)

## Inicio rápido

### 1. Iniciar el servidor

En el primer inicio del modelo, Xinference lo descarga e instala las dependencias necesarias para el motor.

```bash
pip install xinference
xinference-local --host 127.0.0.1
```

Abre [http://127.0.0.1:9997](http://127.0.0.1:9997). En una instancia nueva, usa el token de configuración que aparece en el registro del servidor para crear una cuenta de administrador. Luego inicia sesión. [Configuración inicial](https://inference.readthedocs.io/en/latest/user_guide/auth_system.html#initial-admin-account)

### 2. Desplegar un modelo de chat

En **Launch Model**, busca `qwen3.5` y abre el formulario de despliegue. Haz clic en **Recommend configuration** para obtener una configuración para tu hardware. Revísala y pulsa **Deploy**.

![Formulario completo de Qwen3.5 con la configuración recomendada aplicada](../assets/guide-launch-recommend-auto.webp)

### 3. Enviar la primera solicitud

Abre el modelo en **Running Models** para probarlo en la interfaz web. La primera instancia usa `qwen3.5` como UID del modelo. Si aparece otro UID, úsalo en el campo `model` de la solicitud. Crea una clave en **API Key Management** para llamar a la API.

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

![Interfaz de chat de Qwen3.5 con una respuesta completa](../assets/guide-chat-en.webp)

## Funciones principales

| Área | Compatibilidad de Xinference |
| --- | --- |
| Catálogo de modelos | Más de 300 modelos integrados de lenguaje, visión, embeddings, reranking, imagen, audio, vídeo y generación de mundos (experimental). Registra tus propios modelos, incluidos los de aprendizaje automático clásico mediante el tipo Flexible. [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) |
| Motores y hardware | Según el LLM, elige vLLM, SGLang, TRT-LLM, Transformers, llama.cpp o MLX. La compatibilidad con CPU, GPU, Apple Silicon y algunas configuraciones MLU depende del motor. [Motores de inferencia](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Compatibilidad con MLU](https://github.com/xorbitsai/inference/pull/3693) |
| Inferencia distribuida | Ejecuta los modelos en un equipo, añade workers y réplicas, o ejecuta un modelo grande en varios workers con un motor compatible. Los despliegues compatibles permiten separar el prefill y la decodificación en hardware heterogéneo. [Inferencia distribuida](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Separación P/D](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html) |
| API y herramientas | Endpoints compatibles con OpenAI y Anthropic, con streaming y llamadas a herramientas según el modelo. Despliega y gestiona los modelos desde la interfaz web, la CLI o el SDK de Python. [Guía del cliente](https://inference.readthedocs.io/en/latest/user_guide/client_api.html) · [Llamadas a herramientas](https://inference.readthedocs.io/en/latest/models/model_abilities/tools.html) |
| Rendimiento y operaciones | Mide el rendimiento de los modelos de chat activos desde la interfaz web. Los despliegues compatibles pueden compartir la caché de prefijos con Xavier, cambiar determinados ajustes en ejecución y recargar las réplicas de una en una. [Pruebas de rendimiento](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html) · [Configuración en ejecución](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-without-reloading-weights) |
| Idiomas | La interfaz web está disponible en inglés, chino simplificado, chino tradicional, japonés y coreano. El README está disponible en diez idiomas. |

## Comparación de plataformas

| Función | **Xinference** | LocalAI | KServe | NVIDIA Dynamo |
| --- | :---: | :---: | :---: | :---: |
| **Ejecución local sin Kubernetes** | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Sin compatibilidad documentada][compare-undocumented] | ![Compatible][compare-supported] |
| **Inferencia en varios nodos** | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Compatible][compare-supported] |
| **CPU / GPU / Apple Metal** | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Parcial o mediante integración][compare-partial] | ![Parcial o mediante integración][compare-partial] |
| **llama.cpp / vLLM / SGLang** | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Parcial o mediante integración][compare-partial] | ![Parcial o mediante integración][compare-partial] |
| **Prefill y decodificación entre CUDA y Metal** | ![Compatible][compare-supported] | ![Sin compatibilidad documentada][compare-undocumented] | ![Sin compatibilidad documentada][compare-undocumented] | ![Sin compatibilidad documentada][compare-undocumented] |
| **Distintos motores bajo un mismo ID de modelo** | ![Compatible][compare-supported] | ![Parcial o mediante integración][compare-partial] | ![Parcial o mediante integración][compare-partial] | ![Parcial o mediante integración][compare-partial] |
| **Reconfiguración gradual con reutilización de pesos en la GPU** | ![Compatible][compare-supported] | ![Sin compatibilidad documentada][compare-undocumented] | ![Sin compatibilidad documentada][compare-undocumented] | ![Sin compatibilidad documentada][compare-undocumented] |
| Catálogo de modelos | ![Compatible][compare-supported] 300+ | ![Compatible][compare-supported] | ![Parcial o mediante integración][compare-partial] | ![Parcial o mediante integración][compare-partial] |
| Despliegue de modelos desde la interfaz web | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Parcial o mediante integración][compare-partial] | ![Sin compatibilidad documentada][compare-undocumented] |
| Embeddings y reranking | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Compatible][compare-supported] |
| Generación de imágenes y vídeo | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Sin compatibilidad documentada][compare-undocumented] | ![Compatible][compare-supported] |
| Reconocimiento y síntesis de voz | ![Compatible][compare-supported] | ![Compatible][compare-supported] | ![Sin compatibilidad documentada][compare-undocumented] | ![Parcial o mediante integración][compare-partial] |

![][compare-supported] Compatible · ![][compare-partial] Parcial o mediante integración · ![][compare-undocumented] Sin compatibilidad documentada

<details>
<summary>Alcance y fuentes</summary>

La comparación se basa en las funciones documentadas de las versiones de código abierto. Los motores y el hardware compatibles dependen del modelo. El prefill y la decodificación entre CUDA y Metal requieren un modelo y un par de motores compatibles.

Xinference puede servir en paralelo réplicas con distintos motores bajo un mismo ID de modelo. LocalAI usa una cadena de conmutación por error ordenada bajo un mismo nombre de modelo y envía las solicitudes al primer destino disponible. KServe y Dynamo pueden dirigir solicitudes entre despliegues separados. La reutilización de pesos en la GPU se aplica a las configuraciones compatibles de vLLM y SGLang.

- **Xinference** · [Catálogo de modelos](https://inference.readthedocs.io/en/latest/models/builtin/index.html) · [Motores](https://inference.readthedocs.io/en/latest/user_guide/backends.html) · [Inferencia distribuida](https://inference.readthedocs.io/en/latest/user_guide/distributed_inference.html) · [Separación P/D entre CUDA y Metal](https://inference.readthedocs.io/en/latest/user_guide/pd_separation.html#nvidia-and-mlx-cross-engine-p-d) · [Motores de las réplicas](https://inference.readthedocs.io/en/latest/user_guide/launch.html#replica) · [Reconfiguración gradual](https://inference.readthedocs.io/en/latest/user_guide/launch.html#reconfigure-and-upgrade-running-models)
- **LocalAI** · [Modelos y despliegue](https://localai.io/docs/features/index.html) · [Motores y hardware](https://localai.io/docs/reference/index.print.html) · [Modo distribuido](https://localai.io/docs/features/distributed-mode/) · [Alias y conmutación por error](https://localai.io/docs/features/model-failover/index.html)
- **KServe** · [Despliegue y hardware](https://kserve.github.io/website/docs/intro) · [Motores generativos integrados](https://kserve.github.io/website/docs/model-serving/generative-inference/overview) · [Catálogo de Kubeflow](https://www.kubeflow.org/docs/components/hub/installation/) · [Interfaz web de Kubeflow](https://www.kubeflow.org/docs/ecosystem/kserve/webapp/) · [Despliegues canary](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/canary-rollout-observability)
- **NVIDIA Dynamo** · [Despliegue, hardware y motores](https://docs.dynamo.nvidia.com/dynamo) · [Recetas de despliegue](https://docs.dynamo.nvidia.com/dynamo/recipes/browse) · [Generación de imágenes, vídeo y voz](https://docs.dynamo.nvidia.com/dynamo/diffusion/overview) · [Embeddings y reranking](https://docs.dynamo.nvidia.com/dynamo/reference/releases/v1-6-0) · [Enrutamiento entre despliegues](https://docs.dynamo.nvidia.com/dynamo/reference/components/global-router-configuration)

</details>

[compare-supported]: ../assets/compare-supported.svg
[compare-partial]: ../assets/compare-partial.svg
[compare-undocumented]: ../assets/compare-undocumented.svg

## Novedades

[Notas de la versión v4.2.0](https://github.com/xorbitsai/inference/releases/tag/v4.2.0)

<details>
<summary>Modelos integrados en las dos últimas versiones (v4.1 y v4.2)</summary>

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

## Pruebas de rendimiento

En **Performance Tests**, elige un modelo de chat en ejecución y define los niveles de concurrencia que quieres probar. Compara el rendimiento, el tiempo hasta el primer token y el tiempo por token de salida antes de elegir la configuración. [Guía de pruebas de rendimiento](https://inference.readthedocs.io/en/latest/user_guide/benchmarks.html)

![Gráficos de rendimiento y latencia a distintos niveles de concurrencia](../assets/guide-benchmark-en.webp)

*Ejemplo de una ejecución local de Qwen3.5-0.8B. Los resultados dependen del hardware, el motor y la carga.*

## Usar Xinference

| Objetivo | Por dónde empezar |
| --- | --- |
| Instalar en Linux, macOS o Windows | [Instalación con un comando](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guía de instalación](https://inference.readthedocs.io/en/latest/getting_started/installation.html) |
| Ejecutar en un contenedor o en Kubernetes | [Docker](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html) · [Kubernetes](https://inference.readthedocs.io/en/latest/getting_started/using_kubernetes.html) |
| Registrar un modelo propio | [Modelos personalizados](https://inference.readthedocs.io/en/latest/models/custom.html) |
| Conectar una aplicación | [API compatibles con OpenAI](https://inference.readthedocs.io/en/latest/user_guide/client_api.html#openai-client) · [Xagent](https://github.com/xorbitsai/xagent) · [Dify](https://marketplace.dify.ai/plugin/langgenius/xinference) |

**Aplicaciones y frameworks** · [LangChain](https://reference.langchain.com/python/langchain-community/llms/xinference/Xinference) · [LlamaIndex](https://developers.llamaindex.ai/python/framework/integrations/llm/xinference_local_deployment/) · [FastGPT](https://doc.fastgpt.io/en/self-host/custom-models/xinference) · [RAGFlow](https://github.com/infiniflow/ragflow) · [MaxKB](https://github.com/1Panel-dev/MaxKB) · [Chatbox](https://chatboxai.app/)

### Xinference Enterprise

Despliega Xinference Enterprise en tu propia nube o centro de datos. Xinference Cloud ofrece despliegues dedicados gestionados y Model API proporciona endpoints de modelos alojados. [Información sobre Enterprise](https://xinference.co/enterprise.html)

**Consultas comerciales** · [info@xinference.co](mailto:info@xinference.co?subject=%5BGitHub%5DBusiness%20License%20Inquiry)

## Participar en el proyecto

| Plataforma | Para qué sirve |
| --- | --- |
| [GitHub Issues](https://github.com/xorbitsai/inference/issues) | Comunicar errores y proponer funciones. |
| [Discord](https://discord.gg/Xw9tszSkr5) | Hacer preguntas y conversar con otros usuarios. |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1) | Chatear con la comunidad de Xinference. |
| [X](https://x.com/Xinference_ai) | Seguir las novedades y las publicaciones del proyecto. |

## Cita

Si usas Xinference en tu investigación, cita [Xinference: Making Large Model Serving Easy](https://aclanthology.org/2024.emnlp-demo.30/)

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

Gracias a quienes contribuyen a Xinference. Abre tu primera pull request para añadir un modelo, mejorar la documentación o corregir un error. [Guía de contribución](https://inference.readthedocs.io/en/latest/development/contributing_codebase.html)

[![Mosaico de avatares de quienes contribuyen a Xinference](https://contrib.rocks/image?repo=xorbitsai/inference)](https://github.com/xorbitsai/inference/graphs/contributors)
