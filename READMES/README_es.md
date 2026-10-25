<div align="center">
<img src="../assets/xorbits-logo.png"  width="180px" alt="xorbits" />

# Xorbits Inference: Servir modelos con facilidad 🤖

<p align="center">
	<a href="https://xinference.co">Xinference Enterprise</a> ·
	<a href="https://inference.readthedocs.io/en/latest/getting_started/installation.html#installation">Self-Hosting</a> ·
	<a href="https://inference.readthedocs.io/">Documentación</a>
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
	<a href="./README_es.md"><img alt="Español" src="https://img.shields.io/badge/Español-454545?style=for-the-badge"></a>
	<a href="./README_it.md"><img alt="Italiano" src="https://img.shields.io/badge/Italiano-d9d9d9?style=for-the-badge"></a>
	<a href="./README_pt_BR.md"><img alt="Português" src="https://img.shields.io/badge/Português-d9d9d9?style=for-the-badge"></a>
	<a href="./README_zh_TW.md"><img alt="繁體中文" src="https://img.shields.io/badge/繁體中文-d9d9d9?style=for-the-badge"></a>
	<a href="./README_zh_CN.md"><img alt="简体中文" src="https://img.shields.io/badge/简体中文-d9d9d9?style=for-the-badge"></a>
</p>
</div>
<br />


Xorbits Inference (Xinference) es una biblioteca potente y versátil para modelos de lenguaje, reconocimiento de voz y modelos multimodales. Con Xorbits Inference puedes desplegar tu propio modelo o modelos avanzados integrados con un solo comando y ofrecerlos como servicio. Investigadores, desarrolladores y científicos de datos pueden aprovechar al máximo las capacidades de los modelos de IA modernos.

<div align="center">
<i><a href="https://discord.gg/Xw9tszSkr5">👉 ¡Únete a nuestra comunidad de Discord!</a> · <a href="https://t.me/+nCNpwmySwk9iYmI1">Únete a nuestro grupo de Telegram</a></i>
</div>

## 🔥 Temas destacados
### Mejora del framework
- Xinference 4.0.0 está disponible con Xavier multiengine, separación P/D heterogénea y reconfiguración de modelos en tiempo de ejecución: [Notas de la versión](https://github.com/xorbitsai/inference/releases/tag/v4.0.0)
- Implementación nativa de agentes: Xinference se integra con [Xagent](https://github.com/xorbitsai/xagent) y permite planificación dinámica, uso de herramientas e inferencias multietapa autónomas, superando los límites de las tuberías estáticas.
- Batching automático: múltiples solicitudes concurrentes se agrupan automáticamente para aumentar significativamente el rendimiento.: [#4197](https://github.com/xorbitsai/inference/pull/4197)
- [Xllamacpp](https://github.com/xorbitsai/xllamacpp): los nuevos bindings de Python para llama.cpp, mantenidos por el equipo de Xinference, soportan batching continuo y son más aptos para producción.: [#2997](https://github.com/xorbitsai/inference/pull/2997)
- Inferencia distribuida: los modelos pueden ejecutarse entre workers: [#2877](https://github.com/xorbitsai/inference/pull/2877)
- Mejoras en vLLM: compartir el KV-cache entre réplicas: [#2732](https://github.com/xorbitsai/inference/pull/2732)
### Nuevos modelos
- Soporte integrado para [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein): [#5792](https://github.com/xorbitsai/inference/pull/5792)
- Soporte integrado para la serie FIBO 2.0 ([Dev](https://huggingface.co/briaai/Fibo-2.0-dev), [Edit Turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)): [#5790](https://github.com/xorbitsai/inference/pull/5790)
- Soporte integrado para la serie Ming-3.0 TTS ([Tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny), [0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)): [#5788](https://github.com/xorbitsai/inference/pull/5788)
- Soporte integrado para [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0): [#5764](https://github.com/xorbitsai/inference/pull/5764)
- Soporte integrado para [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT): [#5755](https://github.com/xorbitsai/inference/pull/5755)
- Soporte integrado para [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): [#5753](https://github.com/xorbitsai/inference/pull/5753)
- Soporte integrado para la serie Iris-3B ([Base](https://huggingface.co/speridlabs/iris-3b), [Turbo](https://huggingface.co/speridlabs/iris-3b/tree/main/turbo)): [#5686](https://github.com/xorbitsai/inference/pull/5686)
- Soporte integrado para la serie MiniCPM-V-4.7 ([1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B), [35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B)): [#5676](https://github.com/xorbitsai/inference/pull/5676)
- Soporte integrado para la serie Qwen3.8-Flash-Next ([Base](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), [FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)): [#5673](https://github.com/xorbitsai/inference/pull/5673)
- Soporte integrado para la serie GLM-5.3 ([BF16](https://huggingface.co/zai-org/GLM-5.3-BF16), [FP8](https://huggingface.co/zai-org/GLM-5.3)): [#5666](https://github.com/xorbitsai/inference/pull/5666)
- Soporte integrado para [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo): [#5665](https://github.com/xorbitsai/inference/pull/5665)
- Soporte integrado para [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash): [#5661](https://github.com/xorbitsai/inference/pull/5661)
- Soporte integrado para [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2): [#5644](https://github.com/xorbitsai/inference/pull/5644)
- Soporte integrado para [Confucius4-TTS](https://huggingface.co/netease-youdao/Confucius4-TTS): [#5591](https://github.com/xorbitsai/inference/pull/5591)
- Soporte integrado para [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1): [#5588](https://github.com/xorbitsai/inference/pull/5588)
### Integraciones
- [Xagent](https://github.com/xorbitsai/xagent): plataforma de agentes enterprise con planificación, memoria e integración de herramientas.
- [Dify](https://docs.dify.ai/advanced/model-configuration/xinference): plataforma LLMOps para construir aplicaciones rápidamente con visualización y control.
- [FastGPT](https://github.com/labring/FastGPT): plataforma de conocimiento basada en LLM para procesamiento de datos y llamadas a modelos.
- [RAGFlow](https://github.com/infiniflow/ragflow): motor RAG open-source para comprensión profunda de documentos.
- [MaxKB](https://github.com/1Panel-dev/MaxKB): asistente de base de conocimiento open-source con integración RAG.

## Funcionalidades principales
🌟 Servir modelos con facilidad: Simplifica el despliegue de LLMs, reconocimiento de voz y modelos multimodales. Los modelos de prueba y producción se pueden configurar y desplegar con un solo comando.

⚡️ Modelos de vanguardia accesibles: Prueba modelos integrados con un solo comando. Xinference ofrece acceso a modelos Open-Source avanzados.

🖥 Aprovechamiento de hardware heterogéneo: Utiliza GPU y CPU (por ejemplo, mediante [ggml](https://github.com/ggerganov/ggml)) para acelerar la inferencia.

⚙️ APIs y interfaces flexibles: API RESTful compatible con OpenAI (incluyendo Function Calling), RPC, CLI, Web UI y más.

🌐 Despliegue distribuido: Facilita la distribución de la inferencia a través de varios dispositivos y máquinas.

🔌 Integraciones de terceros: Integración con [LangChain](https://python.langchain.com/docs/integrations/providers/xinference), [LlamaIndex], [Dify], [Chatbox], etc.

## Por qué Xinference
| Función                                      | Xinference | FastChat | OpenLLM | RayLLM |
|----------------------------------------------|------------|----------|---------|--------|
| API RESTful compatible con OpenAI             | ✅         | ✅        | ✅       | ✅      |
| Integración vLLM                              | ✅         | ✅        | ✅       | ✅      |
| Diversos motores de inferencia (GGML, TensorRT)| ✅         | ❌        | ✅       | ✅      |
| Diversas plataformas (CPU, Metal)              | ✅         | ✅        | ❌       | ❌      |
| Despliegue en clúster multi-nodo              | ✅         | ❌        | ❌       | ✅      |
| Modelos de imagen (Texto→Imagen)               | ✅         | ✅        | ❌       | ❌      |
| Modelos de embedding de texto                  | ✅         | ❌        | ❌       | ❌      |
| Modelos multimodales                            | ✅         | ❌        | ❌       | ❌      |
| Modelos de voz                                  | ✅         | ❌        | ❌       | ❌      |
| Funcionalidad OpenAI (Function Calling)        | ✅         | ❌        | ❌       | ❌      |

## Cómo usar Xinference

- Self-Hosting Xinference Community Edition
	Sigue la [guía de inicio](#inicio) para poner en marcha Xinference localmente. Más detalles en la documentación: https://inference.readthedocs.io/.

- Xinference para empresas
	**Xinference Enterprise** es la edición comercial para equipos que ejecutan modelos en producción. Puede desplegarla en su propia nube o centro de datos, o elegir un despliegue dedicado gestionado con **Xinference Cloud**. La **Model API** alojada le permite acceder a los modelos compatibles mediante una API compatible con OpenAI sin gestionar la infraestructura de inferencia. Más información en [xinference.co](https://xinference.co).
	Para consultas empresariales, [envíenos un correo](mailto:info@xinference.co?subject=[GitHub]Business%20License%20Inquiry).

## Mantente al día

Dale una estrella a Xinference en GitHub para recibir actualizaciones de lanzamientos.

![star-us](../assets/stay_ahead.gif)

## Inicio

* [Documentación](https://inference.readthedocs.io/en/latest/index.html)
* [Modelos integrados](https://inference.readthedocs.io/en/latest/models/builtin/index.html)
* [Modelos personalizados](https://inference.readthedocs.io/en/latest/models/custom.html)
* [Documentación de deployment](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html)

### Quickstart

**pip**

```bash
pip install xinference
xinference-local
```

**Instalación con un comando**

Linux/macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/xorbitsai/inference/main/scripts/install.ps1 | iex
```

[Guía de instalación](https://inference.readthedocs.io/en/latest/getting_started/installation.html#one-line-install) · [Guía de uso](https://inference.readthedocs.io/en/latest/getting_started/using_xinference.html#run-xinference-locally)

### Docker

Usuarios con GPU NVIDIA pueden usar la imagen Docker de Xinference. Asegúrate de tener Docker y CUDA antes de la instalación.

```bash
docker run --name xinference -d -p 9997:9997 -e XINFERENCE_HOME=/data -v </on/your/host>:/data --gpus all xprobe/xinference:latest xinference-local -H 0.0.0.0
```

### K8s (Helm)

Tras habilitar GPU en tu clúster Kubernetes, instala así:

```
# Añadir repositorio
helm repo add xinference https://xorbitsai.github.io/xinference-helm-charts

# Actualizar índice y comprobar versiones
helm repo update xinference
helm search repo xinference/xinference --devel --versions

# Instalar Xinference
helm install xinference xinference/xinference -n xinference --version 0.0.1-v<xinference_release_version>
```

Más opciones de K8s en la documentación.

![web UI](../assets/screenshot.png)

## Contribuir

| Plataforma                                                                 | Propósito                                |
|---------------------------------------------------------------------------|------------------------------------------|
| [Github Issues](https://github.com/xorbitsai/inference/issues)             | Reporte de bugs y solicitudes de features |
| [Discord](https://discord.gg/Xw9tszSkr5)                                  | Colaboración con otros usuarios          |
| [Telegram](https://t.me/+nCNpwmySwk9iYmI1)                                | Discusiones con la comunidad              |
| [Twitter](https://twitter.com/xorbitsio)                                  | Noticias y anuncios                       |

## Citación

Si este proyecto te fue útil, cítalo así:

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

## Historial de estrellas

[![Star History Chart](https://star-history.dera.page/svg?repos=xorbitsai/inference&type=Date)](https://star-history.dera.page/#xorbitsai/inference&Date)
