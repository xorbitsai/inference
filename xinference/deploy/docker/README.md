# Xinference Docker Images

Xinference 3.0 Docker images are slim runtime images. They keep only the
Xinference service stack and virtual environment tooling in the base image.
Model and engine dependencies are installed on demand into per-model,
per-engine virtual environments when a model is launched.

## Base Images

- GPU image: `python:3.12-slim-bookworm`
- CPU image: `python:3.12-slim-bookworm`
- aarch64 image: `python:3.12-slim-bookworm`

The Dockerfiles use a separate web build stage with Node.js. Node.js is not
kept in the runtime image.

## Runtime Dependencies

The runtime image keeps:

- Python 3.12, pip, setuptools, and wheel.
- `uv`, used by the virtual environment backend.
- Xinference service dependencies such as xoscar, FastAPI, Uvicorn, Gradio,
  auth, metrics, OpenTelemetry, and model download clients.
- Minimal system tools and libraries required by the service process and
  common runtime operations: `git`, `curl`, `procps`, `sqlite3`, `rsync`,
  `ffmpeg`, `libgl1`, `libglib2.0-0`, `libgomp1`, and `libsndfile1`.

The runtime image does not preinstall vLLM, SGLang, Transformers,
sentence-transformers, xllamacpp, Diffusers, PyTorch, image/audio/video model
packages, or engine-specific acceleration packages. Those belong in model
virtual environments.

## Validation

For Docker changes, build all published image variants and record their sizes:

```bash
docker build -t xinference:slim-gpu -f xinference/deploy/docker/Dockerfile .
docker build -t xinference:slim-cpu -f xinference/deploy/docker/Dockerfile.cpu .
docker buildx build --platform linux/arm64 --load \
  -t xinference:slim-aarch64 \
  -f xinference/deploy/docker/Dockerfile.aarch64 .

docker image inspect xinference:slim-gpu \
  --format 'gpu={{.Size}} bytes'
docker image inspect xinference:slim-cpu \
  --format 'cpu={{.Size}} bytes'
docker image inspect xinference:slim-aarch64 \
  --format 'aarch64={{.Size}} bytes'
```

At runtime, the first launch of a model should create a virtual environment
under `.xinference/virtualenv/v4/{model_name}/{model_engine}/{python_version}`
and install the engine dependencies declared by that model family.
