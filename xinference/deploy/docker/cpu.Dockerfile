FROM continuumio/miniconda3:23.10.0-1

COPY . /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y build-essential curl procps git \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

ARG PIP_INDEX=https://pypi.org/simple
RUN python -m pip install --upgrade -i "$PIP_INDEX" pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -i "$PIP_INDEX" \
      "xoscar>=0.2.1" \
      "gradio>=3.39.0" \
      pillow \
      click \
      "tqdm>=4.27" \
      tabulate \
      requests \
      pydantic \
      fastapi \
      uvicorn \
      "huggingface-hub>=0.19.4,<1.0" \
      typing_extensions \
      "fsspec>=2023.1.0,<=2023.10.0" \
      s3fs \
      "modelscope>=1.10.0" \
      "sse_starlette>=1.6.5" \
      "openai>1" \
      "python-jose[cryptography]" \
      "passlib[bcrypt]" \
      "aioprometheus[starlette]>=23.12.0" \
      pynvml \
      async-timeout \
      "transformers>=4.34.1" \
      "accelerate>=0.20.3" \
      sentencepiece \
      transformers_stream_generator \
      bitsandbytes \
      protobuf \
      einops \
      tiktoken \
      "sentence-transformers>=2.3.1" \
      diffusers \
      controlnet_aux \
      orjson \
      auto-gptq \
      optimum && \
    pip install -i "$PIP_INDEX" -U chatglm-cpp && \
    pip install -i "$PIP_INDEX" -U llama-cpp-python && \
    cd /opt/inference && \
    python setup.py build_web && \
    pip install -i "$PIP_INDEX" --no-deps "."
