FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

COPY . /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y curl procps \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/compat:$LD_LIBRARY_PATH

ARG PIP_INDEX=https://pypi.org/simple
RUN python -m pip install --upgrade -i "$PIP_INDEX" pip && \
    CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install -i "$PIP_INDEX" -U chatglm-cpp && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -i "$PIP_INDEX" -U llama-cpp-python && \
    cd /opt/inference && \
    python setup.py build_web && \
    pip install -i "$PIP_INDEX" ".[all]"
