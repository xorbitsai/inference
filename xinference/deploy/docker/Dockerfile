FROM vllm/vllm-openai:latest

COPY . /opt/inference
WORKDIR /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y curl procps git libgl1 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib

ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip && \
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    # use pre-built whl package for llama-cpp-python, otherwise may core dump when init llama in some envs
    pip install "llama-cpp-python>=0.2.82" -i https://abetlen.github.io/llama-cpp-python/whl/cu124 && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements.txt && \
    pip install -i "$PIP_INDEX" --no-deps sglang && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    # clean packages
    pip cache purge

ENTRYPOINT []
