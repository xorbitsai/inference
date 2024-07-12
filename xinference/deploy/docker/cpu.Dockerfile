FROM continuumio/miniconda3:23.10.0-1

COPY . /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y build-essential curl procps git libgl1 \
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
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements_cpu.txt && \
    pip install "llama-cpp-python==0.2.77" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu && \
    cd /opt/inference && \
    python setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    # clean packages
    pip cache purge
