FROM vllm/vllm-openai:v0.6.0

COPY . /opt/inference
WORKDIR /opt/inference

ENV NVM_DIR /usr/local/nvm
ENV NODE_VERSION 14.21.1

RUN apt-get -y update \
  && apt install -y wget curl procps git libgl1 \
  # upgrade libstdc++ and libc for llama-cpp-python
  && printf "\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
  && apt-get -y update \
  && apt-get install -y --only-upgrade libstdc++6 && apt install -y libc6 libnuma1 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

RUN wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run && \
    chmod +x cuda_12.4.0_550.54.14_linux.run && \
    ./cuda_12.4.0_550.54.14_linux.run --silent --toolkit --override && \
    rm -f cuda_12.4.0_550.54.14_linux.run

ENV PATH $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD TRUE

ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip setuptools wheel&& \
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-base.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-ml.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-models.txt && \
    pip install -i "$PIP_INDEX" transformers==4.53.2 && \
    pip install -i "$PIP_INDEX" --no-deps sglang==0.4.6.post5 && \
    pip install WeTextProcessing==1.0.4.1 --no-deps && \
    pip install importlib_resources && \
    pip install -i "$PIP_INDEX" sgl-kernel==0.1.8 && \
    pip install torch==2.7.1 && \
    pip uninstall flashinfer -y && \
    pip install flashinfer-python==0.2.9rc2 && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.18/flash_attn-2.7.4+cu124torch2.7-cp310-cp310-linux_x86_64.whl && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    pip uninstall xllamacpp -y && \
    pip install "xllamacpp>=0.2.0" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124 && \
    # clean packages
    pip cache purge

# Install Miniforge3 (only for FFmpeg, do not replace system Python)
RUN apt-get install -y wget && \
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh

RUN /opt/conda/bin/conda create -n ffmpeg-env -c conda-forge 'ffmpeg<7' -y && \
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffprobe /usr/local/bin/ffprobe && \
    /opt/conda/bin/conda clean --all -y

# Overwrite the entrypoint of vllm's base image
ENTRYPOINT []
CMD ["/bin/bash"]
