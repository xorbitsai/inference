FROM vllm/vllm-openai:v0.11.2

COPY . /opt/inference
WORKDIR /opt/inference

ENV NVM_DIR=/usr/local/nvm
ENV NODE_VERSION=14.21.1

# Install system dependencies and Node.js (libfst-dev should be able to solve the errors of pyini)
RUN apt-get -y update \
  && apt install -y wget curl procps git libgl1 libfst-dev cmake libssl-dev \
  && printf "\ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse" >> /etc/apt/sources.list \
  && apt-get -y update \
  && apt-get install -y --only-upgrade libstdc++6 && apt install -y libc6 \
  && mkdir -p $NVM_DIR \
  && curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash \
  && . $NVM_DIR/nvm.sh \
  && nvm install $NODE_VERSION \
  && nvm alias default $NODE_VERSION \
  && nvm use default \
  && apt-get -yq clean

ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib
# ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Install pip dependencies
ARG LLAMA_CPP_USE_CUDA=true
ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip setuptools wheel && \
    apt-get -y update && \
    ( wget -O openfst-1.7.2.tar.gz http://www.openslr.org/resources/2/openfst-1.7.2.tar.gz \
      || wget -O openfst-1.7.2.tar.gz https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz ) && \
    tar zxvf openfst-1.7.2.tar.gz && cd openfst-1.7.2 && \
    ./configure --enable-shared --enable-static && make -j$(nproc) && make install && ldconfig && \
    CPLUS_INCLUDE_PATH=/usr/local/include LIBRARY_PATH=/usr/local/lib LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    pip install -i "$PIP_INDEX" pynini==2.1.6.post1 && \
    apt install -y wget curl procps git libgl1 rsync sqlite libpcre3 libpcre3-dev dmidecode libssl-dev perl make build-essential zlib1g-dev && \
    apt-get -yq clean && \
    # use pre-built whl package for llama-cpp-python, otherwise may core dump when init llama in some envs
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    pip install -i "$PIP_INDEX" "cython>=0.29" && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-base.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-ml.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements/requirements-models.txt && \
    pip install -i "$PIP_INDEX" transformers==4.55.2 && \
    pip install -i "$PIP_INDEX" --no-deps sglang==0.5.6 && \
    pip install -i "$PIP_INDEX" sgl-kernel==0.3.18.post2 && \
    pip install -i "$PIP_INDEX" wetext && \
    pip uninstall flashinfer -y && \
    pip install -i "$PIP_INDEX" flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 && \
    pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.5.3/flashinfer_jit_cache-0.5.3+cu129-cp39-abi3-manylinux_2_28_x86_64.whl && \ 
    pip install -i "$PIP_INDEX" SQLAlchemy==1.4.54 && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    pip uninstall xllamacpp -y && \
    wget https://github.com/xorbitsai/xllamacpp/releases/download/v0.2.0-cu124/xllamacpp-0.2.0-cp312-cp312-linux_x86_64.whl && \
    pip install xllamacpp-0.2.0-cp312-cp312-linux_x86_64.whl && \
    # clean packages
    pip cache purge

# Install Miniforge3 and FFmpeg
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/download/4.12.0-0/Miniforge3-4.12.0-0-Linux-x86_64.sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh

# When installing the Conda environment, only FFmpeg should be installed to avoid modifying the system Python
RUN /opt/conda/bin/conda create -n ffmpeg-env -c conda-forge 'ffmpeg<7' -y && \
    #Create a soft link to the system path
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /opt/conda/envs/ffmpeg-env/bin/ffprobe /usr/local/bin/ffprobe && \
    # Clear the Conda cache
    /opt/conda/bin/conda clean --all -y

# The pre-release version used should be noted for date changes
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --no-deps && \
    pip install triton==3.4.0 && \
    pip install torchcodec==0.9.0 && \
    pip cache purge

# Override the default entrypoint of the vllm base image
ENTRYPOINT []
CMD ["/bin/bash"]
