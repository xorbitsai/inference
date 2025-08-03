FROM vllm/vllm-openai:v0.9.2

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
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH"
# ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Install pip dependencies
ARG LLAMA_CPP_USE_CUDA=true
ARG PIP_INDEX=https://pypi.org/simple
RUN pip install --upgrade -i "$PIP_INDEX" pip setuptools wheel && \
    apt-get -y update && \
    wget https://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz && \
    tar zxvf openfst-1.7.2.tar.gz && cd openfst-1.7.2 && \
    ./configure --enable-shared --enable-static && make -j$(nproc) && make install && ldconfig && \
    CPLUS_INCLUDE_PATH=/usr/local/include LIBRARY_PATH=/usr/local/lib LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    pip install pynini==2.1.6.post1 && \
    apt install -y wget curl procps git libgl1 rsync sqlite libpcre3 libpcre3-dev dmidecode libssl-dev perl make build-essential zlib1g-dev && \
    apt-get -yq clean && \
    # use pre-built whl package for llama-cpp-python, otherwise may core dump when init llama in some envs
    pip install -i "$PIP_INDEX" "diskcache>=5.6.1" "jinja2>=2.11.3" && \
    pip install -i "$PIP_INDEX" "cython>=0.29" && \
    # Determine whether to use the CUDA version (false represents CPU build,true represents CUDA build (GPU supported))
    if [ "$LLAMA_CPP_USE_CUDA" = "true" ]; then \
        echo "🔧 Using CUDA version llama-cpp-python..." && \
        pip install "llama-cpp-python>=0.2.82" -i https://abetlen.github.io/llama-cpp-python/whl/cu124; \
    else \
        echo "⚙️ Using CPU version llama-cpp-python..." && \
        pip install "llama-cpp-python>=0.2.82" -i "$PIP_INDEX";  \
    fi && \
    pip install flash-attn==2.7.4.post1 --no-build-isolation && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements_12.8/requirements-base.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements_12.8/requirements-ml.txt && \
    pip install -i "$PIP_INDEX" --upgrade-strategy only-if-needed -r /opt/inference/xinference/deploy/docker/requirements_12.8/requirements-ml.txt && \
    pip install -i "$PIP_INDEX" transformers>=4.51.3 && \
    pip install -i "$PIP_INDEX" --no-deps sglang==0.4.6.post5 && \
    pip install  https://github.com/sgl-project/whl/releases/download/v0.2.6/sgl_kernel-0.2.6+cu128-cp39-abi3-manylinux2014_x86_64.whl && \
    pip install WeTextProcessing==1.0.4.1 --no-deps && \
    pip uninstall flashinfer -y && \
    
    pip install flashinfer-python && \
    pip install -i "$PIP_INDEX" SQLAlchemy==1.4.54 && \
    cd /opt/inference && \
    python3 setup.py build_web && \
    git restore . && \
    pip install -i "$PIP_INDEX" --no-deps "." && \
    pip uninstall xllamacpp -y && \
    pip install "xllamacpp>=0.1.23" --index-url https://xorbitsai.github.io/xllamacpp/whl/cu124 && \
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
    --no-deps

# Override the default entrypoint of the vllm base image
ENTRYPOINT []
CMD ["/bin/bash"]
