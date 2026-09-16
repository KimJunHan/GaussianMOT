FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS base
ENV FORCE_CUDA="1"

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    libgl1-mesa-glx \
    mesa-utils \
    libglapi-mesa \
    libqt5gui5 \
    fonts-liberation \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    jupyter-notebook \
    figlet \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_ENVS_PATH="/opt/conda/envs"

# ToS 동의 + conda-forge 채널 설정
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /opt/conda/bin/conda config --add channels conda-forge \
    && /opt/conda/bin/conda config --set channel_priority strict

# conda init (root)
RUN /opt/conda/bin/conda init bash \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# Jupyter config (no auth)
RUN jupyter notebook --generate-config \
    && echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# Terminal prompt & env
RUN echo "PS1='\[\e[1;31m\]\u\[\e[1;37m\]@\[\e[1;33m\]\h\[\e[1;37m\]:\[\e[0;37m\]\w\[\e[1;37m\] → \[\e[0m\]'" >> /root/.bashrc \
    && echo 'export HF_HOME="/workspace/.cache/hugging-face"' >> /root/.bashrc \
    && echo "source /opt/conda/etc/profile.d/conda.sh && conda activate gaussiancar 2>/dev/null || true" >> /root/.bashrc

WORKDIR /workspace

COPY . /workspace
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/bin/bash"]
