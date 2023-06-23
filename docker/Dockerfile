FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    ssh \
    build-essential \
    locales \
    ca-certificates \
    curl \
    unzip \
    vim \
    wget \
    tmux \
    screen \
    pciutils

RUN wget -O- https://aka.ms/install-vscode-server/setup.sh | sh

RUN pip install --upgrade pip
RUN pip install jupyterlab wandb wget gpustats
COPY requirements.txt ./
RUN pip install -r requirements.txt

WORKDIR /workspace/codes/

ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME