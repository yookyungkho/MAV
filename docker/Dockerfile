FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# RUN : 터미널 창에서 해당 명령어 실행
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

# install vscode-server
RUN wget -O- https://aka.ms/install-vscode-server/setup.sh | sh

#pip 업그레이드
RUN pip install --upgrade pip
RUN pip install jupyterlab wandb wget gpustats
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 이미지 생성 시 시작 경로 (컨테이너 생성 시 root 경로에서 시작 함)
WORKDIR /workspace/codes/

# ARG : 변수 설정/변수 설정하면 관리자 권한 없어도 사용가능
ARG UNAME
ARG UID
ARG GID
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
USER $UNAME