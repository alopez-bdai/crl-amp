FROM nvidia/cuda:11.6.2-base-ubuntu20.04
LABEL name="Jin Cheng"
LABEL email="jin.cheng@inf.ethz.ch"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
RUN export PATH="$HOME/.local/bin:$PATH"
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install -y \
    python3.8 \
    build-essential  \
    python3.8-dev  \
    python3.8-distutils
