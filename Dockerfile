FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

RUN mkdir -p /tmp/ultralytics && chmod 777 /tmp/ultralytics

WORKDIR /workspace/retail-video-analytics



RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install \
        torch==2.11.0 \
        --index-url https://download.pytorch.org/whl/cu126 && \
    python -m pip install -r requirements.txt

COPY . .