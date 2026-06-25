# ZyinkLabs CpGPT training container.
#
# Base image: NVIDIA CUDA runtime with cuDNN, matching the CUDA version expected by
# the PyTorch/Lightning versions CpGPT pins. Confirm this still matches CpGPT's
# pyproject.toml torch version before each major upgrade.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ---------------------------------------------------------------------------
# AWS CLI (for downloading CpGPT's pretrained weights/dependencies from S3,
# and for the training job to read/write S3 directly if configured to)
# ---------------------------------------------------------------------------
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip \
    && unzip -q awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

# ---------------------------------------------------------------------------
# Poetry + CpGPT
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip poetry

WORKDIR /workspace

# Clone CpGPT at build time. Pin to a specific commit in production once you've
# validated a version, so a future upstream change can't silently alter behavior
# mid-pipeline. Replace <COMMIT_SHA> below once you've picked one.
RUN git clone https://github.com/lcamillo/CpGPT.git
WORKDIR /workspace/CpGPT
# RUN git checkout <COMMIT_SHA>

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# ---------------------------------------------------------------------------
# ZyinkLabs pipeline scripts + their dependencies
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

COPY scripts/ /workspace/scripts/
COPY configs/experiment/*.yaml /workspace/CpGPT/configs/experiment/
COPY docker/orchestrate_training.py /workspace/orchestrate_training.py
COPY docker/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENV PROJECT_ROOT=/workspace/CpGPT

ENTRYPOINT ["/workspace/entrypoint.sh"]
