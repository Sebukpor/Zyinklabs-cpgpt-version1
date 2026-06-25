#!/usr/bin/env bash
# EC2 user-data bootstrap script for ZyinkLabs CpGPT training.
#
# Target AMI: AWS Deep Learning AMI (Ubuntu 22.04) -- already has NVIDIA drivers
# and CUDA installed, which is why this script does NOT install GPU drivers itself.
# If you instead use a plain Ubuntu AMI, you must install NVIDIA drivers manually
# before this script's Docker GPU steps will work -- see the "Plain Ubuntu AMI"
# note at the bottom of aws/README.md.
#
# This runs automatically on first boot when passed as EC2 user-data.
set -euo pipefail

LOG_FILE="/var/log/zyinklabs-bootstrap.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "== ZyinkLabs EC2 bootstrap starting at $(date -u) =="

# ---------------------------------------------------------------------------
# 1. Docker Engine + Compose plugin
# ---------------------------------------------------------------------------
if ! command -v docker &> /dev/null; then
    echo "--> Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu || true
    rm -f get-docker.sh
else
    echo "--> Docker already installed."
fi

# Docker Compose v2 ships as a plugin with get-docker.sh on recent versions;
# verify and install explicitly if missing.
if ! docker compose version &> /dev/null; then
    echo "--> Installing docker-compose-plugin explicitly..."
    apt-get update
    apt-get install -y docker-compose-plugin
fi

# ---------------------------------------------------------------------------
# 2. NVIDIA Container Toolkit (lets Docker containers see the GPU)
#    Skip the driver install itself -- the Deep Learning AMI already has it.
#    Verify with `nvidia-smi` below; if that fails, your AMI choice is wrong.
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This script assumes the AWS Deep Learning"
    echo "AMI (which ships with NVIDIA drivers preinstalled). If you used a plain"
    echo "Ubuntu AMI instead, install NVIDIA drivers manually before proceeding --"
    echo "see aws/README.md 'Plain Ubuntu AMI' section."
else
    echo "--> GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

if ! command -v nvidia-ctk &> /dev/null; then
    echo "--> Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
else
    echo "--> NVIDIA Container Toolkit already installed."
fi

# Sanity check: confirm a container can actually see the GPU.
echo "--> Verifying GPU is visible inside a container..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi \
    || echo "WARNING: GPU not visible inside containers. Check NVIDIA Container Toolkit setup."

# ---------------------------------------------------------------------------
# 3. AWS CLI (usually already present on the Deep Learning AMI, but confirm)
# ---------------------------------------------------------------------------
if ! command -v aws &> /dev/null; then
    echo "--> Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
    unzip -q awscliv2.zip
    ./aws/install
    rm -rf awscliv2.zip aws/
fi

# ---------------------------------------------------------------------------
# 4. Clone the ZyinkLabs pipeline repo
#    Replace with your actual private repo URL. If private, ensure the
#    instance's IAM role / deploy key has clone access (e.g. via CodeCommit,
#    a deploy key, or a short-lived token -- do not bake long-lived credentials
#    into this script or the AMI).
# ---------------------------------------------------------------------------
REPO_URL="${ZYINKLABS_REPO_URL:-https://github.com/zyinklabs/cpgpt-pipeline.git}"
TARGET_DIR="/home/ubuntu/cpgpt-pipeline"

if [ ! -d "$TARGET_DIR" ]; then
    echo "--> Cloning $REPO_URL..."
    sudo -u ubuntu git clone "$REPO_URL" "$TARGET_DIR"
else
    echo "--> Repo already present at $TARGET_DIR, pulling latest..."
    cd "$TARGET_DIR" && sudo -u ubuntu git pull
fi

echo "== Bootstrap complete at $(date -u) =="
echo "Next: SSH in, populate docker/.env (see docker/.env.example), then run:"
echo "  cd $TARGET_DIR/docker && docker compose run --rm download_dependencies"
echo "  cd $TARGET_DIR/docker && docker compose run --rm prepare_data"
echo "  cd $TARGET_DIR/docker && docker compose run --rm train"
