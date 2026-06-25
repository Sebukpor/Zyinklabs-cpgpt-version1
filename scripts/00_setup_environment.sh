#!/usr/bin/env bash
# ZyinkLabs CpGPT pipeline — one-time environment setup.
#
# Run this once per machine (dev laptop for dry runs, or the GPU box used for real
# training). Safe to re-run; steps are idempotent where possible.
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIPELINE_DIR"

echo "== ZyinkLabs CpGPT pipeline setup =="
echo "Working directory: $PIPELINE_DIR"

# ---------------------------------------------------------------------------
# 1. Clone CpGPT if not already present
# ---------------------------------------------------------------------------
if [ ! -d "CpGPT" ]; then
    echo "--> Cloning CpGPT..."
    git clone https://github.com/lcamillo/CpGPT.git
else
    echo "--> CpGPT already cloned, skipping."
fi

# ---------------------------------------------------------------------------
# 2. Python environment for CpGPT (Poetry, as recommended upstream)
# ---------------------------------------------------------------------------
echo "--> Checking for Poetry..."
if ! command -v poetry &> /dev/null; then
    pip install --break-system-packages poetry
fi

echo "--> Installing CpGPT dependencies via Poetry..."
(cd CpGPT && poetry install)

# ---------------------------------------------------------------------------
# 3. ZyinkLabs-specific Python dependencies
#    - huggingface_hub: download the ComputAgeBench dataset and push our finetuned
#      checkpoint
#    - computage: official ComputAgeBench library, used both as the source of our
#      Version 1 training/benchmark data and for SOTA comparison against Horvath,
#      Hannum, PhenoAge, etc. (see scripts/04_benchmark_vs_sota.py)
#    methylprep (raw IDAT -> beta value conversion) is intentionally NOT installed
#    here -- Version 1 uses only public, already-processed methylation data
#    (ComputAgeBench), not raw IDATs from a wet lab.
# ---------------------------------------------------------------------------
echo "--> Installing ZyinkLabs pipeline dependencies..."
pip install --break-system-packages \
    computage \
    huggingface_hub \
    pandas \
    pyarrow \
    scikit-learn \
    numpy

# ---------------------------------------------------------------------------
# 4. AWS CLI — required to download CpGPT pretrained weights & dependencies
# ---------------------------------------------------------------------------
if ! command -v aws &> /dev/null; then
    echo "--> Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install
    rm -rf awscliv2.zip aws/
else
    echo "--> AWS CLI already installed, skipping."
fi

cat <<'EOF'

== Setup complete ==

Next steps:
  1. Run `aws configure` if you have not already set up AWS credentials.
     CpGPT's pretrained weights are hosted on a public S3 bucket; you need an AWS
     account (free tier is sufficient) to download them. See the CpGPT README for
     the full walkthrough.

  2. Download the pretrained CpGPT dependencies (DNA LLM embeddings) and the large
     model weights (Version 1 finetunes the large backbone for the strongest
     production-quality result -- see docs/DATASET.md and aws/README.md for the
     full reasoning):
       cd CpGPT
       poetry run python -c "
from cpgpt.infer.cpgpt_inferencer import CpGPTInferencer
inferencer = CpGPTInferencer(dependencies_dir='dependencies', data_dir='data')
inferencer.download_dependencies(species='human')
inferencer.download_model('large')           # ~100M param model -- Version 1 default
inferencer.download_model('clock_proxies')   # for biological age comparison/calibration
"

  3. Download the ComputAgeBench dataset (Version 1's training + benchmark data):
       cd ..
       python scripts/01_download_computage_bench.py --out_dir data/processed/computage

  4. Confirm GPU availability before attempting any real finetuning run:
       nvidia-smi
     If this fails, you are on a CPU-only machine — fine for dry runs and pipeline
     validation, but real finetuning needs a GPU box (g6e.xlarge, see aws/README.md).

EOF
