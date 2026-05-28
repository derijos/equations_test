#!/bin/bash
set -e

echo "========================================"
echo " SciBERT Citation Model - Setup & Start"
echo "========================================"

# ============================================================
# PREREQUISITES CHECK
# ============================================================
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN is not set!"
    echo "   Add it to your RunPod Secrets Manager as HF_TOKEN"
    exit 1
fi
echo "✅ HuggingFace token found"

# ============================================================
# PART 1: CLONE REPO
# ============================================================
echo ""
echo "--- Cloning repo ---"

if [ ! -d "/workspace/equations_test" ]; then
    git clone https://github.com/derijos/equations_test.git /workspace/equations_test
else
    echo "Repo already exists, pulling latest..."
    git -C /workspace/equations_test pull
fi

# Use the script's own directory as base — works wherever the script is run from
CITATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${CITATION_DIR}/scibert_citation_model_v2/run-0/checkpoint-2424"
VENV_DIR="${CITATION_DIR}/.venv"

echo "✅ Repo ready"
echo "   Citation dir : ${CITATION_DIR}"
echo "   Model dir    : ${MODEL_DIR}"

# ============================================================
# PART 2: CREATE VIRTUAL ENVIRONMENT
# Isolates from system packages to avoid torch/torchvision/
# transformers version conflicts in RunPod base image.
# ============================================================
echo ""
echo "--- Setting up virtual environment ---"

if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "✅ Virtual environment created at ${VENV_DIR}"
else
    echo "✅ Virtual environment already exists"
fi

source "${VENV_DIR}/bin/activate"

# ============================================================
# PART 3: INSTALL DEPENDENCIES (inside venv)
# ============================================================
echo ""
echo "--- Installing dependencies ---"

pip install --upgrade pip

pip install \
    torch==2.12.0 \
    transformers==5.9.0 \
    tokenizers==0.22.2 \
    safetensors==0.7.0 \
    flask==3.1.3 \
    hf_transfer \
    numpy

echo "✅ Dependencies installed"

# ============================================================
# PART 4: DOWNLOAD MODEL FROM HUGGINGFACE
# ============================================================
echo ""
echo "--- Downloading SciBERT citation model ---"

mkdir -p "${MODEL_DIR}"

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
import os

token = os.environ.get('HF_TOKEN')
snapshot_download(
    repo_id='Dereck-Jos/scibert-citation-model',
    token=token,
    local_dir='${MODEL_DIR}'
)
print('✅ Model downloaded!')
"

echo "✅ Model ready at ${MODEL_DIR}"

# ============================================================
# PART 5: START FLASK API (inside venv)
# ============================================================
echo ""
echo "--- Starting Citation API on port 5002 ---"

python3 "${CITATION_DIR}/api.py"
