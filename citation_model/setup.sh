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

CITATION_DIR="/workspace/equations_test/citation_model"
MODEL_DIR="${CITATION_DIR}/scibert_citation_model_v2/run-0/checkpoint-2424"

echo "✅ Repo ready"

# ============================================================
# PART 2: INSTALL DEPENDENCIES
# ============================================================
echo ""
echo "--- Installing dependencies ---"

pip install --ignore-installed \
    torch==2.12.0 \
    transformers==5.9.0 \
    flask==3.1.3 \
    hf_transfer

echo "✅ Dependencies installed"

# ============================================================
# PART 3: DOWNLOAD MODEL FROM HUGGINGFACE
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
# PART 4: START FLASK API
# ============================================================
echo ""
echo "--- Starting Citation API on port 5002 ---"

python3 ${CITATION_DIR}/api.py

