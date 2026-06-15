#!/bin/bash
set -e

echo "========================================"
echo " Installation Script - A100 SXM / RTX 6000 Ada"
echo " PaddleOCR + gpt-oss-20b"
echo " CUDA 12.6 | vLLM 0.10.2 (pulled by PaddleOCR)"
echo "========================================"

# ============================================================
# VERSION PINS — edit here only if you need to upgrade
# ============================================================
# NOTE: vLLM is intentionally NOT pinned to 0.6.6 anymore.
# PaddleOCR 3.x pulls in vLLM 0.10.2 as a hard dependency and
# overrides any earlier pin. Pinning 0.6.6 here is a no-op.
# We accept 0.10.2 and fix the prometheus compatibility instead.
PADDLE_VERSION="3.2.1"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# FIX: prometheus-fastapi-instrumentator 8.0.0 (released ~June 2026)
# introduced a breaking change in how it iterates FastAPI routes.
# _IncludedRouter objects no longer have a .path attribute in 8.x,
# which causes every vLLM API request to return 500.
# Pin to 7.x which works correctly with vLLM 0.10.2 + FastAPI.
PROMETHEUS_INSTRUMENTATOR_VERSION="7.0.0"

# CUDA deps — pinned to match PaddlePaddle 3.2.1
NCCL_VER="2.25.1"
NVJITLINK_VER="12.6.85"
NVTX_VER="12.6.77"
CURAND_VER="10.3.7.77"
CUSOLVER_VER="11.7.1.2"
CUSPARSE_VER="12.5.4.2"
CUSPARSELT_VER="0.6.3"

# ============================================================
# PREREQUISITES CHECK
# ============================================================
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN is not set!"
    echo "   Run: export HF_TOKEN=hf_your_token_here"
    exit 1
fi
echo "✅ HuggingFace token found"

# ============================================================
# PART 1: CUDA TOOLKIT 12.6 INSTALLATION
# ============================================================
echo ""
echo "--- PART 1: Installing CUDA Toolkit 12.6 ---"

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update -qq
sudo apt-get -y install cuda-toolkit-12-6

if ! grep -q "cuda-12.6/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
fi
if ! grep -q "cuda-12.6/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
fi

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo "✅ CUDA 12.6 installed"

# ============================================================
# PART 2: PaddleOCR + vLLM Setup
# ============================================================
echo ""
echo "--- PART 2: PaddleOCR + vLLM Setup ---"

mkdir -p /workspace/paddle_setup && cd /workspace/paddle_setup
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

echo "Installing ninja..."
pip install ninja

echo "Installing Flash Attention..."
pip install https://github.com/derijos/vllm_wheels/releases/download/v1.0.0/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl

echo "Installing PaddleOCR..."
pip install "paddleocr[doc-parser]"
# PaddleOCR will pull in vLLM 0.10.2 automatically as a dependency.
# Do NOT install vLLM separately before this — it will just get overridden.

echo "Installing PaddlePaddle GPU ${PADDLE_VERSION}..."
pip install paddlepaddle-gpu==${PADDLE_VERSION} -i ${PADDLE_INDEX}

# ✅ KEY FIX: prometheus-fastapi-instrumentator 8.0.0 broke vLLM's API
# server with an AttributeError on _IncludedRouter.path. Pin to 7.x.
# This is the reason gpt-oss-20b returned 500 on every /v1/models call
# after June 12 2026 when 8.0.0 was released.
echo "Pinning prometheus-fastapi-instrumentator to ${PROMETHEUS_INSTRUMENTATOR_VERSION}..."
pip install --force-reinstall "prometheus-fastapi-instrumentator==${PROMETHEUS_INSTRUMENTATOR_VERSION}"

# ✅ Pin CUDA libs to match PaddlePaddle 3.2.1
echo "Pinning CUDA dependency versions..."
pip install --force-reinstall --no-deps \
    nvidia-nccl-cu12==${NCCL_VER} \
    nvidia-nvjitlink-cu12==${NVJITLINK_VER} \
    nvidia-nvtx-cu12==${NVTX_VER} \
    nvidia-curand-cu12==${CURAND_VER} \
    nvidia-cusolver-cu12==${CUSOLVER_VER} \
    nvidia-cusparse-cu12==${CUSPARSE_VER} \
    nvidia-cusparselt-cu12==${CUSPARSELT_VER}

echo "Verifying installations..."
python3 -c "import torch; print(f'torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"
python3 -c "import paddle; paddle.utils.run_check()"

echo "Installing PaddleOCR genai server deps..."
.venv_vllm/bin/paddleocr install_genai_server_deps vllm

echo "✅ PaddleOCR + vLLM installed"

# ============================================================
# PART 3: GPT-OSS-20B MODEL DOWNLOAD
# ============================================================
echo ""
echo "--- PART 3: Downloading gpt-oss-20b ---"

mkdir -p /workspace/models/gpt-oss-20b

pip install flask hf_transfer

echo "Downloading gpt-oss-20b model (~14GB)..."
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
import os

token = os.environ.get('HF_TOKEN')
snapshot_download(
    repo_id='openai/gpt-oss-20b',
    token=token,
    local_dir='/workspace/models/gpt-oss-20b'
)
print('✅ Model downloaded!')
"

echo "✅ gpt-oss-20b downloaded"

deactivate

# ============================================================
# PART 4: CREATE CONFIG FILES & DIRECTORIES
# ============================================================
echo ""
echo "--- PART 4: Creating config files ---"

cd /workspace/paddle_setup

cat > vllm_ocr_config.yaml << 'EOF'
gpu-memory-utilization: 0.30
max-num-batched-tokens: 16384
no-enable-prefix-caching: true
mm-processor-cache-gb: 0
EOF

mkdir -p logs

echo "✅ Config files created"

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "Pinned versions:"
echo "  - vLLM:                           0.10.2 (set by PaddleOCR)"
echo "  - PaddlePaddle GPU:               ${PADDLE_VERSION}"
echo "  - prometheus-fastapi-instrumentator: ${PROMETHEUS_INSTRUMENTATOR_VERSION}"
echo "  - NCCL:                           ${NCCL_VER}"
echo ""
echo "Models installed:"
echo "  - PaddleOCR-VL-0.9B"
echo "  - gpt-oss-20b"
echo ""
echo "Next step:"
echo "  ./start_services.sh"
