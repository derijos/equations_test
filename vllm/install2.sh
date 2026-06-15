#!/bin/bash
set -e

echo "========================================"
echo " Installation Script - A100 SXM / RTX 6000 Ada"
echo " PaddleOCR 3.7.0 + gpt-oss-20b"
echo " CUDA 12.6 | vLLM 0.6.6 (.venv_gpt) | vLLM 0.10.2 (.venv_vllm)"
echo "========================================"

# ============================================================
# VERSION PINS
# ============================================================
VLLM_GPT_VERSION="0.6.6"        # for gpt-oss-20b
PADDLE_VERSION="3.2.1"
PADDLE_INDEX="https://www.paddlepaddle.org.cn/packages/stable/cu126/"

# CUDA libs for .venv_gpt (vLLM 0.6.6 + torch 2.4.x)
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
# PART 1: CUDA TOOLKIT 12.6
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
# PART 2: PaddleOCR venv (.venv_vllm)
# Fully isolated: PaddleOCR 3.7.0 + vLLM 0.10.2 + torch 2.8.x
# NO version fighting, let paddleocr install_genai_server_deps
# pull whatever vLLM it needs naturally
# Model downloaded from HuggingFace (avoids Baidu CDN SSL issues)
# ============================================================
echo ""
echo "--- PART 2: PaddleOCR Setup (.venv_vllm) ---"

mkdir -p /workspace/paddle_setup && cd /workspace/paddle_setup
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

echo "Installing ninja..."
pip install ninja

echo "Installing PaddleOCR 3.7.0..."
pip install "paddleocr[doc-parser]==3.7.0"

echo "Installing PaddlePaddle GPU ${PADDLE_VERSION}..."
pip install paddlepaddle-gpu==${PADDLE_VERSION} -i ${PADDLE_INDEX}

echo "Installing PaddleOCR genai server deps (will pull vLLM 0.10.2)..."
.venv_vllm/bin/paddleocr install_genai_server_deps vllm

echo "Pinning prometheus-client to prevent future breakage..."
pip install --force-reinstall prometheus-client==0.20.0

echo "Downloading PaddleOCR-VL model from HuggingFace (avoids Baidu SSL issues)..."
mkdir -p /root/.paddlex/official_models/PaddleOCR-VL
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='PaddlePaddle/PaddleOCR-VL',
    token=os.environ.get('HF_TOKEN'),
    local_dir='/root/.paddlex/official_models/PaddleOCR-VL'
)
print('✅ PaddleOCR-VL model downloaded!')
"

echo "Verifying PaddleOCR venv..."
python3 -c "import torch; print(f'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
python3 -c "import paddle; paddle.utils.run_check()"

deactivate
echo "✅ PaddleOCR installed in .venv_vllm"

# # ============================================================
# # PART 3: gpt-oss-20b venv (.venv_gpt)
# # Fully isolated: vLLM 0.6.6 + torch 2.4.x + pinned CUDA libs
# # No PaddleOCR, no conflicts
# # ============================================================
# echo ""
# echo "--- PART 3: gpt-oss-20b Setup (.venv_gpt, vLLM==${VLLM_GPT_VERSION}) ---"

# python3 -m venv .venv_gpt
# source .venv_gpt/bin/activate

# echo "Installing vLLM==${VLLM_GPT_VERSION}..."
# pip install vllm==${VLLM_GPT_VERSION}

# echo "Installing ninja..."
# pip install ninja

# echo "Pinning CUDA libs for vLLM 0.6.6 + torch 2.4.x..."
# pip install --force-reinstall --no-deps \
#     nvidia-nccl-cu12==${NCCL_VER} \
#     nvidia-nvjitlink-cu12==${NVJITLINK_VER} \
#     nvidia-nvtx-cu12==${NVTX_VER} \
#     nvidia-curand-cu12==${CURAND_VER} \
#     nvidia-cusolver-cu12==${CUSOLVER_VER} \
#     nvidia-cusparse-cu12==${CUSPARSE_VER} \
#     nvidia-cusparselt-cu12==${CUSPARSELT_VER}

# echo "Verifying .venv_gpt..."
# python3 -c "import torch; print(f'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# deactivate
# echo "✅ gpt-oss-20b venv ready in .venv_gpt"

# # ============================================================
# # PART 4: GPT-OSS-20B MODEL DOWNLOAD
# # ============================================================
# echo ""
# echo "--- PART 4: Downloading gpt-oss-20b ---"

# mkdir -p /workspace/models/gpt-oss-20b
# source .venv_gpt/bin/activate

# pip install hf_transfer

# echo "Downloading gpt-oss-20b model (~14GB)..."
# HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
# from huggingface_hub import snapshot_download
# import os
# snapshot_download(
#     repo_id='openai/gpt-oss-20b',
#     token=os.environ.get('HF_TOKEN'),
#     local_dir='/workspace/models/gpt-oss-20b'
# )
# print('✅ gpt-oss-20b downloaded!')
# "

# deactivate
# echo "✅ gpt-oss-20b downloaded"

# ============================================================
# PART 5: CONFIG FILES
# ============================================================
echo ""
echo "--- PART 5: Creating config files ---"

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
echo "Venvs:"
echo "  .venv_vllm → PaddleOCR 3.7.0  (vLLM 0.10.2, torch 2.8.x)"
echo "  .venv_gpt  → gpt-oss-20b      (vLLM ${VLLM_GPT_VERSION}, torch 2.4.x)"
echo ""
echo "Models:"
echo "  PaddleOCR-VL  → /root/.paddlex/official_models/PaddleOCR-VL"
echo "  gpt-oss-20b   → /workspace/models/gpt-oss-20b"
echo ""
echo "Next step: ./start_services.sh"
