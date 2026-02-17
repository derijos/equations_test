#!/bin/bash
set -e

echo "========================================"
echo " RunPod Installation (No Ollama)"
echo " CUDA 12.6 + PaddlePaddle + vLLM"
echo "========================================"

# ============================================================
# PART 1: CUDA 12.6 SETUP
# ============================================================

echo ""
echo "--- PART 1: CUDA 12.6 Setup ---"

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update -q
sudo apt-get -y install cuda-toolkit-12-6

if ! grep -q "cuda-12.6/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
fi
if ! grep -q "cuda-12.6/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
fi

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

echo "✅ CUDA 12.6 configured."


# ============================================================
# PART 2: vLLM VIRTUAL ENVIRONMENT
# (.venv_vllm) — PaddleOCR + vLLM
# Must be separate from app venv due to dependency conflicts
# ============================================================

echo ""
echo "--- PART 2: vLLM Virtual Environment ---"

python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

echo "Installing PaddleOCR..."
pip install -q "paddleocr[doc-parser]"

echo "Installing PaddlePaddle GPU (CUDA 12.6)..."
pip install -q paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "Installing FlashAttention (must be before vLLM)..."
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "Python version: $PY_VERSION"

if [ "$PY_VERSION" = "310" ]; then
    pip install -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
elif [ "$PY_VERSION" = "311" ]; then
    pip install -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-linux_x86_64.whl"
else
    echo "⚠️  No prebuilt wheel for Python $PY_VERSION, building from source..."
    pip install flash-attn
fi

echo "Installing vLLM via PaddleOCR CLI..."
paddleocr install_genai_server_deps vllm

deactivate
echo "✅ vLLM venv ready (.venv_vllm)"


# ============================================================
# PART 3: APP VIRTUAL ENVIRONMENT
# (.venv_app) — Flask API for paddle_ocr_api.py
# ============================================================

echo ""
echo "--- PART 3: App Virtual Environment ---"

python3 -m venv .venv_app
source .venv_app/bin/activate

echo "Installing app dependencies..."
pip install -q \
    flask \
    "paddleocr[doc-parser]" \
    requests

echo "Installing PaddlePaddle GPU (CUDA 12.6)..."
pip install -q paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

deactivate
echo "✅ App venv ready (.venv_app)"


# ============================================================
# PART 4: VLLM CONFIG FILES
# RTX 6000 Ada = 48GB VRAM
# OCR model (0.9B)  → 40% = ~19GB
# LLM model (20B)   → 50% = ~24GB
# Total             → 90% = ~43GB
# ============================================================

echo ""
echo "--- PART 4: vLLM Config Files ---"

cat > vllm_ocr_config.yaml << 'EOF'
gpu-memory-utilization: 0.4
max-num-batched-tokens: 16384
no-enable-prefix-caching: true
mm-processor-cache-gb: 0
EOF

cat > vllm_llm_config.yaml << 'EOF'
gpu-memory-utilization: 0.5
max-num-batched-tokens: 32768
max-model-len: 32768
no-enable-prefix-caching: true
EOF

echo "✅ vllm_ocr_config.yaml created"
echo "✅ vllm_llm_config.yaml created"


# ============================================================
# PART 5: RUNTIME ENV VARS
# ============================================================

echo ""
echo "--- PART 5: Runtime Environment Variables ---"

if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bashrc; then
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc
    echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
    echo 'export CUDA_LAUNCH_BLOCKING=1' >> ~/.bashrc
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# ============================================================
# PART 6: CREATE LOGS DIRECTORY
# ============================================================

mkdir -p logs

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  source ~/.bashrc"
echo "  chmod +x start_services.sh stop_services.sh"
echo "  ./start_services.sh"
