#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Full Stack Installation (vLLM, NO Ollama)"
echo "   CUDA 12.6 + PaddlePaddle + PyTorch + vLLM + Flask"

# ============================================================
# PART 1: SYSTEM & CUDA SETUP
# ============================================================

echo "📌 Configuring APT preferences..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

echo "🔑 Installing CUDA keyring..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo "📦 Installing CUDA Toolkit 12.6..."
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

echo "🌍 Configuring CUDA Environment Variables..."
if ! grep -q "cuda-12.6/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
fi
if ! grep -q "cuda-12.6/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
fi

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# ============================================================
# PART 2: VLLM VIRTUAL ENVIRONMENT
# (Separate venv — PaddlePaddle and vLLM conflict if mixed)
# ============================================================

echo ""
echo "🐍 Creating vLLM virtual environment (.venv_vllm)..."
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

echo "📖 Installing PaddleOCR in vLLM venv..."
pip install "paddleocr[doc-parser]"

echo "🚣 Installing PaddlePaddle GPU (CUDA 12.6) in vLLM venv..."
pip install paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "⚡ Installing FlashAttention (must be before vLLM)..."
# Check Python version and pick the right wheel
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
echo "   Detected Python: $PY_VERSION"

if [ "$PY_VERSION" = "310" ]; then
    pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl"
elif [ "$PY_VERSION" = "311" ]; then
    pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-linux_x86_64.whl"
else
    echo "⚠️  FlashAttention prebuilt wheel not found for Python $PY_VERSION"
    echo "   Trying to install from source (may take a while)..."
    pip install flash-attn
fi

echo "🚀 Installing vLLM dependencies via PaddleOCR CLI..."
paddleocr install_genai_server_deps vllm

deactivate


# ============================================================
# PART 3: MAIN APP VIRTUAL ENVIRONMENT
# (For app.py, heuristics, pdf_layout_formula etc.)
# ============================================================

echo ""
echo "🐍 Creating main app virtual environment (.venv_app)..."
python3 -m venv .venv_app
source .venv_app/bin/activate

echo "📦 Installing main app dependencies..."
pip install \
    flask \
    pymupdf \
    pymupdf4llm \
    opencv-python-headless \
    fpdf2 \
    openai \
    requests \
    pydantic \
    "paddleocr[doc-parser]"

echo "🚣 Installing PaddlePaddle GPU (CUDA 12.6) in app venv..."
pip install paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

deactivate


# ============================================================
# PART 4: CREATE VLLM CONFIG FILES
# ============================================================

echo ""
echo "⚙️  Creating vLLM config files..."

# Config for PaddleOCR-VL-0.9B (OCR model — small, fast)
cat > vllm_ocr_config.yaml << EOF
# PaddleOCR-VL-0.9B — Vision-Language OCR model
gpu-memory-utilization: 0.4
max-num-batched-tokens: 16384
no-enable-prefix-caching: true
mm-processor-cache-gb: 0
EOF

# Config for gpt-oss:20b (LLM — larger, needs more VRAM)
cat > vllm_llm_config.yaml << EOF
# gpt-oss:20b — Text LLM for equation analysis
gpu-memory-utilization: 0.5
max-num-batched-tokens: 32768
max-model-len: 32768
no-enable-prefix-caching: true
EOF

echo "   vllm_ocr_config.yaml created (OCR model — 40% VRAM)"
echo "   vllm_llm_config.yaml created (LLM model — 50% VRAM)"
echo "   Total: 90% VRAM usage on RTX 6000 Ada (48GB)"

# ============================================================
# PART 5: RUNTIME ENV VARS
# ============================================================

echo ""
echo "🔧 Configuring runtime environment variables..."
if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bashrc; then
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc
    echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
    echo 'export CUDA_LAUNCH_BLOCKING=1' >> ~/.bashrc
fi

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

echo "✅ Installation complete!"
echo ""
echo "Now run: ./start_services.sh to start all services"
