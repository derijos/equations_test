#!/bin/bash
set -e

echo "========================================"
echo " Installation Script - RTX 6000 Ada"
echo " PaddleOCR + gpt-oss-20b + DeepSeek-OCR"
echo " CUDA 12.6"
echo "========================================"

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

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Configure environment variables
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
# PART 2: PADDLEOCR + vLLM SETUP
# ============================================================

echo ""
echo "--- PART 2: PaddleOCR + vLLM Setup ---"

mkdir -p /workspace/paddle_setup && cd /workspace/paddle_setup
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate

echo "Installing vLLM..."
pip install vllm

echo "Installing ninja build tool..."
pip install ninja

echo "Installing Flash Attention..."
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl

echo "Installing PaddleOCR..."
pip install "paddleocr[doc-parser]"

echo "Installing PaddlePaddle GPU..."
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

echo "Restoring torch CUDA dependencies..."
pip install nvidia-nccl-cu12==2.27.5 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1

echo "Verifying installations..."
python3 -c "import torch; print(f'torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
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

echo "Installing hf_transfer..."
pip install hf_transfer

echo "Downloading gpt-oss-20b model (~14GB)..."
python3 -c "
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
# PART 4: DEEPSEEK-OCR SETUP
# ============================================================

echo ""
echo "--- PART 4: DeepSeek-OCR Setup ---"

python3 -m venv .venv_deepseek

echo "Installing vLLM nightly in separate venv..."
.venv_deepseek/bin/pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

echo "Installing ninja in deepseek venv..."
.venv_deepseek/bin/pip install ninja

echo "Installing HuggingFace tools..."
.venv_deepseek/bin/pip install huggingface_hub hf_transfer

mkdir -p /workspace/models/deepseek-ocr

echo "Downloading DeepSeek-OCR model (~7GB)..."
.venv_deepseek/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='deepseek-ai/DeepSeek-OCR',
    local_dir='/workspace/models/deepseek-ocr'
)
print('✅ Model downloaded!')
"

echo "✅ DeepSeek-OCR installed"

# ============================================================
# PART 5: CREATE CONFIG FILES & DIRECTORIES
# ============================================================

echo ""
echo "--- PART 5: Creating config files ---"

cd /workspace/paddle_setup

cat > vllm_ocr_config.yaml << 'EOF'
gpu-memory-utilization: 0.15
max-num-batched-tokens: 16384
no-enable-prefix-caching: true
mm-processor-cache-gb: 0
EOF

cat > vllm_llm_config.yaml << 'EOF'
gpu-memory-utilization: 0.50
max-num-batched-tokens: 32768
max-model-len: 32768
no-enable-prefix-caching: true
EOF

mkdir -p logs

echo "✅ Config files created"

echo ""
echo "========================================"
echo "✅ Installation Complete!"
echo "========================================"
echo ""
echo "Models installed:"
echo "  - PaddleOCR-VL-0.9B"
echo "  - gpt-oss-20b"
echo "  - DeepSeek-OCR"
echo ""
echo "Next step:"
echo "  ./start_services.sh"
