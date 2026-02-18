#!/bin/bash
set -e

echo "========================================"
echo " Complete vLLM Setup - RTX 6000 Ada"
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
# PART 3: CREATE CONFIG FILES
# ============================================================

echo ""
echo "--- PART 3: Creating config files ---"

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

# ============================================================
# PART 4: START PADDLEOCR SERVER
# ============================================================

echo ""
echo "--- PART 4: Starting PaddleOCR Server (port 8118) ---"

nohup .venv_vllm/bin/paddleocr genai_server \
    --model_name PaddleOCR-VL-0.9B \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118 \
    --backend_config vllm_ocr_config.yaml \
    > logs/vllm_ocr.log 2>&1 &

PADDLE_PID=$!
echo "PaddleOCR PID: $PADDLE_PID"

echo "Waiting for PaddleOCR to be ready..."
until curl -s http://localhost:8118/v1/models > /dev/null 2>&1; do
    sleep 5
    echo "  ...waiting"
done
echo "✅ PaddleOCR server ready on port 8118"

# ============================================================
# PART 5: GPT-OSS-20B SETUP
# ============================================================

echo ""
echo "--- PART 5: gpt-oss-20b Setup ---"

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

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

echo "Starting gpt-oss-20b server (port 8119)..."
nohup .venv_vllm/bin/vllm serve /workspace/models/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8119 \
    --gpu-memory-utilization 0.50 \
    --max-model-len 32768 \
    --trust-remote-code \
    > logs/vllm_llm.log 2>&1 &

GPT_PID=$!
echo "gpt-oss-20b PID: $GPT_PID"

echo "Waiting for gpt-oss-20b to be ready..."
until curl -s http://localhost:8119/v1/models > /dev/null 2>&1; do
    sleep 5
    echo "  ...waiting"
done
echo "✅ gpt-oss-20b server ready on port 8119"

# ============================================================
# PART 6: DEEPSEEK-OCR SETUP
# ============================================================

echo ""
echo "--- PART 6: DeepSeek-OCR Setup ---"

python3 -m venv .venv_deepseek

echo "Installing vLLM nightly in separate venv..."
.venv_deepseek/bin/pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

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

echo "Starting DeepSeek-OCR server (port 8120)..."
nohup /workspace/paddle_setup/.venv_deepseek/bin/vllm serve /workspace/models/deepseek-ocr \
    --host 0.0.0.0 \
    --port 8120 \
    --gpu-memory-utilization 0.25 \
    --max-model-len 2048 \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
    > logs/deepseek_ocr.log 2>&1 &

DEEPSEEK_PID=$!
echo "DeepSeek-OCR PID: $DEEPSEEK_PID"

echo "Waiting for DeepSeek-OCR to be ready..."
until curl -s http://localhost:8120/v1/models > /dev/null 2>&1; do
    sleep 5
    echo "  ...waiting"
done
echo "✅ DeepSeek-OCR server ready on port 8120"

# ============================================================
# SAVE PIDS & SUMMARY
# ============================================================

echo "$PADDLE_PID $GPT_PID $DEEPSEEK_PID" > pids.txt

echo ""
echo "========================================"
echo "✅ All Services Running!"
echo "========================================"
echo ""
echo "  PaddleOCR-VL-0.9B  → port 8118"
echo "  gpt-oss-20b        → port 8119"
echo "  DeepSeek-OCR       → port 8120"
echo ""
echo "  Logs:"
echo "    tail -f logs/vllm_ocr.log"
echo "    tail -f logs/vllm_llm.log"
echo "    tail -f logs/deepseek_ocr.log"
echo ""
echo "  PIDs saved to pids.txt"
echo "  To stop: kill \$(cat pids.txt)"
echo ""
echo "  VRAM Allocation (RTX 6000 Ada 49GB):"
echo "    PaddleOCR-VL  ~7GB   (0.15)"
echo "    gpt-oss-20b   ~24GB  (0.50)"
echo "    DeepSeek-OCR  ~12GB  (0.25)"
echo "    Total         ~43GB"
echo "    Buffer        ~6GB"
