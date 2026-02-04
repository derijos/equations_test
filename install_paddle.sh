#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting Full Stack Installation for RTX 6000 Ada..."
echo "   (CUDA 12.6 + PaddlePaddle + PyTorch + Flask + Ollama + OCR API)"

# --- PART 1: SYSTEM & CUDA SETUP ---

# 1. Prefer NVIDIA repo over Ubuntu's default
echo "📌 Configuring APT preferences..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 2. Download and install the CUDA keyring
echo "🔑 Installing CUDA keyring..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

# 3. Update package lists and install CUDA Toolkit 12.6
echo "📦 Installing CUDA Toolkit 12.6..."
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# 4. Configure Environment Variables
echo "🌍 Configuring CUDA Environment Variables..."
# Add to .bashrc if not already present
if ! grep -q "cuda-12.6/bin" ~/.bashrc; then
    echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
fi

if ! grep -q "cuda-12.6/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
fi

# Export immediately for the rest of this script to work
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


# --- PART 2: PYTHON LIBRARIES ---

# 5. Install PaddlePaddle GPU
echo "🚣 Installing PaddlePaddle-GPU..."
python3 -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 6. Reinstall PyTorch (Uninstall old, Install new for CU124)
echo "🔥 Reinstalling PyTorch..."
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 7. Install Flask (Ignoring installed blinker to fix conflict)
echo "🌶️ Installing Flask..."
python3 -m pip install flask --ignore-installed blinker

# 8. Install PaddleOCR and dependencies
echo "📖 Installing PaddleOCR..."
python3 -m pip install "paddleocr[doc-parser]"


# --- PART 3: OLLAMA SETUP ---

# 9. Install Ollama
echo "🦙 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 10. Configure Ollama Environment
echo "⚙️ Configuring Ollama Host..."
if ! grep -q "OLLAMA_HOST" ~/.bashrc; then
    echo 'export OLLAMA_HOST=0.0.0.0' >> ~/.bashrc
fi
export OLLAMA_HOST=0.0.0.0

# 11. Start Ollama in background
echo "▶️ Starting Ollama service..."
nohup ollama serve > ollama.log 2>&1 &

# Wait for Ollama to actually start before pulling models
echo "⏳ Waiting for Ollama to become active..."
until curl -s http://localhost:11434 > /dev/null; do
    sleep 2
    echo "   ...waiting for server"
done
echo "🟢 Ollama is active!"

# 12. Pull Models
echo "📥 Pulling AI Models..."
# Note: Ensure these model names are exact. If a model doesn't exist in the library, the script will stop (due to set -e).
ollama pull gemma3:27b
ollama pull gemma3:12b
ollama pull gpt-oss:20b
ollama pull llama3.1:8b
# ollama pull deepseek-ocr:3b # Uncomment if this model exists in your Ollama library


# --- PART 4: FLASK API SETUP ---

echo "🔧 Configuring Runtime Environment Variables..."
# Add fix for MKL/OpenMP conflict to .bashrc for future sessions
if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bashrc; then
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc
    echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
    echo 'export CUDA_LAUNCH_BLOCKING=1' >> ~/.bashrc
fi

# Export for the current run so nohup picks them up immediately
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

echo "🚀 Starting Flask API (paddle_ocr_api.py)..."
# Check if the file exists before running to avoid error
if [ -f "paddle_ocr_api.py" ]; then
    nohup python3 paddle_ocr_api.py > flask_api.log 2>&1 &
    echo "✅ Flask API started in background! (Logs: flask_api.log)"
else
    echo "⚠️ Warning: 'paddle_ocr_api.py' not found in current directory. Cannot start Flask app."
fi

echo "✅ All Installations & Services Started! Environment ready."
echo "   Please run 'source ~/.bashrc' to apply environment changes manually."
