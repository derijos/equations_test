#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting installation for RTX 6000 Ada (CUDA 12.6 + PaddlePaddle + PyTorch)..."

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
echo "🌍 Configuring Environment Variables..."
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

# 5. Install PaddlePaddle GPU
# Note: Using version 3.3.0 and the specific index URL as requested. 
# If 3.3.0 is not found (as 3.0.0-beta is common), you may need to remove '==3.3.0' to get the latest.
echo "🚣 Installing PaddlePaddle-GPU..."
python3 -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 6. Reinstall PyTorch (Uninstall old, Install new for CU124)
# Note: Pytorch allows CU124 runtime alongside system CU126 driver usually.
echo "🔥 Reinstalling PyTorch..."
python3 -m pip uninstall -y torch torchvision torchaudio
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 7. Install PaddleOCR and dependencies
echo "📖 Installing PaddleOCR..."
python3 -m pip install "paddleocr[doc-parser]"

echo "✅ Installation Complete! Please restart your terminal or run 'source ~/.bashrc' to apply environment changes."
