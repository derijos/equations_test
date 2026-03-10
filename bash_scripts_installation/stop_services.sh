#!/bin/bash

cd /workspace/paddle_setup

echo "Stopping all vLLM services..."

if [ -f pids.txt ]; then
    kill $(cat pids.txt) 2>/dev/null && echo "✅ All services stopped." || echo "⚠️  Some already stopped."
    rm pids.txt
else
    echo "No pids.txt found, killing by port..."
    pkill -f "paddleocr genai_server" 2>/dev/null && echo "Stopped PaddleOCR (8118)" || true
    pkill -f "vllm serve /workspace/models/gpt-oss-20b" 2>/dev/null && echo "Stopped gpt-oss (8119)" || true  
    pkill -f "vllm serve /workspace/models/deepseek-ocr" 2>/dev/null && echo "Stopped DeepSeek-OCR (8120)" || true
fi

echo ""
echo "Check status:"
echo "  nvidia-smi"
echo "  ps aux | grep vllm"
