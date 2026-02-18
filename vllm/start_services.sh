#!/bin/bash
set -e

cd /workspace/paddle_setup

echo "========================================"
echo " Starting vLLM Services"
echo " PaddleOCR + gpt-oss-20b + DeepSeek-OCR"
echo "========================================"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# ============================================================
# STEP 1: START PADDLEOCR SERVER (PORT 8118)
# ============================================================

echo ""
echo "--- STEP 1: Starting PaddleOCR Server (port 8118) ---"

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
sleep 10
until curl -s http://localhost:8118/v1/models > /dev/null 2>&1; do
    sleep 5
    echo "  ...waiting"
done
echo "✅ PaddleOCR server ready on port 8118"

# ============================================================
# STEP 2: START GPT-OSS-20B SERVER (PORT 8119)
# ============================================================

echo ""
echo "--- STEP 2: Starting gpt-oss-20b Server (port 8119) ---"

nohup .venv_vllm/bin/vllm serve /workspace/models/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8119 \
    --gpu-memory-utilization 0.50 \
    --max-model-len 32768 \
    --trust-remote-code \
    > logs/vllm_llm.log 2>&1 &

GPT_PID=$!
echo "gpt-oss-20b PID: $GPT_PID"

echo "Waiting for gpt-oss-20b to be ready (this takes 3-5 minutes)..."
sleep 30
until curl -s http://localhost:8119/v1/models > /dev/null 2>&1; do
    sleep 10
    echo "  ...waiting"
done
echo "✅ gpt-oss-20b server ready on port 8119"

# ============================================================
# STEP 3: START DEEPSEEK-OCR SERVER (PORT 8120)
# ============================================================

echo ""
echo "--- STEP 3: Starting DeepSeek-OCR Server (port 8120) ---"

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

echo "Waiting for DeepSeek-OCR to be ready (this takes 2-3 minutes)..."
sleep 20
until curl -s http://localhost:8120/v1/models > /dev/null 2>&1; do
    sleep 10
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
echo "  Verify servers:"
echo "    curl -s http://localhost:8118/v1/models"
echo "    curl -s http://localhost:8119/v1/models"
echo "    curl -s http://localhost:8120/v1/models"
echo ""
echo "  Logs:"
echo "    tail -f logs/vllm_ocr.log"
echo "    tail -f logs/vllm_llm.log"
echo "    tail -f logs/deepseek_ocr.log"
echo ""
echo "  PIDs saved to pids.txt"
echo "  To stop all: kill \$(cat pids.txt)"
echo ""
echo "  Check VRAM: nvidia-smi"
