#!/bin/bash
set -e

# Uses full venv paths instead of source activate
# so this works cleanly with: bash start_services.sh

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs

echo "========================================"
echo " Starting All Services"
echo "========================================"


# ============================================================
# STEP 1: vLLM OCR Server
# Model: PaddleOCR-VL-0.9B
# Port:  8118
# ============================================================

echo ""
echo "--- STEP 1: vLLM OCR Server (port 8118) ---"

nohup .venv_vllm/bin/paddleocr genai_server \
    --model_name PaddleOCR-VL-0.9B \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118 \
    --backend_config vllm_ocr_config.yaml \
    > logs/vllm_ocr.log 2>&1 &

VLLM_OCR_PID=$!
echo "vLLM OCR PID: $VLLM_OCR_PID"

echo "Waiting for vLLM OCR to be ready..."
until curl -s http://localhost:8118/v1/models > /dev/null 2>&1; do
    sleep 3
    echo "  ...waiting"
done
echo "✅ vLLM OCR ready!"


# ============================================================
# STEP 2: vLLM LLM Server
# Model: gpt-oss:20b
# Port:  8119
# ============================================================

echo ""
echo "--- STEP 2: vLLM LLM Server (port 8119) ---"

nohup .venv_vllm/bin/vllm serve gpt-oss:20b \
    --host 0.0.0.0 \
    --port 8119 \
    --config vllm_llm_config.yaml \
    > logs/vllm_llm.log 2>&1 &

VLLM_LLM_PID=$!
echo "vLLM LLM PID: $VLLM_LLM_PID"

echo "Waiting for vLLM LLM to be ready..."
until curl -s http://localhost:8119/v1/models > /dev/null 2>&1; do
    sleep 3
    echo "  ...waiting"
done
echo "✅ vLLM LLM ready!"


# ============================================================
# STEP 3: PaddleOCR Flask API
# File:  paddle_ocr_api.py
# Port:  5001
# ============================================================

echo ""
echo "--- STEP 3: PaddleOCR Flask API (port 5001) ---"

export VLLM_SERVER_URL="http://localhost:8118/v1"

nohup .venv_app/bin/python3 paddle_ocr_api.py \
    > logs/paddle_ocr_api.log 2>&1 &

PADDLE_PID=$!
echo "PaddleOCR API PID: $PADDLE_PID"

echo "Waiting for PaddleOCR API to be ready..."
until curl -s http://localhost:5001 > /dev/null 2>&1; do
    sleep 2
    echo "  ...waiting"
done
echo "✅ PaddleOCR API ready!"


# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "========================================"
echo "✅ All Services Running on RunPod"
echo "========================================"
echo ""
echo "  vLLM OCR  (PaddleOCR-VL-0.9B) → :8118"
echo "  vLLM LLM  (gpt-oss:20b)        → :8119"
echo "  PaddleOCR Flask API             → :5001"
echo ""
echo "  Logs:"
echo "    tail -f logs/vllm_ocr.log"
echo "    tail -f logs/vllm_llm.log"
echo "    tail -f logs/paddle_ocr_api.log"
echo ""
echo "  Expose these ports in RunPod dashboard:"
echo "    5001  → PaddleOCR API  (used by app.py)"
echo "    8119  → vLLM LLM       (used by app.py)"
echo "    8118  → vLLM OCR       (internal only)"
echo ""

# Save PIDs for easy shutdown
echo "$VLLM_OCR_PID $VLLM_LLM_PID $PADDLE_PID" > pids.txt
echo "PIDs saved to pids.txt"
echo "To stop: ./stop_services.sh"
