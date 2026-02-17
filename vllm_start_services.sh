#!/bin/bash

# Start all services in correct order:
# 1. vLLM OCR server  (PaddleOCR-VL-0.9B) → port 8118
# 2. vLLM LLM server  (gpt-oss:20b)        → port 8119
# 3. paddle_ocr_api   (Flask)               → port 5001
# 4. app.py           (Flask)               → port 7860

set -e

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1

# ============================================================
# STEP 1: vLLM OCR Server (PaddleOCR-VL-0.9B)
# ============================================================

echo "=================================================="
echo "STEP 1: Starting vLLM OCR Server (port 8118)"
echo "=================================================="

source .venv_vllm/bin/activate

nohup paddleocr genai_server \
    --model_name PaddleOCR-VL-0.9B \
    --backend vllm \
    --host 0.0.0.0 \
    --port 8118 \
    --backend_config vllm_ocr_config.yaml > logs/vllm_ocr.log 2>&1 &

VLLM_OCR_PID=$!
echo "   vLLM OCR PID: $VLLM_OCR_PID"

echo "   Waiting for vLLM OCR to be ready..."
until curl -s http://localhost:8118/v1/models > /dev/null 2>&1; do
    sleep 3
    echo "   ...waiting"
done
echo "✅ vLLM OCR is ready!"

# ============================================================
# STEP 2: vLLM LLM Server (gpt-oss:20b)
# ============================================================

echo ""
echo "=================================================="
echo "STEP 2: Starting vLLM LLM Server (port 8119)"
echo "=================================================="

nohup vllm serve gpt-oss:20b \
    --host 0.0.0.0 \
    --port 8119 \
    --config vllm_llm_config.yaml > logs/vllm_llm.log 2>&1 &

VLLM_LLM_PID=$!
echo "   vLLM LLM PID: $VLLM_LLM_PID"

echo "   Waiting for vLLM LLM to be ready..."
until curl -s http://localhost:8119/v1/models > /dev/null 2>&1; do
    sleep 3
    echo "   ...waiting"
done
echo "✅ vLLM LLM is ready!"

deactivate

# ============================================================
# STEP 3: paddle_ocr_api.py Flask API
# ============================================================

echo ""
echo "=================================================="
echo "STEP 3: Starting PaddleOCR Flask API (port 5001)"
echo "=================================================="

source .venv_app/bin/activate

export VLLM_SERVER_URL="http://localhost:8118/v1"

nohup python3 paddle_ocr_api.py > logs/paddle_ocr_api.log 2>&1 &
PADDLE_PID=$!
echo "   PaddleOCR API PID: $PADDLE_PID"

echo "   Waiting for PaddleOCR API to be ready..."
until curl -s http://localhost:5001 > /dev/null 2>&1; do
    sleep 2
    echo "   ...waiting"
done
echo "✅ PaddleOCR API is ready!"

# ============================================================
# STEP 4: app.py Main Flask App
# ============================================================

echo ""
echo "=================================================="
echo "STEP 4: Starting Main Flask App (port 7860)"
echo "=================================================="

nohup python3 app.py > logs/app.log 2>&1 &
APP_PID=$!
echo "   App PID: $APP_PID"

sleep 3
echo "✅ Main App started!"

deactivate

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo "=================================================="
echo "✅ ALL SERVICES RUNNING"
echo "=================================================="
echo ""
echo "  vLLM OCR  (PaddleOCR-VL-0.9B) → http://localhost:8118"
echo "  vLLM LLM  (gpt-oss:20b)        → http://localhost:8119"
echo "  PaddleOCR Flask API             → http://localhost:5001"
echo "  Main App                        → http://localhost:7860"
echo ""
echo "  Logs:"
echo "    logs/vllm_ocr.log"
echo "    logs/vllm_llm.log"
echo "    logs/paddle_ocr_api.log"
echo "    logs/app.log"
echo ""
echo "  PIDs: $VLLM_OCR_PID $VLLM_LLM_PID $PADDLE_PID $APP_PID"
echo ""
echo "  To stop everything:"
echo "    kill $VLLM_OCR_PID $VLLM_LLM_PID $PADDLE_PID $APP_PID"
echo ""
echo "  Or save PIDs and stop later:"
echo "    echo '$VLLM_OCR_PID $VLLM_LLM_PID $PADDLE_PID $APP_PID' > pids.txt"
echo "    kill \$(cat pids.txt)"

# Save PIDs to file for easy shutdown
echo "$VLLM_OCR_PID $VLLM_LLM_PID $PADDLE_PID $APP_PID" > pids.txt
