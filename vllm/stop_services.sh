#!/bin/bash

echo "Stopping all services..."

if [ -f pids.txt ]; then
    kill $(cat pids.txt) 2>/dev/null && echo "✅ All services stopped." || echo "⚠️  Some already stopped."
    rm pids.txt
else
    fuser -k 8118/tcp 2>/dev/null && echo "Stopped vLLM OCR  (8118)" || true
    fuser -k 8119/tcp 2>/dev/null && echo "Stopped vLLM LLM  (8119)" || true
    fuser -k 5001/tcp 2>/dev/null && echo "Stopped PaddleOCR API (5001)" || true
fi
