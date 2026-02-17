#!/bin/bash
# Stop all services

if [ -f pids.txt ]; then
    echo "Stopping all services..."
    kill $(cat pids.txt) 2>/dev/null && echo "✅ All services stopped." || echo "⚠️  Some processes already stopped."
    rm pids.txt
else
    echo "No pids.txt found. Killing by port..."
    fuser -k 8118/tcp 2>/dev/null && echo "Stopped vLLM OCR (8118)"
    fuser -k 8119/tcp 2>/dev/null && echo "Stopped vLLM LLM (8119)"
    fuser -k 5001/tcp 2>/dev/null && echo "Stopped PaddleOCR API (5001)"
    fuser -k 7860/tcp 2>/dev/null && echo "Stopped Main App (7860)"
fi
