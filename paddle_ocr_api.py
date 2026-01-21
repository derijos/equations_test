import os
import numpy as np
import shutil
import uuid
from flask import Flask, request, jsonify
from paddleocr import PaddleOCRVL

app = Flask(__name__)

# --- GLOBAL MODEL ---
print("⏳ Loading PaddleOCR-VL Model...")
try:
    pipeline = PaddleOCRVL()
    print("✅ Model Loaded.")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    exit()

@app.get('/')
def home():
    return "PaddleOCR-VL Equation Extraction API is running."   


@app.post('/json')
def ocr_markdown():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']

    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    json_data = {}

    try:
        pdf_path = os.path.join(temp_dir, file.filename)
        file.save(pdf_path)
        
        initial_output = pipeline.predict(pdf_path)

        for res in initial_output:
            json_data = res.json

    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return jsonify(json_data)


@app.post('/ocr')
def ocr_markdown():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']

    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    ocr_markdown = ""

    try:
        pdf_path = os.path.join(temp_dir, file.filename)
        file.save(pdf_path)
        
        initial_output = pipeline.predict(pdf_path)

        for res in initial_output:
            ocr_markdown = res.markdown

    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return ocr_markdown


if __name__ == '__main__':
    os.makedirs("temp_api", exist_ok=True)
    app.run(host='0.0.0.0', port=5001)
