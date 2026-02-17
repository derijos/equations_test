import os
import shutil
import uuid
from flask import Flask, request, jsonify
from paddleocr import PaddleOCRVL

app = Flask(__name__)

# vLLM OCR server URL — set via environment variable or default
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8118/v1")

print("⏳ Loading PaddleOCR-VL pipeline...")
print(f"   Connecting to vLLM OCR server at: {VLLM_SERVER_URL}")
try:
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=VLLM_SERVER_URL
    )
    print("✅ PaddleOCR pipeline ready.")
except Exception as e:
    print(f"❌ Pipeline load failed: {e}")
    exit()


@app.get('/')
def home():
    return "PaddleOCR-VL (vLLM backend) API is running."


@app.post('/paddleocr_json')
def paddleocr_json():
    """
    Accepts a page image (PNG), runs PaddleOCR layout + formula detection.
    Returns JSON with bounding boxes, formula regions, coordinates.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api", req_id)
    os.makedirs(temp_dir, exist_ok=True)

    json_data = {}
    try:
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)

        initial_output = pipeline.predict(img_path)
        for res in initial_output:
            json_data = res.json
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return jsonify(json_data)


@app.post('/paddleocr_ocr')
def paddleocr_ocr():
    """
    Accepts a page image (PNG), returns full page markdown.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api", req_id)
    os.makedirs(temp_dir, exist_ok=True)

    markdown = ""
    try:
        img_path = os.path.join(temp_dir, file.filename)
        file.save(img_path)

        initial_output = pipeline.predict(img_path)
        for res in initial_output:
            markdown = res.markdown
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return markdown


if __name__ == '__main__':
    os.makedirs("temp_api", exist_ok=True)
    app.run(host='0.0.0.0', port=5001)
