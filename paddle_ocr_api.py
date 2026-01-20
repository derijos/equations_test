from flask import Flask, request, jsonify, Response
from paddleocr import PaddleOCRVL
import os

app = Flask(__name__)

# Initialize Model Once
print("Loading Model...")
pipeline = PaddleOCRVL()
print("Model Loaded.")

def save_and_predict(file):
    """Helper to save uploaded file and run OCR"""
    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    try:
        output = pipeline.predict(temp_path)
        return output, temp_path
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

@app.get("/")
def home():
    return "Hello Friends"

@app.route('/ocr/json', methods=['POST'])
def ocr_json():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        results, temp_path = save_and_predict(file)        
        # Cleanup
        os.remove(temp_path)
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
