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

@app.route('/ocr/json', methods=['POST'])
def ocr_json():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        output, temp_path = save_and_predict(file)
        
        # Combine results from all detected regions
        results = [res.json for res in output]
        
        # Cleanup
        os.remove(temp_path)
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ocr/markdown', methods=['POST'])
def ocr_markdown():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        output, temp_path = save_and_predict(file)
        
        # Generate Markdown
        # PaddleOCRVL usually saves files, but we can extract the text directly 
        # or rely on its internal saving mechanism and read it back.
        # Ideally, we construct the markdown from the rec_text if available.
        
        full_markdown = ""
        for res in output:
            # Try to get text directly from json result
            if 'rec_text' in res.json:
                full_markdown += res.json['rec_text'] + "\n\n"
            else:
                # Fallback: Let paddle save it and we read it
                res.save_to_markdown(save_path=".")
                md_filename = os.path.basename(temp_path).rsplit('.', 1)[0] + ".md"
                if os.path.exists(md_filename):
                    with open(md_filename, "r") as f:
                        full_markdown += f.read() + "\n\n"
                    os.remove(md_filename)

        # Cleanup
        os.remove(temp_path)
        
        # Return as plain text response
        return Response(full_markdown, mimetype='text/markdown')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
