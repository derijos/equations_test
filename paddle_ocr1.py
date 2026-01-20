import os
import time
import cv2
import numpy as np
import json
import re
import shutil
import uuid
import fitz  # PyMuPDF
from flask import Flask, request, jsonify
from paddleocr import PaddleOCRVL

# --- 1. CONFIGURATION & SETUP ---
app = Flask(__name__)

# Initialize Model Once (Global)
print("⏳ Loading PaddleOCR-VL Model...")
try:
    pipeline = PaddleOCRVL()
    print("✅ Model Loaded.")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    exit()

# --- 2. HELPER FUNCTIONS (From your Actual Code) ---
def get_vertical_overlap(box_a, box_b):
    """Calculates vertical overlap height."""
    y_min_a, y_max_a = box_a[1], box_a[3]
    y_min_b, y_max_b = box_b[1], box_b[3]
    inter_min = max(y_min_a, y_min_b)
    inter_max = min(y_max_a, y_max_b)
    return max(0, inter_max - inter_min)

def apply_smart_width_logic(json_data):
    """
    Groups formulas by line and extends width to include equation numbers.
    """
    # Handle structure variations
    if "res" in json_data:
        all_boxes = json_data["res"].get("boxes", [])
    elif "layout_det_res" in json_data:
        all_boxes = json_data["layout_det_res"].get("boxes", [])
    else:
        all_boxes = json_data.get("boxes", []) if isinstance(json_data, dict) else []

    if not all_boxes:
        return []

    # Extract only formulas
    display_formulas = [b for b in all_boxes if b.get('label') == 'display_formula']
    display_formulas.sort(key=lambda x: x['coordinate'][1]) # Sort by Y

    # Group by Line
    lines = []
    if display_formulas:
        current_line = [display_formulas[0]]
        for i in range(1, len(display_formulas)):
            curr = display_formulas[i]
            prev = current_line[-1]
            if get_vertical_overlap(prev['coordinate'], curr['coordinate']) > 0:
                current_line.append(curr)
            else:
                lines.append(current_line)
                current_line = [curr]
        lines.append(current_line)

    final_formula_boxes = []
    
    # Process each line
    for line in lines:
        formula_coords = [item['coordinate'] for item in line]
        
        # Vertical bounds
        min_y = min(c[1] for c in formula_coords)
        max_y = max(c[3] for c in formula_coords)
        
        # Horizontal bounds
        min_x = min(c[0] for c in formula_coords) 
        max_x = max(c[2] for c in formula_coords) 
        
        # Scan ALL boxes to find content to the right (like equation numbers)
        for box in all_boxes:
            b_coord = box['coordinate']
            b_ymin, b_ymax = b_coord[1], b_coord[3]
            overlap = max(0, min(max_y, b_ymax) - max(min_y, b_ymin))
            
            if overlap > 0:
                if b_coord[2] > max_x:
                    max_x = b_coord[2]

        best_score = max(item['score'] for item in line)
        
        new_box = {
            "label": "display_formula",
            "score": best_score,
            "coordinate": [min_x, min_y, max_x, max_y]
        }
        final_formula_boxes.append(new_box)

    return final_formula_boxes

# --- 3. CORE PROCESSING LOGIC ---
def process_pdf_logic(pdf_path, temp_dir):
    """
    Refactored version of your 'Actual Code' loop.
    Returns a list of results instead of printing.
    """
    results_data = []

    # A. Open PDF
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"error": f"Could not open PDF: {str(e)}"}

    # B. Regex Scan
    print("   🔍 Scanning PDF for equation citations...")
    pattern = re.compile(r"(?i)\b(?:equations?|equ\.?|eqn|eq\.?|formula)\s*[\(\.\- ]*\s*([0-9]+|\([0-9]+\))\b")

    pages_of_interest_indices = set()
    for i, page in enumerate(doc):
        text = page.get_text()
        if pattern.search(text):
            pages_of_interest_indices.update({i - 1, i, i + 1})

    pages_to_process = sorted([idx + 1 for idx in pages_of_interest_indices if 0 <= idx < len(doc)])
    
    if not pages_to_process:
        return {"message": "No equations found matching pattern", "results": []}

    print(f"   ✅ Pages selected: {pages_to_process}")

    # C. Process Selected Pages
    for page_num in pages_to_process:
        page_idx = page_num - 1
        page = doc[page_idx]
        
        # 1. Convert to Image
        mat = fitz.Matrix(2, 2) 
        pix = page.get_pixmap(matrix=mat)
        
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 3: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif pix.n == 4: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Save Full Page Temp Image (Required for Paddle)
        page_img_name = f"page_{page_num}.png"
        page_img_path = os.path.join(temp_dir, page_img_name)
        cv2.imwrite(page_img_path, img_bgr)

        # 2. Layout Detection
        initial_output = pipeline.predict(page_img_path)
        
        for res in initial_output:
            json_data = res.json
            
            # Apply Smart Logic
            updated_formula_boxes = apply_smart_width_logic(json_data)
            
            # 3. Crop & Re-OCR
            for idx, box in enumerate(updated_formula_boxes):
                coords = [int(c) for c in box['coordinate']]
                x1, y1, x2, y2 = coords
                
                # Boundary Checks
                y1, x1 = max(0, y1), max(0, x1)
                y2, x2 = min(img_bgr.shape[0], y2), min(img_bgr.shape[1], x2)
                
                if y2 > y1 and x2 > x1:
                    # Crop with padding (from your code: y1-2, y2+2...)
                    crop_img = img_bgr[max(0, y1-2):y2+2, x1:x2+2]
                    
                    crop_filename = f"p{page_num}_f{idx + 1}.png"
                    crop_path = os.path.join(temp_dir, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    
                    # Re-OCR
                    crop_output = pipeline.predict(crop_path)
                    
                    for crop_res in crop_output:
                        # Append result to list
                        results_data.append({
                            "page": page_num,
                            "formula_id": idx + 1,
                            "markdown": crop_res.markdown,
                            # Optional: include raw text or confidence if needed
                            # "text": crop_res.json.get("rec_text", "")
                        })

    return {"status": "success", "processed_pages": pages_to_process, "equations": results_data}

# --- 4. FLASK ROUTE ---
@app.route('/ocr/process', methods=['POST'])
def process_pdf_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    # Create a unique temp directory for this request
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api_processing", request_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded PDF
        file.save(pdf_path)
        
        # RUN YOUR ACTUAL LOGIC
        result = process_pdf_logic(pdf_path, temp_dir)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        # CLEANUP: Delete the temp folder and all images inside it
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- 5. START SERVER ---
if __name__ == '__main__':
    # Ensure base temp folder exists
    os.makedirs("temp_api_processing", exist_ok=True)
    app.run(host='0.0.0.0', port=5001)
