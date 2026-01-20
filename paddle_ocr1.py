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

app = Flask(__name__)

# --- GLOBAL MODEL ---
print("⏳ Loading PaddleOCR-VL Model...")
try:
    pipeline = PaddleOCRVL()
    print("✅ Model Loaded.")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    exit()

# --- HELPER FUNCTIONS ---
def get_vertical_overlap(box_a, box_b):
    y_min_a, y_max_a = box_a[1], box_a[3]
    y_min_b, y_max_b = box_b[1], box_b[3]
    inter_min = max(y_min_a, y_min_b)
    inter_max = min(y_max_a, y_max_b)
    return max(0, inter_max - inter_min)

def apply_smart_width_logic(json_data, logger):
    """
    Groups formulas by line. Returns list of boxes.
    """
    # 1. EXTRACT BOXES CAREFULLY
    # all_boxes = []
    all_boxes = json_data.get("layout_det_res", {}).get("boxes", [])
    # if isinstance(json_data, dict):
    #     if "res" in json_data and isinstance(json_data["res"], dict):
    #         all_boxes = json_data["res"].get("boxes", [])
    #     elif "layout_det_res" in json_data:
    #         all_boxes = json_data["layout_det_res"].get("boxes", [])
    #     elif "boxes" in json_data:
    #         all_boxes = json_data["boxes"]
    
    if not all_boxes:
        logger(f"   ⚠️ No 'boxes' found in JSON structure. Keys: {json_data.keys() if isinstance(json_data, dict) else 'Not Dict'}")
        return []

    # 2. FILTER FORMULAS
    display_formulas = [b for b in all_boxes if b.get('label') == 'display_formula']
    logger(f"   📊 All Boxes: {len(all_boxes)} | Display Formulas: {len(display_formulas)}")

    if not display_formulas:
        return []

    display_formulas.sort(key=lambda x: x['coordinate'][1]) 

    # 3. GROUP BY LINE
    lines = []
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

    # 4. CALCULATE SMART WIDTH
    final_formula_boxes = []
    for line in lines:
        formula_coords = [item['coordinate'] for item in line]
        min_y = min(c[1] for c in formula_coords)
        max_y = max(c[3] for c in formula_coords)
        min_x = min(c[0] for c in formula_coords) 
        max_x = max(c[2] for c in formula_coords) 
        
        for box in all_boxes:
            b_coord = box['coordinate']
            b_ymin, b_ymax = b_coord[1], b_coord[3]
            overlap = max(0, min(max_y, b_ymax) - max(min_y, b_ymin))
            if overlap > 0 and b_coord[2] > max_x:
                max_x = b_coord[2]

        best_score = max(item['score'] for item in line)
        final_formula_boxes.append({
            "label": "display_formula",
            "score": best_score,
            "coordinate": [min_x, min_y, max_x, max_y]
        })

    return final_formula_boxes

# --- PROCESS LOGIC ---
def process_pdf_logic(pdf_path, temp_dir):
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    results_data = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    # REGEX SCAN
    pattern = re.compile(r"(?i)\b(?:equations?|equ\.?|eqn|eq\.?|formula)\s*[\(\.\- ]*\s*([0-9]+|\([0-9]+\))\b")
    pages_of_interest = set()
    for i, page in enumerate(doc):
        if pattern.search(page.get_text()):
            pages_of_interest.update({i - 1, i, i + 1})
    
    pages_to_process = sorted([p + 1 for p in pages_of_interest if 0 <= p < len(doc)])
    log(f"✅ Pages Selected: {pages_to_process}")

    for page_num in pages_to_process:
        try:
            log(f"--- Processing Page {page_num} ---")
            page = doc[page_num - 1]
            
            # SAVE IMAGE
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 3: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pix.n == 4: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else: img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            page_img_path = os.path.join(temp_dir, f"page_{page_num}.png")
            cv2.imwrite(page_img_path, img_bgr)
            
            # PREDICT LAYOUT
            initial_output = pipeline.predict(page_img_path)
            log(f"   Layout detected. Results count: {len(initial_output)}")

            for res in initial_output:
                json_data = res.json
                # print(json_data)
                # log(f"   JSON Keys: {json_data.keys()}") # Uncomment if desperate
                
                # EXTRACT FORMULAS
                formulas = apply_smart_width_logic(json_data["res"], log)
                log(f"   Found {len(formulas)} smart formulas.")

                # CROP & RE-OCR
                for idx, box in enumerate(formulas):
                    coords = [int(c) for c in box['coordinate']]
                    x1, y1, x2, y2 = coords
                    
                    y1, x1 = max(0, y1), max(0, x1)
                    y2, x2 = min(img_bgr.shape[0], y2), min(img_bgr.shape[1], x2)
                    # Add Padding & Clip
                    # x1, y1 = max(0, x1), max(0, y1)
                    # x2, y2 = min(img_bgr.shape[1], x2 + 2), min(img_bgr.shape[0], y2 + 2)

                    if y2 > y1 and x2 > x1:
                        crop = img_bgr[y1-2:y2+2, x1:x2+2]
                        # crop = img_bgr[max(0, y1-2):y2, x1:x2]
                        crop_path = os.path.join(temp_dir, f"p{page_num}_f{idx}.png")
                        cv2.imwrite(crop_path, crop)

                        crop_out = pipeline.predict(crop_path)
                        for c_res in crop_out:
                            md = c_res.markdown
                            log(f"      Markdown: {md}...") 
                            results_data.append({
                                "page": page_num,
                                "id": idx+1,
                                "markdown": md["markdown_texts"]
                            })
        except Exception as e:
            log(f"❌ Error on Page {page_num}: {e}")

    return {
        "status": "success",
        "processed_pages": pages_to_process,
        "equations": results_data,
        "debug_logs": logs  # <--- CHECK THIS IN RESPONSE
    }

# --- ROUTE ---
@app.route('/ocr/process', methods=['POST'])
def process_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    req_id = str(uuid.uuid4())
    temp_dir = os.path.join("temp_api", req_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        pdf_path = os.path.join(temp_dir, file.filename)
        file.save(pdf_path)
        
        result = process_pdf_logic(pdf_path, temp_dir)
        return jsonify(result)
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    os.makedirs("temp_api", exist_ok=True)
    app.run(host='0.0.0.0', port=5001)
