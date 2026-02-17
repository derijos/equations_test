import os
import threading
from datetime import datetime
import logging

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import fitz  # PyMuPDF
import time
import cv2
import numpy as np
import json
import re
import requests
import ast
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from fpdf import FPDF
import uuid
import shutil
import zipfile as zipfile_mod

# --- LOCAL MODULES ---
try:
    from pdf_layout_formula import extract_formula_using_pumupdf_layouts
    from heuristics_based_pages import scan_markdown_for_equations
except ImportError:
    print("⚠️  Warning: pdf_layout_formula or heuristics_based_pages not found.")
    def extract_formula_using_pumupdf_layouts(path): return []
    def scan_markdown_for_equations(path, client, excluded_pages=None): return []


# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
app.secret_key = "smartreview-equation-analyzer-secret-key"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Suppress Flask polling logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory stores
analysis_store = {}
batch_store = {}

# Model name on vLLM
LLM_MODEL = "gpt-oss:20b"


# ============================================================
# PYDANTIC MODEL
# ============================================================

class MathAnalysisResult(BaseModel):
    total_equations: int = Field(..., description="The total count of equations found")
    total_equations_reason: str = Field(..., description="Explanation for the equation count")
    all_labeled: Literal["yes", "no"] = Field(..., description="Whether every equation has a label")
    all_labeled_reason: str = Field(..., description="Detailed explanation regarding labeling status")
    sequential: Literal["yes", "no"] = Field(..., description="Whether labeled equations follow 1, 2, 3...")
    sequential_reason: str = Field(..., description="Explanation for sequential numbering logic")


# ============================================================
# HELPER: OpenAI client → vLLM
# ============================================================

def get_llm_client(vllm_url):
    """
    vllm_url: host:port  e.g. "123.456.789.0:8119"
    Returns OpenAI client pointing at vLLM server.
    """
    return OpenAI(
        base_url=f"http://{vllm_url}/v1",
        api_key="dummy"  # vLLM does not need a real key
    )


# ============================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================

def clean_and_parse_json(raw_text: str):
    if not raw_text:
        raise ValueError("Model returned empty response.")
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        json_candidate = match.group(1)
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            fixed_json = re.sub(r",\s*\}", "}", json_candidate)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {json_candidate[:50]}...")
    raise ValueError(f"No JSON found: {raw_text[:100]}...")


def analyze_with_gemma(equations, client):
    """Final equation analysis using vLLM."""
    json_schema = MathAnalysisResult.model_json_schema()
    prompt = f"""
    You are an expert mathematical document analyzer.
    Return ONLY valid JSON matching this schema: {json_schema}

    Task:
    1. Count total equations.
    2. Check if every equation has a label.
    3. Check if labels follow sequential numbering (1, 2, 3...).
    4. If numbering is sequential but some labels are missing, provide a warning in the reason.
    5. If zero labels exist, sequential is no.

    Extracted Equations:
    {equations}
    """
    print("📤 Sending final analysis to vLLM...")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048
        )
        raw_json = response.choices[0].message.content
        cleaned_dict = clean_and_parse_json(raw_json)
        validated_data = MathAnalysisResult.model_validate(cleaned_dict)
        return validated_data.model_dump()
    except Exception as e:
        return {
            "error": "Validation failed",
            "details": str(e),
            "raw": raw_json if 'raw_json' in locals() else "No response"
        }


def get_vertical_overlap(box_a, box_b):
    y_min_a, y_max_a = box_a[1], box_a[3]
    y_min_b, y_max_b = box_b[1], box_b[3]
    inter_min = max(y_min_a, y_min_b)
    inter_max = min(y_max_a, y_max_b)
    return max(0, inter_max - inter_min)


def extract_equations(data_list, client):
    """Per-formula extraction using vLLM."""
    embedded_pattern = re.compile(r"(\((?:\d+(?:\.\d+)*)\)|\[(?:\d+(?:\.\d+)*)\])[\s\u4e00-\u9fff\.\)\}]*$")
    lx = []
    lx1 = []

    for i in data_list:
        data_str = json.dumps(i)
        prompt = f"""
        You are a mathematical formula extractor. Parse the following JSON data and extract equations.

        Rules:
        1. Return a python list of strings in format: "Equation Latex (label)".
        2. If there is more than one equation then split equations and return all equations in python list.
        3. If an equation label exists inside the latex (like `(11)`), use it.
        4. If the last equation has no embedded label, use the formula_number field.

        Input Data:
        {data_str}

        Output only the list:
        """
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        equations = response.choices[0].message.content
        i["equations_extracted"] = equations
        lx.append(i)

    for i in lx:
        for j in ast.literal_eval(i["equations_extracted"]):
            match = embedded_pattern.search(j)
            if match:
                eq_number = match.group(1)
                temp = i.copy()
                temp["formula_number"] = eq_number
                temp["equations_extracted"] = j
                lx1.append(temp)
            else:
                temp = i.copy()
                temp["formula_number"] = ""
                temp["equations_extracted"] = j
                lx1.append(temp)
    return lx1


def equation_overlap_y(data_list, client):
    """Overlap analysis between adjacent formulas using vLLM."""
    print("Starting equation overlap analysis...")
    lx = []
    i = 0
    overlapping_equations = []
    x1, x2, y1, y2 = [], [], [], []
    formula = ""

    while True:
        data_str = data_list[i]

        if i == len(data_list) - 1:
            if len(overlapping_equations) > 0:
                overlapping_equations.append(data_str)
                for j in overlapping_equations:
                    x1.append(j["coordinate"][0])
                    y1.append(j["coordinate"][1])
                    x2.append(j["coordinate"][2])
                    y2.append(j["coordinate"][3])
                    formula = formula + j["equations_extracted"] + "\n\n\n\n"
                coordinate = [min(x1), min(y1), max(x2), max(y2)]
                temp = overlapping_equations[0].copy()
                temp["coordinate"] = coordinate
                temp["equations_extracted"] = formula
                lx.append(temp)
                break
            else:
                lx.append(data_str)
                break

        coords_close = abs(data_str["coordinate"][3] - data_list[i + 1]["coordinate"][1]) <= 20
        coords_same = (data_str["coordinate"][0] == data_list[i + 1]["coordinate"][0]
                       and data_str["coordinate"][1] == data_list[i + 1]["coordinate"][1]
                       and data_str["coordinate"][2] == data_list[i + 1]["coordinate"][2]
                       and data_str["coordinate"][3] == data_list[i + 1]["coordinate"][3])

        fn1 = data_str["formula_number"]
        fn2 = data_list[i + 1]["formula_number"]
        formula_compatible = (fn1 == fn2) or (fn1 == "") or (fn2 == "")

        if (coords_close or coords_same) and formula_compatible:
            formula1 = data_str["equations_extracted"]
            formula2 = data_list[i + 1]["equations_extracted"]
            print(formula1, data_str["coordinate"])
            print(formula2, data_list[i + 1]["coordinate"])

            prompt = f"""
            You are a mathematics expert. Compare the two LaTeX formulas below.

            Task: Determine if Formula 2 is derived or relies on Formula 1 or vice versa.
            This can be through direct reference, shared variables, or logical progression.
            If they are symmetrically equivalent or share the same structural pattern but involve
            different variables then give yes, if completely unrelated give no.

            Formula 1: {formula1}
            Formula 2: {formula2}

            Output Requirement:
            Return ONLY a Python list containing two strings: ["yes" or "no", "A short 1-sentence explanation"].
            Do not output Markdown or extra text. Just the list.
            """
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256
                )
                content = ast.literal_eval(response.choices[0].message.content.strip())
                print("Overlap Response:", content)
                if content[0].lower() == "yes":
                    data_str["formula_number"] = data_list[i + 1]["formula_number"]
                    data_str["equations_combined_reason"] = content[1]
                    overlapping_equations.append(data_str)
                    i += 1
                else:
                    lx.append(data_str)
                    i += 1
            except Exception as e:
                print(f"Error in overlap analysis: {e}")
                i += 1
                continue
        else:
            if len(overlapping_equations) > 0:
                overlapping_equations.append(data_str)
                for j in overlapping_equations:
                    x1.append(j["coordinate"][0])
                    y1.append(j["coordinate"][1])
                    x2.append(j["coordinate"][2])
                    y2.append(j["coordinate"][3])
                    formula = formula + j["equations_extracted"] + "\n\n\n\n"
                coordinate = [min(x1), min(y1), max(x2), max(y2)]
                temp = overlapping_equations[0].copy()
                temp["coordinate"] = coordinate
                temp["equations_extracted"] = formula
                lx.append(temp)
                i += 1
                overlapping_equations = []
                x1, x2, y1, y2 = [], [], [], []
                formula = ""
            else:
                lx.append(data_str)
                i += 1

    for i in lx:
        print("Final Equation with Numbering:\n", i)
    return lx


def normalize_equation_numbers(data_list):
    print("Starting equation numbering normalization...")
    lx = []
    lx1 = [data_list[0]]
    x1, x2, y1, y2 = [], [], [], []
    formula = ""

    if len(data_list) == 1:
        lx.append(data_list[0])
        return lx

    for i in range(0, len(data_list)):
        if i == len(data_list) - 1:
            if len(lx1) == 0:
                lx.append(data_list[i])
                break
            else:
                if lx1[-1]["formula_number"] == "":
                    for k in lx1:
                        lx.append(k)
                    break
                else:
                    for j in lx1:
                        x1.append(j["coordinate"][0])
                        y1.append(j["coordinate"][1])
                        x2.append(j["coordinate"][2])
                        y2.append(j["coordinate"][3])
                        formula = formula + j["equations_extracted"] + "\n\n\n\n"
                    coordinate = [min(x1), min(y1), max(x2), max(y2)]
                    temp = lx1[0].copy()
                    temp["coordinate"] = coordinate
                    temp["equations_extracted"] = formula
                    lx.append(temp)
                    break

        if data_list[i]["formula_number"] != data_list[i + 1]["formula_number"]:
            if len(lx1) == 0:
                lx.append(data_list[i])
            else:
                if lx1[-1]["formula_number"] == "":
                    for k in lx1:
                        lx.append(k)
                else:
                    for j in lx1:
                        x1.append(j["coordinate"][0])
                        y1.append(j["coordinate"][1])
                        x2.append(j["coordinate"][2])
                        y2.append(j["coordinate"][3])
                        formula = formula + j["equations_extracted"] + "\n\n\n\n"
                    coordinate = [min(x1), min(y1), max(x2), max(y2)]
                    temp = lx1[0].copy()
                    temp["coordinate"] = coordinate
                    temp["equations_extracted"] = formula
                    lx.append(temp)
                    lx1 = [data_list[i + 1]]
                    x1, x2, y1, y2 = [], [], [], []
                    formula = ""
        else:
            lx1.append(data_list[i + 1])

    for i in lx:
        print("Normalized Equation Block:", i)
    return lx


def apply_smart_width_logic1(json_data, client):
    all_boxes = json_data.get("parsing_res_list", [])
    display_formulas = []

    for b in all_boxes:
        if b['block_label'] == 'display_formula':
            display_formulas.append(b)
        elif b['block_label'] == 'formula_number':
            if display_formulas:
                display_formulas[-1]["formula_number"] = b['block_content']

    lines = []
    if display_formulas:
        current_line = [display_formulas[0]]
        for i in range(1, len(display_formulas)):
            curr = display_formulas[i]
            prev = current_line[-1]
            if get_vertical_overlap(prev['block_bbox'], curr['block_bbox']) > 0:
                current_line.append(curr)
            else:
                lines.append(current_line)
                current_line = [curr]
        lines.append(current_line)

    final_formula_boxes = []

    for line in lines:
        formula_coords = [item['block_bbox'] for item in line]
        min_y = min(c[1] for c in formula_coords)
        max_y = max(c[3] for c in formula_coords)
        min_x = min(c[0] for c in formula_coords)
        max_x = max(c[2] for c in formula_coords)

        for box in all_boxes:
            b_coord = box['block_bbox']
            b_ymin, b_ymax = b_coord[1], b_coord[3]
            overlap = max(0, min(max_y, b_ymax) - max(min_y, b_ymin))
            if overlap > 0:
                if b_coord[2] > max_x:
                    max_x = b_coord[2]

        for i in line:
            new_box = {
                "label": "equation_detected",
                "coordinate": [min_x, min_y, max_x, max_y],
                "formula_latex": i['block_content'],
                "formula_number": i.get('formula_number', '')
            }
            final_formula_boxes.append(new_box)

    if len(final_formula_boxes) > 0:
        results = extract_equations(final_formula_boxes, client)
        print("Extracted Equations:")
        for r in results:
            print(r)

        results = equation_overlap_y(results, client)
        return results

    return final_formula_boxes


# ============================================================
# MAIN ANALYSIS PIPELINE
# ============================================================

def run_analysis(pdf_path, vllm_url, paddle_url):
    """
    Full analysis pipeline.
    vllm_url:   host:port of vLLM on RunPod  e.g. "123.45.67.89:8119"
    paddle_url: full URL of PaddleOCR API    e.g. "http://123.45.67.89:5001/paddleocr_json"
    """
    client = get_llm_client(vllm_url)

    total_start_time = time.time()
    timing = {}

    session_id = str(uuid.uuid4())
    output_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📄 Opening PDF: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return None, None, {"error": f"Could not open PDF: {e}"}, session_id, 0, None, None

    # --- Stage 1: PyMuPDF layout detection ---
    print("🔍 Running PyMuPDF layout detection...")
    t0 = time.time()
    pdf_layout_detected_pages = extract_formula_using_pumupdf_layouts(pdf_path)
    timing["pymupdf_layout"] = round(time.time() - t0, 2)
    print(f"   PyMuPDF found pages: {pdf_layout_detected_pages}")

    # --- Stage 2: Heuristic scan on remaining pages ---
    print("🔍 Running heuristic scan...")
    t0 = time.time()
    heuristic_pages = scan_markdown_for_equations(
        pdf_path,
        client=client,
        excluded_pages=pdf_layout_detected_pages
    )
    timing["heuristic_search"] = round(time.time() - t0, 2)
    print(f"   Heuristic found pages: {heuristic_pages}")

    # --- Merge all detected pages ---
    pages_to_process = sorted(list(set(pdf_layout_detected_pages + heuristic_pages)))
    print(f"✅ Final pages to process: {pages_to_process}")

    all_equation_data = []
    page_equation_data = {}
    raw_paddle_data = {}
    all_page_images = []
    annotated_page_map = {}
    total_pages = len(doc)

    # --- Render ALL pages as images ---
    print("🖼️  Rendering all pages as images...")
    for page_num in range(total_pages):
        page = doc[page_num]
        real_page_num = page_num + 1
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        plain_img_path = os.path.join(output_dir, f"page_{real_page_num}.png")
        pix.save(plain_img_path)
        all_page_images.append(plain_img_path)

    # --- Process equation pages via PaddleOCR on RunPod ---
    t0 = time.time()
    for page_num in range(total_pages):
        real_page_num = page_num + 1
        if real_page_num not in pages_to_process:
            continue

        print(f"\n--- Processing Page {real_page_num}/{total_pages} ---")
        start_page = time.time()

        plain_img_path = all_page_images[page_num]
        img_bgr = cv2.imread(plain_img_path)

        # Send page image to PaddleOCR API on RunPod
        with open(plain_img_path, 'rb') as f:
            try:
                response = requests.post(url=paddle_url, files={'file': f})
                json_data = response.json()
                raw_paddle_data[real_page_num] = json_data
            except Exception as e:
                print(f"   ❌ Error calling PaddleOCR API: {e}")
                continue

        if "res" in json_data:
            updated_formula_boxes = apply_smart_width_logic1(json_data["res"], client)
        else:
            updated_formula_boxes = apply_smart_width_logic1(json_data, client)

        print(f"   📊 Found {len(updated_formula_boxes)} formula regions.")

        # Draw bounding boxes on image
        for box in updated_formula_boxes:
            coord = box['coordinate']
            x1, y1, x2, y2 = map(int, coord)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label_text = box.get('label', '')
            if label_text and label_text.strip():
                font_scale = 0.8
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                pad = 6
                bg_top = y1 - th - baseline - pad + 4
                bg_bottom = y1 + 4
                cv2.rectangle(img_bgr, (x1, bg_top), (x1 + tw + pad, bg_bottom), (0, 200, 0), -1)
                cv2.putText(img_bgr, label_text, (x1 + 3, y1 - baseline + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        all_equation_data.extend(updated_formula_boxes)
        page_equation_data[real_page_num] = updated_formula_boxes

        annotated_path = os.path.join(output_dir, f"page_{real_page_num}_annotated.png")
        cv2.imwrite(annotated_path, img_bgr)
        annotated_page_map[real_page_num] = annotated_path

        print(f"   ⏱️  Page time: {time.time() - start_page:.2f}s")

    timing["paddleocr_and_overlap"] = round(time.time() - t0, 2)

    # --- Final LLM analysis ---
    equations_string = ''
    for j, i in enumerate(all_equation_data):
        equations_string += f"{i}\n\n"

    print("\n📤 LLM Input:")
    print(equations_string)

    t0 = time.time()
    val = analyze_with_gemma(equations_string, client)
    timing["final_llm_analysis"] = round(time.time() - t0, 2)

    val["extracted_equations"] = equations_string
    val["pages_to_process"] = pages_to_process

    # --- Build annotated PDF ---
    annotated_pdf_path = build_annotated_pdf(all_page_images, annotated_page_map, output_dir)

    timing["total_time"] = round(time.time() - total_start_time, 2)
    val["timing"] = timing

    print("\n--- Final Analysis Result ---")
    print(val)
    print(f"⏱️  Timing: {timing}")

    # --- Build output package ---
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    package_dir = os.path.join(output_dir, pdf_basename)
    os.makedirs(package_dir, exist_ok=True)

    shutil.copy2(pdf_path, os.path.join(package_dir, os.path.basename(pdf_path)))
    shutil.copy2(annotated_pdf_path, os.path.join(package_dir, "annotated_manuscript.pdf"))

    json_dir = os.path.join(package_dir, "json_annotations")
    os.makedirs(json_dir, exist_ok=True)
    for pn, raw_json in raw_paddle_data.items():
        with open(os.path.join(json_dir, f"page_{pn}.json"), 'w') as f:
            json.dump(raw_json, f, indent=2, default=str)

    with open(os.path.join(package_dir, "final_output.json"), 'w') as f:
        json.dump({
            "total_equations": val.get("total_equations"),
            "total_equations_reason": val.get("total_equations_reason"),
            "all_labeled": val.get("all_labeled"),
            "all_labeled_reason": val.get("all_labeled_reason"),
            "sequential": val.get("sequential"),
            "sequential_reason": val.get("sequential_reason"),
            "pages_to_process": val.get("pages_to_process"),
        }, f, indent=2, default=str)

    with open(os.path.join(package_dir, "extracted_equations.txt"), 'w') as f:
        f.write(equations_string)

    with open(os.path.join(package_dir, "performance.json"), 'w') as f:
        json.dump(timing, f, indent=2)

    zip_path = os.path.join(output_dir, f"{pdf_basename}_output")
    shutil.make_archive(zip_path, 'zip', output_dir, pdf_basename)
    zip_file_path = zip_path + ".zip"

    return all_page_images, annotated_page_map, val, session_id, total_pages, annotated_pdf_path, zip_file_path


def build_annotated_pdf(all_page_images, annotated_page_map, output_dir):
    pdf = FPDF()
    for idx, plain_path in enumerate(all_page_images):
        real_page_num = idx + 1
        img_path = annotated_page_map.get(real_page_num, plain_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        pdf.add_page()
        img_height, img_width = img.shape[:2]
        scale = min(190 / img_width, 277 / img_height)
        display_width = img_width * scale
        display_height = img_height * scale
        x_offset = (210 - display_width) / 2
        pdf.image(img_path, x=x_offset, y=10, w=display_width, h=display_height)

    pdf_path = os.path.join(output_dir, "annotated_manuscript.pdf")
    pdf.output(pdf_path)
    return pdf_path


# ============================================================
# FLASK ROUTES — SINGLE PDF
# ============================================================

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'pdf_file' not in request.files:
        return redirect(url_for('index'))

    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return redirect(url_for('index'))

    # RunPod URLs — entered in the web form
    vllm_url   = request.form.get('vllm_url',   '123.45.67.89:8119')
    paddle_url = request.form.get('paddle_url', 'http://123.45.67.89:5001/paddleocr_json')

    session_id = str(uuid.uuid4())
    upload_dir = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(upload_dir, exist_ok=True)
    pdf_path = os.path.join(upload_dir, pdf_file.filename)
    pdf_file.save(pdf_path)

    all_page_images, annotated_page_map, result, sid, total_pages, annotated_pdf_path, zip_file_path = \
        run_analysis(pdf_path, vllm_url, paddle_url)

    analysis_store[session_id] = {
        'all_page_images': all_page_images,
        'annotated_page_map': annotated_page_map,
        'result': result,
        'total_pages': total_pages,
        'annotated_pdf_path': annotated_pdf_path,
        'zip_file_path': zip_file_path,
        'pdf_filename': pdf_file.filename,
    }

    return redirect(url_for('results', session_id=session_id))


@app.route('/results/<session_id>')
def results(session_id):
    data = analysis_store.get(session_id)
    if not data:
        return redirect(url_for('index'))
    return render_template('results.html',
                           session_id=session_id,
                           result=data['result'],
                           total_pages=data['total_pages'],
                           annotated_pages=list(data['annotated_page_map'].keys()),
                           pdf_filename=data['pdf_filename'])


@app.route('/page_image/<session_id>/<int:page_num>')
def page_image(session_id, page_num):
    data = analysis_store.get(session_id)
    if not data:
        return "Not found", 404
    annotated_map = data['annotated_page_map']
    all_images = data['all_page_images']
    if page_num in annotated_map:
        img_path = annotated_map[page_num]
    elif 0 < page_num <= len(all_images):
        img_path = all_images[page_num - 1]
    else:
        return "Page not found", 404
    return send_file(img_path, mimetype='image/png')


@app.route('/download_pdf/<session_id>')
def download_pdf(session_id):
    data = analysis_store.get(session_id)
    if not data:
        return "Not found", 404
    zip_path = data.get('zip_file_path')
    if not zip_path or not os.path.exists(zip_path):
        return "Output package not found", 404
    pdf_basename = os.path.splitext(data['pdf_filename'])[0]
    return send_file(zip_path, as_attachment=True,
                     download_name=f"{pdf_basename}_output.zip")


# ============================================================
# FLASK ROUTES — BATCH PROCESSING
# ============================================================

@app.route('/analyze_batch', methods=['POST'])
def analyze_batch():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('pdf_files')
    pdf_files = [f for f in files if f.filename and f.filename.lower().endswith('.pdf')]
    if not pdf_files:
        return jsonify({"error": "No valid PDF files found"}), 400

    vllm_url   = request.form.get('vllm_url',   '123.45.67.89:8119')
    paddle_url = request.form.get('paddle_url', 'http://123.45.67.89:5001/paddleocr_json')

    batch_id = str(uuid.uuid4())
    batch_dir = os.path.join(UPLOAD_FOLDER, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)

    saved_paths = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(batch_dir, pdf_file.filename)
        pdf_file.save(pdf_path)
        saved_paths.append((pdf_file.filename, pdf_path))

    batch_store[batch_id] = {
        "status": "processing",
        "total": len(saved_paths),
        "processed": 0,
        "current_file": "",
        "results": [],
    }

    thread = threading.Thread(
        target=_process_batch,
        args=(batch_id, saved_paths, vllm_url, paddle_url),
        daemon=True,
    )
    thread.start()

    return jsonify({"batch_id": batch_id})


def _process_batch(batch_id, saved_paths, vllm_url, paddle_url):
    batch = batch_store[batch_id]

    for i, (filename, pdf_path) in enumerate(saved_paths):
        batch["current_file"] = filename
        print(f"\n{'='*60}")
        print(f"📁 BATCH [{i+1}/{len(saved_paths)}]: {filename}")
        print(f"{'='*60}")

        try:
            all_page_images, annotated_page_map, result, sid, total_pages, annotated_pdf_path, zip_file_path = \
                run_analysis(pdf_path, vllm_url, paddle_url)

            batch["results"].append({
                "filename": filename,
                "status": "success",
                "result": {
                    "total_equations": result.get("total_equations", 0),
                    "total_equations_reason": result.get("total_equations_reason", ""),
                    "all_labeled": result.get("all_labeled", "no"),
                    "all_labeled_reason": result.get("all_labeled_reason", ""),
                    "sequential": result.get("sequential", "no"),
                    "sequential_reason": result.get("sequential_reason", ""),
                    "pages_to_process": result.get("pages_to_process", []),
                },
                "zip_file": zip_file_path,
                "error": None,
            })
            print(f"✅ {filename} done.")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            batch["results"].append({
                "filename": filename,
                "status": "error",
                "result": None,
                "zip_file": None,
                "error": str(e),
            })

        batch["processed"] = i + 1

    batch["status"] = "complete"
    batch["current_file"] = ""
    print(f"\n🎉 Batch {batch_id} complete!")


@app.route('/batch_progress/<batch_id>')
def batch_progress(batch_id):
    batch = batch_store.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404
    return jsonify({
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
        "current_file": batch["current_file"],
    })


@app.route('/batch_results/<batch_id>')
def batch_results(batch_id):
    batch = batch_store.get(batch_id)
    if not batch:
        return redirect(url_for('index'))
    return render_template('batch_results.html',
                           batch_id=batch_id,
                           results=batch["results"])


@app.route('/download_batch_zip/<batch_id>/<int:index>')
def download_batch_zip(batch_id, index):
    batch = batch_store.get(batch_id)
    if not batch:
        return "Batch not found", 404
    if index < 0 or index >= len(batch["results"]):
        return "Invalid index", 404
    item = batch["results"][index]
    zip_path = item.get("zip_file")
    if not zip_path or not os.path.exists(zip_path):
        return "ZIP file not found", 404
    pdf_basename = os.path.splitext(item["filename"])[0]
    return send_file(zip_path, as_attachment=True,
                     download_name=f"{pdf_basename}_output.zip")


@app.route('/download_batch_all/<batch_id>')
def download_batch_all(batch_id):
    batch = batch_store.get(batch_id)
    if not batch:
        return "Batch not found", 404

    zip_files = []
    for item in batch["results"]:
        if item["status"] == "success":
            zp = item.get("zip_file")
            if zp and os.path.exists(zp):
                zip_files.append((item["filename"], zp))

    if not zip_files:
        return "No downloadable results", 404

    date_str = datetime.now().strftime("%Y-%m-%d")
    master_zip_name = f"batch_results_{date_str}_{batch_id[:8]}.zip"
    master_zip_path = os.path.join(UPLOAD_FOLDER, master_zip_name)

    with zipfile_mod.ZipFile(master_zip_path, 'w', zipfile_mod.ZIP_DEFLATED) as zf:
        for original_name, zpath in zip_files:
            arc_name = os.path.splitext(original_name)[0] + "_output.zip"
            zf.write(zpath, arc_name)

    return send_file(master_zip_path, as_attachment=True,
                     download_name=master_zip_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
