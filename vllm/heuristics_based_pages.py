import pymupdf
import pymupdf4llm
import re
import json
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI


# --- PYDANTIC MODEL ---
class MathAnalysisResult(BaseModel):
    equation_present: Literal["yes", "no"] = Field(..., description="Check whether standalone equation present in the given chunk")
    extracted_equation: str = Field(..., description="Return equation if present")
    equation_label: str = Field(..., description="Return equation label if present")


def get_llm_client(vllm_url):
    return OpenAI(
        base_url=f"http://{vllm_url}/v1",
        api_key="dummy"
    )


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
    raise ValueError(f"No JSON found in response: {raw_text[:100]}...")


def analyze_with_gpt(equations, client):
    """Validates a suspect line with vLLM via OpenAI-compatible API."""
    schema = MathAnalysisResult.model_json_schema()
    prompt = f"""
    You are an expert mathematical document analyzer.
    Return ONLY valid JSON matching this schema: {schema}

    Check if there is a standalone equation (not inline) in the given text.
    If inline equation return no. If standalone equation return yes with equation and label if present.

    Text:
    {equations}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512
        )
        raw_json = response.choices[0].message.content
        cleaned_dict = clean_and_parse_json(raw_json)
        validated = MathAnalysisResult.model_validate(cleaned_dict)
        return validated.model_dump()
    except Exception as e:
        return {
            "error": "Validation failed",
            "details": str(e),
            "equation_present": "no"
        }


def scan_markdown_for_equations(pdf_path, client, excluded_pages=None):
    """
    Scans PDF pages using symbol-based heuristics + LLM validation.
    Skips pages in excluded_pages (already detected by PyMuPDF).
    Collects ALL suspects per page, but stops LLM validation per page
    once one suspect is confirmed as an equation.

    Returns: sorted list of unique page numbers confirmed to contain equations.
    """
    if excluded_pages is None:
        excluded_pages = []

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return []

    # --- SYMBOL HEURISTICS ---
    math_symbols = [
        r'[\u2200-\u22FF]',    # Math Operators & Logic
        r'[\u0370-\u03FF]',    # Greek Characters
        r'[\u2190-\u21FF]',    # Arrows
        r'[\u2070-\u209F]',    # Superscripts/Subscripts
        r'[∫∑∏∂∇√≈≠≤≥±∞∆Ωμσερϕτ]'
    ]

    label_pattern = re.compile(r"(\(\d+(?:\.\d+)*\)|\[\d+(?:\.\d+)*\]|\d+\.\d+(?:\.\d+)*)\s*$")
    caption_pattern = re.compile(r"^\s*(?:\*\*|)?(?:Table|Tab\.|Figure|Fig\.)\s+[0-9IVX]+", re.IGNORECASE)
    table_pattern = re.compile(r"^\s*(\+[-=]+\+|\|.*\|)", re.MULTILINE)
    reference_pattern = re.compile(r"^\s*(\(\d+\)|\[\d+\])\s+[A-Z]", re.IGNORECASE)

    skip_next = False
    suspects = []

    for page_num in range(1, len(doc)):
        # Skip pages already detected by PyMuPDF layout
        if (page_num + 1) in excluded_pages:
            continue

        try:
            page_text = pymupdf4llm.to_text(doc, pages=[page_num])
            lines = page_text.split('\n')
        except Exception:
            continue

        for line in lines:
            clean_line = line.strip()

            if not clean_line or len(clean_line) < 3:
                continue
            if clean_line.isdigit():
                continue
            if skip_next:
                skip_next = False
                continue
            if caption_pattern.match(clean_line):
                skip_next = True
                continue
            if table_pattern.match(clean_line):
                continue
            if reference_pattern.match(clean_line):
                continue

            # --- SCORING ---
            score = 0
            for pattern in math_symbols:
                matches = re.findall(pattern, clean_line)
                score += len(matches) * 5
            if label_pattern.search(clean_line):
                score += 25
            if len(clean_line) < 150 and score > 0:
                score += 5
            if clean_line.startswith("∆") or clean_line.startswith("δ"):
                score += 10

            if score >= 20:
                print(f"🔎 [Suspect] Page: {page_num + 1} | Score: {score} | \"{clean_line}\"")
                suspects.append([page_num + 1, clean_line])
                # Collect ALL suspects on this page, don't break

    # --- LLM VALIDATION ---
    # Once a page is confirmed, skip remaining suspects from that page
    confirmed_pages = set()

    for suspect in suspects:
        page_num = suspect[0]
        line_text = suspect[1]

        # Page already confirmed, skip other suspects from this page
        if page_num in confirmed_pages:
            continue

        print(f"🤖 Validating page {page_num}: \"{line_text[:80]}...\"")
        val = analyze_with_gpt(line_text, client)

        if val.get("equation_present") == "yes":
            print(f"   ✅ Confirmed equation on page {page_num}")
            confirmed_pages.add(page_num)
        else:
            print(f"   ❌ Not an equation")

    return sorted(confirmed_pages)
