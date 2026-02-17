import pymupdf
import pymupdf4llm
import json


def extract_formula_using_pumupdf_layouts(pdf_path):
    """
    Uses PyMuPDF's built-in layout detection to find pages with formula boxes.
    Returns: list of page numbers (1-indexed) that contain formulas.
    """
    doc = pymupdf.open(pdf_path)
    md = json.loads(pymupdf4llm.to_json(doc))

    pages_with_formulas = []

    for page in md["pages"]:
        for box in page["boxes"]:
            if box["boxclass"] == "formula":
                print(f"PyMuPDF Layout → Page Number: {page['page_number']}")
                print(box)
                print("-----")
                pages_with_formulas.append(page["page_number"])
                break  # Only need to add page once

    return list(set(pages_with_formulas))
