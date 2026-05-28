"""
e2e_predictor
─────────────
End-to-End PDF to BERT prediction pipeline.

Stitches together PDF extraction, sentence splitting, tokenisation, and
BERT inference to extract citations directly from a PDF, bypassing the
LLM entirely.

Output: JSONL matching the original pipeline format:
    {
        "pdf_file_path":        str,
        "sentence":             str,
        "citations":            List[str],
        "is_citation_sentence": bool
    }

Entry point
───────────
    python run.py
"""
