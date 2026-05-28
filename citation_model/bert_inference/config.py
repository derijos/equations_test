"""
config.py
─────────
Single source of truth for every constant and tuning knob in the
bert_inference module.
Edit this file only — nothing else needs changing for basic configuration.
"""

from pathlib import Path

# ── Repo root (two levels up from this file) ──────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = _REPO_ROOT / "outputs" / "scibert_citation_model"

# ── I/O paths ─────────────────────────────────────────────────────────────────
CONTENT_DIR  = _REPO_ROOT / "content"
INPUT_FILE   = CONTENT_DIR / "sample_test_bert_bio.jsonl"

OUTPUT_DIR       = _REPO_ROOT / "inference_outputs"
OUT_TOKENS       = OUTPUT_DIR / "inference_token_level.jsonl"
OUT_SENTENCES    = OUTPUT_DIR / "inference_sentence_level.jsonl"
OUT_METRICS      = OUTPUT_DIR / "inference_metrics.json"

# ── Inference settings ────────────────────────────────────────────────────────
BATCH_SIZE  = 16
MAX_LENGTH  = 512

# ── Label maps ────────────────────────────────────────────────────────────────
LABEL2ID: dict[str, int] = {"O": 0, "B-CIT": 1, "I-CIT": 2}
ID2LABEL:  dict[int, str] = {0: "O",  1: "B-CIT",  2: "I-CIT"}

# ── Citation label set (used for TP/FP/FN logic) ─────────────────────────────
CITATION_LABELS: frozenset[str] = frozenset({"B-CIT", "I-CIT"})
