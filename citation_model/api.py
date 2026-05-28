"""
api.py
──────
Flask REST API for the SciBERT citation extraction model.

Endpoints
─────────
  POST /predict          — extract citations from a list of sentences
  GET  /health           — liveness check

Usage
─────
  python api.py

RunPod
──────
  Expose port 5002 in your RunPod template (HTTP service).
  The API will be reachable at:
    https://<pod-id>-5002.proxy.runpod.net/predict
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from flask import Flask, jsonify, request

# ── Resolve repo root so sub-packages are importable ─────────────────────────
REPO_ROOT = Path(__file__).parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "e2e_predictor")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from predict_sentences import load, predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Citation marker pattern (same trimming logic as predict_sentences CLI) ────
_CIT_START = re.compile(r'^(\(|\[|<sup>|\^|\d|[A-Z])', re.IGNORECASE)


def _trim_citation(span: str) -> str:
    """Strip leading non-marker words that BERT occasionally bleeds into."""
    tokens = span.split()
    for i, tok in enumerate(tokens):
        if _CIT_START.match(tok):
            return " ".join(tokens[i:])
    return span


# ── Load model at startup (once) ─────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "scibert_citation_model_v2" / "run-0" / "checkpoint-2424"
)

log.info("Loading BERT model from %s …", DEFAULT_MODEL_PATH)
MODEL_BUNDLE = load(DEFAULT_MODEL_PATH)
log.info("Model ready ✅")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# =============================================================================
# ROUTES
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Extract citation spans from a list of sentences.

    Request body (JSON)
    -------------------
    {
        "sentences": ["sentence one", "sentence two", ...]
    }

    Response (JSON)
    ---------------
    {
        "results": [
            {
                "sentence":             "Smith et al. (2020) showed ...",
                "citations":            ["Smith et al. (2020)"],
                "is_citation_sentence": true
            },
            ...
        ]
    }

    Optional — include full token-level detail by passing:
        "verbose": true
    This adds "tokens", "pred_labels", and "probs" to each result.
    """
    body = request.get_json(force=True, silent=True)

    # ── Validate input ────────────────────────────────────────────────────────
    if not body or "sentences" not in body:
        return jsonify({"error": "Request body must contain a 'sentences' list."}), 400

    sentences = body["sentences"]

    if not isinstance(sentences, list) or len(sentences) == 0:
        return jsonify({"error": "'sentences' must be a non-empty list of strings."}), 400

    if not all(isinstance(s, str) for s in sentences):
        return jsonify({"error": "All items in 'sentences' must be strings."}), 400

    verbose = bool(body.get("verbose", False))

    # ── Run inference ─────────────────────────────────────────────────────────
    log.info("Predicting %d sentence(s) …", len(sentences))
    raw = predict(sentences, MODEL_BUNDLE)

    # ── Build response ────────────────────────────────────────────────────────
    results = []
    for r in raw:
        entry = {
            "sentence":             r["sentence"],
            "citations":            [_trim_citation(c) for c in r["citations"]],
            "is_citation_sentence": r["is_citation_sentence"],
        }
        if verbose:
            entry["tokens"]      = r["tokens"]
            entry["pred_labels"] = r["pred_labels"]
            entry["probs"]       = r["probs"]
        results.append(entry)

    return jsonify({"results": results}), 200


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
