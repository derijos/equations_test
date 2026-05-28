"""
predict_sentences.py
────────────────────
Send a list of plain-text sentences directly to the BERT citation model
and receive per-sentence citation predictions.

No PDF extraction, no LLM, no web server — pure BERT inference only.

Usage (Python API)
──────────────────
    from predict_sentences import load, predict

    model_bundle = load()          # call once; caches model in memory

    results = predict(
        ["Smith et al. (2020) showed that ...",
         "No citations here at all."],
        model_bundle,
    )

    for r in results:
        print(r["sentence"])
        print(r["citations"])
        print(r["is_citation_sentence"])

Usage (CLI)
───────────
    python predict_sentences.py "First sentence." "Second sentence [1]."
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── Resolve repo root so all sub-packages are importable ─────────────────────
REPO_ROOT = Path(__file__).parent
for _p in [str(REPO_ROOT), str(REPO_ROOT / "e2e_predictor")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Imports from existing modules (only what is needed) ───────────────────────
from bert_inference.model     import load_model
from bert_inference.predictor import run_batch
from span_reconstructor       import reconstruct_citations

# ── Default model checkpoint (same as app.py) ────────────────────────────────
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "scibert_citation_model_v2" / "run-0" / "checkpoint-2424"
)

# ── Inference batch size ──────────────────────────────────────────────────────
BATCH_SIZE = 16


# =============================================================================
# PUBLIC API
# =============================================================================

def load(model_path: str | Path = DEFAULT_MODEL_PATH) -> tuple:
    """
    Load the BERT model, tokenizer, and resolve the device.

    Parameters
    ----------
    model_path : path-like
        Directory that contains the saved checkpoint.
        Defaults to the v2 checkpoint shipped with this repo.

    Returns
    -------
    tuple  (tokenizer, model, device_string)
        Pass this directly to ``predict()`` as *model_bundle*.
    """
    return load_model(model_path)


def predict(
    sentences: list[str],
    model_bundle: tuple,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Run BERT citation extraction on a list of plain-text sentences.

    Parameters
    ----------
    sentences    : list[str]  — raw sentences to classify
    model_bundle : tuple      — (tokenizer, model, device) from ``load()``
    batch_size   : int        — how many sentences to send per forward pass

    Returns
    -------
    list[dict], one entry per input sentence:
    {
        "sentence"            : str          — original sentence
        "tokens"              : list[str]    — whitespace tokens
        "pred_labels"         : list[str]    — BIO label per token
        "probs"               : list[list]   — [p_O, p_B-CIT, p_I-CIT] per token
        "citations"           : list[str]    — reconstructed citation strings
        "is_citation_sentence": bool         — True if ≥1 citation found
    }
    """
    tokenizer, model, device = model_bundle

    # ── Tokenize: whitespace split preserving character spans ────────────────
    records = []
    for sent in sentences:
        tokens = re.findall(r'\S+', sent)   # same strategy as tokenize_with_spans
        records.append({"text": sent, "tokens": tokens})

    # ── Batch forward passes ─────────────────────────────────────────────────
    all_preds: list[dict] = []
    for i in range(0, len(records), batch_size):
        batch      = records[i : i + batch_size]
        batch_out  = run_batch(batch, tokenizer, model, device)
        all_preds.extend(batch_out)

    # ── Build output ──────────────────────────────────────────────────────────
    results: list[dict] = []
    for record, pred in zip(records, all_preds):
        tokens      = pred["tokens"]
        pred_labels = pred["pred_labels"]
        probs       = pred["probs"]

        citations = reconstruct_citations(tokens, pred_labels)

        results.append({
            "sentence":             record["text"],
            "tokens":               tokens,
            "pred_labels":          pred_labels,
            "probs":                [
                {
                    "token":       tok,
                    "label":       lbl,
                    "prob_O":      round(p[0], 4),
                    "prob_B_CIT":  round(p[1], 4),
                    "prob_I_CIT":  round(p[2], 4),
                    "max_prob":    round(max(p), 4),
                }
                for tok, lbl, p in zip(tokens, pred_labels, probs)
            ],
            "citations":            citations,
            "is_citation_sentence": len(citations) > 0,
        })

    return results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

# ── Citation marker pattern: what a real citation token starts with ───────────
# Keeps spans that begin with (, [, <sup>, ^, digit, or uppercase initial.
_CIT_START = re.compile(
    r'^(\(|\[|<sup>|\^|\d|[A-Z])',
    re.IGNORECASE,
)


def _trim_citation(span: str) -> str:
    """
    Remove any leading words that do not look like citation markers.
    BERT sometimes bleeds the B-CIT tag onto the word immediately before
    the actual marker (e.g. 'susceptible (2, 3)' instead of '(2, 3)').
    """
    tokens = span.split()
    for i, tok in enumerate(tokens):
        if _CIT_START.match(tok):
            return " ".join(tokens[i:])
    return span   # fall back to the full span if nothing matches


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python predict_sentences.py \"sentence 1\" \"sentence 2\" ...")
        sys.exit(1)

    input_sentences = sys.argv[1:]

    print(f"⏳  Loading BERT model from {DEFAULT_MODEL_PATH} …", file=sys.stderr)
    bundle = load()
    print(f"✅  Model ready", file=sys.stderr)

    predictions = predict(input_sentences, bundle)

    output = [
        {
            "sentence":  r["sentence"],
            "citations": [_trim_citation(c) for c in r["citations"]],
        }
        for r in predictions
    ]

    print(json.dumps(output, indent=2, ensure_ascii=False))
