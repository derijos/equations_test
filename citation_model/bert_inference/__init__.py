"""
bert_inference
──────────────
Token-classification inference module for the SciBERT citation model.

Reads a JSONL file of tokenised sentences (same format as the BIO training
dataset) and produces three output files:

  1. inference_token_level.jsonl
       Per-token probabilities, predicted label, true label, correctness flag.

  2. inference_sentence_level.jsonl
       Per-sentence TP / FP / TN / FN, precision, recall, F1, success flag,
       and lists of correctly-identified / misclassified / false-positive tokens.

  3. inference_metrics.json
       Aggregate micro-averaged metrics across all sentences.

Entry point
───────────
    python run.py [--input FILE] [--model DIR] [--batch-size N] ...

Configuration
─────────────
    Edit config.py — all paths, label maps, and thresholds live there.
"""
