"""
predictor.py
────────────
Batched forward pass + first-subtoken label alignment.

run_batch — tokenizes a list of records, runs the model, and maps
            subtoken predictions back to the original word tokens.

Design notes
────────────
* Padding / truncation happens at the batch level (most efficient).
* Word-id alignment is derived by re-encoding each record individually
  so that ``word_ids()`` is available (the batched encoding does not
  expose per-sample word ids conveniently).
* Only the FIRST subtoken for each word receives a label; subsequent
  subtokens are ignored (standard practice for token classification).
* Words truncated beyond MAX_LENGTH default to label "O" with
  probability vector [1.0, 0.0, 0.0].
"""

import numpy as np
import torch
from torch.nn.functional import softmax

try:
    from .config import ID2LABEL, MAX_LENGTH
except ImportError:
    from config import ID2LABEL, MAX_LENGTH


def run_batch(
    batch_records: list[dict],
    tokenizer,
    model,
    device: str,
) -> list[dict]:
    """
    Tokenize *batch_records*, run a single forward pass, and return
    per-word predictions aligned to the original token list.

    Parameters
    ----------
    batch_records : list[dict]
        Each dict must have a ``"tokens"`` key (list of word strings).
        An optional ``"labels"`` key (list of BIO label strings) is
        carried through but not used during the forward pass.
    tokenizer     : HuggingFace tokenizer
    model         : AutoModelForTokenClassification (already on *device*)
    device        : str — "cuda", "mps", or "cpu"

    Returns
    -------
    list[dict] — one dict per record:
        {
          "tokens":      List[str]         — original words
          "true_labels": List[str]         — ground-truth BIO labels
          "pred_labels": List[str]         — predicted BIO labels
          "probs":       List[List[float]] — [p_O, p_B_CIT, p_I_CIT] per word
        }
    """
    token_lists = [r["tokens"] for r in batch_records]

    # ── Batched encoding (for efficient GPU utilisation) ──────────────────────
    encoding = tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    # logits → probabilities  shape: (batch, seq_len, num_labels)
    probs_all: np.ndarray = softmax(outputs.logits.cpu().float(), dim=-1).numpy()

    results: list[dict] = []

    for i, record in enumerate(batch_records):
        n_words = len(record["tokens"])

        # Re-encode individually to obtain reliable word_ids per sample
        enc_single = tokenizer(
            record["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        word_ids: list[int | None] = enc_single.word_ids()

        word_pred_label: list[str | None]         = [None] * n_words
        word_probs:      list[list[float] | None] = [None] * n_words
        seen_words: set[int] = set()

        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen_words or wid >= n_words:
                continue
            seen_words.add(wid)
            pred_id               = int(np.argmax(probs_all[i, tok_idx]))
            word_pred_label[wid]  = ID2LABEL[pred_id]
            word_probs[wid]       = probs_all[i, tok_idx].tolist()

        # Fill truncated words with default "O"
        for wid in range(n_words):
            if word_pred_label[wid] is None:
                word_pred_label[wid] = "O"
                word_probs[wid]      = [1.0, 0.0, 0.0]

        results.append({
            "tokens":      record["tokens"],
            "true_labels": record.get("labels", ["O"] * n_words),
            "pred_labels": word_pred_label,
            "probs":       word_probs,
        })

    return results
