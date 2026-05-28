"""
model.py
────────
Device detection and model / tokenizer loading.

load_model — auto-selects CUDA → MPS → CPU, loads the tokenizer and
             AutoModelForTokenClassification, puts the model in eval mode.
"""

import os
from pathlib import Path

# Force transformers / huggingface_hub to skip remote lookups so that
# local checkpoint directories are accepted without repo-id validation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_model(
    model_path: str | Path,
) -> tuple:
    """
    Load the SciBERT token-classification model and its tokenizer.

    Device priority: CUDA → Apple MPS → CPU.

    Parameters
    ----------
    model_path : path-like — directory containing the saved model + tokenizer

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForTokenClassification, str]
        (tokenizer, model, device_string)
    """
    model_path = str(model_path)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading model from: {model_path}  |  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device)
    model.eval()

    return tokenizer, model, device
