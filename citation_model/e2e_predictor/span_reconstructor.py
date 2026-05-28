"""
span_reconstructor.py
─────────────────────
Reconstructs full citation strings from word tokens and BIO labels.
"""

def reconstruct_citations(tokens: list[str], labels: list[str]) -> list[str]:
    """
    Given a list of words and their corresponding predicted BIO labels,
    reconstruct the continuous citation strings.

    Example:
      tokens: ["This", "is", "a", "test", "[", "1", ",", "2", "]"]
      labels: ["O",    "O",  "O", "O",    "B-CIT", "I-CIT", "I-CIT", "I-CIT", "I-CIT"]
    Returns:
      ["[ 1 , 2 ]"]  # Spaces are preserved from the tokenization
    """
    citations = []
    current_span = []

    for token, label in zip(tokens, labels):
        if label == "B-CIT":
            if current_span:
                citations.append(" ".join(current_span))
            current_span = [token]
        elif label == "I-CIT":
            if current_span:
                current_span.append(token)
            else:
                # Malformed sequence (I-CIT without B-CIT)
                # Recover gracefully by starting a new span
                current_span = [token]
        else:
            if current_span:
                citations.append(" ".join(current_span))
                current_span = []

    if current_span:
        citations.append(" ".join(current_span))

    # Strip surrounding whitespace from each reconstructed citation
    return [c.strip() for c in citations if c.strip()]
