"""NRC VAD scoring (Task 3).

Valence, arousal and dominance come from the NRC VAD lexicon
(data/raw/lexicons/NRC-VAD-Lexicon.txt). Means are divided by the
number of dictionary hits, not by the total number of tokens.

Tokenizing and lemmatizing live in text.py (shared by all tasks).
"""
from pathlib import Path

import pandas as pd

from personality.text import tokenize

VAD_PATH = Path("data/raw/lexicons/NRC-VAD-Lexicon.txt")


def load_vad(path=VAD_PATH):
    """word -> (valence, arousal, dominance). Scores are in [0, 1]."""
    vad = {}
    with open(path, encoding="utf-8") as lines:
        for line in lines:
            parts = line.split()
            if len(parts) != 4:
                continue
            word, valence, arousal, dominance = parts
            vad[word] = (float(valence), float(arousal), float(dominance))
    return vad


def speaker_vad(texts, vad):
    """VAD stats for one speaker's pile of texts.

    Each lemmatized token that sits in `vad` is one hit. Mean / std /
    min / max are over those hits only.
    """
    hits = []
    for text in texts:
        for token in tokenize(text):
            scores = vad.get(token)
            if scores is not None:
                hits.append(scores)
    cols = ["valence", "arousal", "dominance"]
    found = pd.DataFrame(hits, columns=cols)
    row = {"n_vad_hits": len(found)}
    for col in cols:
        row[f"{col}_mean"] = found[col].mean()
        row[f"{col}_std"] = found[col].std()
        row[f"{col}_min"] = found[col].min()
        row[f"{col}_max"] = found[col].max()
    return row
