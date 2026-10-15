"""Word counting (Tasks 1 and 4). The frozen word rules.

The lists live in assets/word_rules.csv, not in code. Provenance:
derived from the LIWC "assent" and "negate" categories (Pennebaker et
al.) — the standard in personality-from-language research, but licensed,
so this is our free, documented approximation. Polysemy traps ("right",
"true", "fine", "way") are deliberately excluded.

Tokenization and lemmatization live in text.py (shared by all tasks).
"""
import csv
from pathlib import Path

import pandas as pd

from personality.text import tokenize

_ASSETS_DIR = Path(__file__).parent / "assets"


def _load_words(wanted):
    """One frozen word list from assets/word_rules.csv."""
    with open(_ASSETS_DIR / "word_rules.csv", encoding="utf-8") as words:
        return frozenset(
            row["word"] for row in csv.DictReader(words) if row["list"] == wanted
        )


YES_WORDS = _load_words("YES")
NO_WORDS = _load_words("NO")


def vocabulary(tokens):
    """The set of different words (lemmatized)."""
    return set(tokens)


def count_yes(tokens):
    """Confirmation-like tokens."""
    return sum(1 for t in tokens if t in YES_WORDS)


def count_no(tokens):
    """Negation-like tokens."""
    return sum(1 for t in tokens if t in NO_WORDS)


def speaker_stats(texts):
    """The Task-1 numbers for one pile of texts (one speaker).

    "Repetitions" = word re-uses inside the same text (one text = one post):
    a text with 30 tokens and 25 different ones repeats 5.

    Returns the vocabulary set too — the summary table uses only its size.
    """
    tokens = []
    repetitions = yes = no = 0
    for text in texts:
        pile = tokenize(text)
        tokens.extend(pile)
        repetitions += len(pile) - len(set(pile))
        yes += count_yes(pile)
        no += count_no(pile)
    return {
        "vocabulary_set": set(tokens),
        "vocabulary_size": len(set(tokens)),
        "tokens": len(tokens),
        "repetitions": repetitions,
        "yes_words": yes,
        "no_words": no,
    }


def chunked_series(texts, chunks=10):
    """Cumulative counting evolution across one pile of texts.

    The pile is split, in reading order, into `chunks` steps rounded to text
    boundaries. Returns a DataFrame with cumulative tokens, repetitions,
    yes_words and no_words at each step.
    """
    per_text = []
    for text in texts:
        tokens = tokenize(text)
        per_text.append((len(tokens), len(tokens) - len(set(tokens)),
                         count_yes(tokens), count_no(tokens)))
    rows = []
    counts = pd.DataFrame(per_text, columns=["tokens", "repetitions",
                                             "yes_words", "no_words"])
    step = (len(counts) - 1) / chunks if len(counts) > 1 else 0
    for i in range(1, chunks + 1):
        end = round(i * step) + 1
        upto = counts.iloc[:end]
        rows.append({
            "tokens": int(upto["tokens"].sum()),
            "repetitions": int(upto["repetitions"].sum()),
            "yes_words": int(upto["yes_words"].sum()),
            "no_words": int(upto["no_words"].sum()),
        })
    return pd.DataFrame(rows)