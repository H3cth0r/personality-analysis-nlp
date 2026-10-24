"""Word counting (Tasks 1 and 4).

The word lists live in assets/word_rules.csv (11 yes-words, 27 no-words).
They follow the LIWC "assent" and "negate" lists (Pennebaker et al.),
trimmed to clear words only — "right", "true", "fine" are too ambiguous
to count.

Tokenizing and lemmatizing live in text.py (shared by all tasks).
"""
import csv
from pathlib import Path

import pandas as pd

from personality.text import tokenize

_ASSETS_DIR = Path(__file__).parent / "assets"


def _load_words(wanted):
    """Read one word list from assets/word_rules.csv."""
    with open(_ASSETS_DIR / "word_rules.csv", encoding="utf-8") as words:
        return frozenset(
            row["word"] for row in csv.DictReader(words) if row["list"] == wanted
        )


YES_WORDS = _load_words("YES")
NO_WORDS = _load_words("NO")


def vocabulary(tokens):
    """All different words."""
    return set(tokens)


def count_yes(tokens):
    """How many yes-words."""
    return sum(1 for t in tokens if t in YES_WORDS)


def count_no(tokens):
    """How many no-words."""
    return sum(1 for t in tokens if t in NO_WORDS)


def speaker_stats(texts):
    """All Task-1 counts for one speaker's pile of texts.

    A "repetition" = a word used again inside the same text:
    30 words with 25 different ones = 5 repetitions.
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
        "vocabulary_set": set(tokens),   # the raw word set, for reuse; never shown
        "vocabulary_size": len(set(tokens)),
        "tokens": len(tokens),
        "repetitions": repetitions,
        "yes_words": yes,
        "no_words": no,
    }


def chunked_series(texts, chunks=10):
    """Counts as they grow through one pile, in reading order.

    The pile is split into `chunks` steps (rounded to text limits).
    Returns a DataFrame: cumulative tokens, repetitions, yes_words,
    no_words at each step.
    """
    per_text = []
    for text in texts:
        tokens = tokenize(text)
        per_text.append((len(tokens), len(tokens) - len(set(tokens)),
                         count_yes(tokens), count_no(tokens)))
    counts = pd.DataFrame(per_text, columns=["tokens", "repetitions",
                                             "yes_words", "no_words"])
    step = (len(counts) - 1) / chunks if len(counts) > 1 else 0
    rows = []
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


def vocab_growth(texts, chunks=10):
    """Vocabulary size as tokens accumulate, in reading order.

    Same stepping as chunked_series. Returns a DataFrame: tokens so far
    vs different words so far — a "how many new words by now?" curve.
    """
    rows = []
    seen = set()
    tokens = 0
    for text in texts:
        pile = tokenize(text)
        tokens += len(pile)
        seen.update(pile)
        rows.append({"tokens": tokens, "vocab_size": len(seen)})
    growth = pd.DataFrame(rows)
    if growth.empty:
        return growth
    step = (len(growth) - 1) / chunks
    marks = pd.DataFrame([growth.iloc[round(i * step)].to_dict()
                          for i in range(1, chunks + 1)])
    return marks