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