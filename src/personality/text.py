"""Text normalization shared by every task. The one tokenizer.

Rules (frozen, Step 0.6): lowercase; letters + apostrophes stay together
(contractions remain one token: "don't"); other punctuation dropped; no
stopword removal. Every token passes through the WordNet lemmatizer
(verb lemma first, then noun), so "agreed" and "agree" count as the
same word.
"""
import re

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_WORD_RE = re.compile(r"[a-z']+")
_LEMMATIZER = WordNetLemmatizer()


def lemmatize(word):
    """Verb lemma first, then noun ("agreed" -> "agree", "prices" -> "price")."""
    return _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(word, "v"), "n")


def tokenize(text):
    """The one tokenizer. Lowercase, word tokens, lemmatized."""
    return [lemmatize(t) for t in _WORD_RE.findall(text.lower())]