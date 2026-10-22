"""Doc2Vec + cosine similarity vs trait definition terms (Task 5)."""

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

from personality.text import tokenize

# inference effort shared by term vectors and speaker docs
INFERENCE_EPOCHS = 50


# Personality-definition terms used to build the meaning map.
# Definition terms used to represent each personality trait.
TRAIT_TERMS = {
    "openness": [
        "intellectual", "artistic", "imaginative", "refined",
    ],
    "conscientiousness": [
        "responsible", "scrupulous", "persevering", "tidy",
    ],
    "extraversion": [
        "talkative", "sociable", "adventurous", "open",
    ],
    "agreeableness": [
        "good-natured", "cooperative", "mild", "gentle",
    ],
    "neuroticism": [
        "anxious", "excitable", "nervous", "tense",
    ],
}


def _tokens(text):
    """Return tokens for a non-empty text value.

    Missing and empty values produce an empty list. Tokenization and
    lemmatization are handled by the shared personality.text module.
    """
    if text is None or pd.isna(text):
        return []

    text = str(text).strip()
    if not text:
        return []

    return tokenize(text)


def train_doc2vec(texts, vector_size=50, epochs=10, seed=42):
    """Doc2Vec on pile texts; seed fixed = reproducible training."""
    documents = []

    # Tokenize each text and assign it a unique document tag.
    for index, text in enumerate(texts):
        words = _tokens(text)
        if words:
            documents.append(
                TaggedDocument(words=words, tags=[str(index)])
            )

    # Use one worker and a fixed seed to improve reproducibility.
    model = Doc2Vec(
        vector_size=vector_size,
        min_count=1,
        workers=1,
        epochs=epochs,
        seed=seed,
    )

    # Build the vocabulary before learning document vectors.
    model.build_vocab(documents)

    model.train(
        documents,
        total_examples=model.corpus_count,
        epochs=model.epochs,
    )

    return model



def score_traits(model, trait_terms=None):
    """One inferred doc vector per definition term."""
    if trait_terms is None:
        trait_terms = TRAIT_TERMS

    trait_vectors = {}

    for trait, terms in trait_terms.items():
        term_vectors = []

        for term in terms:
            words = _tokens(term)

            if not words:
                continue

            # Represent the entire definition term as a document vector,
            # rather than averaging the model's word vectors.
            vector = model.infer_vector(words, epochs=INFERENCE_EPOCHS)
            term_vectors.append(vector)

        if term_vectors:
            trait_vectors[trait] = term_vectors

    return trait_vectors



def predict_traits(model, texts, trait_vectors):
    """Calculate trait similarity scores for each speaker."""

    rows = []

    for text in texts:
        tokens = _tokens(text)

        if not tokens:
            rows.append({trait: np.nan for trait in trait_vectors})
            continue

        vector = model.infer_vector(tokens, epochs=INFERENCE_EPOCHS)
        row = {}

        for trait, term_vectors in trait_vectors.items():
            similarities = cosine_similarity(
                vector.reshape(1, -1),
                np.asarray(term_vectors),
            )[0]

            row[trait] = float(np.max(similarities))

        rows.append(row)

    return pd.DataFrame(rows, columns=list(trait_vectors.keys()))



def predict_labels(scores):
    """The highest-scoring trait of each row."""
    return scores.idxmax(axis=1).rename("predicted_trait")
