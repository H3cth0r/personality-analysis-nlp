
"""Doc2Vec embeddings and personality-trait similarity scoring (Task 5).

This module trains a Doc2Vec model on the available text documents,
creates vectors for personality-definition terms, and scores texts by
their cosine similarity to those terms.

Each trait's score is the maximum cosine similarity across its available
definition-term vectors. The highest-scoring trait can then be used as
the predicted personality label.

The scores are exploratory similarity measures, not probabilities or
direct measurements of personality.
"""

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

from personality.text import tokenize


# Personality-definition terms used to build the meaning map.
# Positive and contrasting descriptors are included for each trait.
TRAIT_TERMS = {
    "openness": [
        "intellectual", "unreflective", "narrow", "artistic",
        "nonartistic", "imaginative", "simple", "direct",
        "refined", "crude", "boorish",
    ],
    "conscientiousness": [
        "responsible", "undependable", "scrupulous", "unscrupulous",
        "persevering", "quitting", "fussy", "tidy", "careless",
    ],
    "extraversion": [
        "talkative", "silent", "sociable", "reclusive",
        "adventurous", "cautious", "open", "secretive",
    ],
    "agreeableness": [
        "good-natured", "irritable", "cooperative", "negativistic",
        "mild", "gentle", "headstrong", "not jealous", "jealous",
    ],
    "neuroticism": [
        "calm", "anxious", "composed", "excitable",
        "not hypochondriacal", "hypochondriacal",
        "poised", "nervous", "tense",
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
    """Train a Doc2Vec model on a collection of text documents.

    Each non-empty text is stored as a tagged document. The model learns
    document representations from the training corpus.

    Args:
        texts: Text documents used to train the model.
        vector_size: Number of dimensions in each vector.
        epochs: Number of training epochs.
        seed: Random seed for reproducibility.

    Returns:
        A trained gensim Doc2Vec model.

    Raises:
        ValueError: If no non-empty documents are available.
    """
    documents = []

    # Tokenize each text and assign it a unique document tag.
    for index, text in enumerate(texts):
        words = _tokens(text)
        if words:
            documents.append(
                TaggedDocument(words=words, tags=[str(index)])
            )

    if not documents:
        raise ValueError(
            "No non-empty documents available for Doc2Vec training."
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
    """Infer a document vector for each personality-definition term.

    Each definition term is tokenized and represented by a Doc2Vec
    inferred vector. These vectors are later compared with text vectors
    using cosine similarity.

    Terms with no tokens are skipped. Traits without usable terms are
    omitted from the returned dictionary.

    Args:
        model: A trained gensim Doc2Vec model.
        trait_terms: Mapping from trait names to definition terms.
            Defaults to TRAIT_TERMS.

    Returns:
        Dictionary mapping each trait to a list of definition-term
        document vectors.
    """
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
            vector = model.infer_vector(words, epochs=50)

            if np.all(np.isfinite(vector)):
                term_vectors.append(vector)

        if term_vectors:
            trait_vectors[trait] = term_vectors

    if not trait_vectors:
        raise ValueError(
            "No personality-definition terms could be embedded."
        )

    return trait_vectors



def predict_traits(model, texts, trait_vectors):
    """Calculate personality-trait similarity scores for each text.

    A document vector is inferred for each non-empty text. Its cosine
    similarity is calculated against every available definition-term
    vector. Each trait receives its maximum similarity across its terms.

    Empty texts receive missing scores.

    Args:
        model: A trained Doc2Vec model.
        texts: Text documents to score.
        trait_vectors: Definition-term vectors returned by score_traits.

    Returns:
        A DataFrame containing one similarity-score column per trait
        and one row per input text.
    """
    if not trait_vectors:
        raise ValueError("trait_vectors is empty; score traits first.")

    rows = []

    for text in texts:
        tokens = _tokens(text)

        if not tokens:
            rows.append({
                trait: np.nan for trait in trait_vectors
            })
            continue

        # Infer a vector representing this individual text.
        vector = model.infer_vector(tokens)
        row = {}

        for trait, term_vectors in trait_vectors.items():
            # Compare the text vector with every available term vector.
            similarities = cosine_similarity(
                vector.reshape(1, -1),
                np.asarray(term_vectors),
            )[0]

            # Ignore invalid similarities before taking the maximum.
            valid_similarities = similarities[
                np.isfinite(similarities)
            ]

            row[trait] = (
                float(np.max(valid_similarities))
                if len(valid_similarities)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows, columns=list(trait_vectors.keys()))


def predict_labels(scores):
    """Assign the highest-scoring trait to each row.

    Rows containing only missing scores remain unclassified.

    Args:
        scores: DataFrame of text or annotator similarity scores.

    Returns:
        A Series named 'predicted_trait' containing the highest-scoring
        trait for each row, or a missing value if no valid score exists.
    """
    if scores.empty or scores.shape[1] == 0:
        return pd.Series(index=scores.index, dtype="object")

    # Prevent infinite values from affecting the selected label.
    valid_scores = scores.replace(
        [np.inf, -np.inf], np.nan
    )

    # Select the trait with the maximum score in each row.
    predictions = valid_scores.idxmax(axis=1)

    # idxmax cannot meaningfully classify a row with no valid scores.
    predictions = predictions.where(
        valid_scores.notna().any(axis=1)
    )

    return predictions.rename("predicted_trait")
