
import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Task 5 — Personality meaning map

    This task uses **Doc2Vec** to represent text as numerical vectors
    and cosine similarity to compare texts with personality-definition
    terms.

    Each text receives a similarity score for each of the five Big Five
    traits. A trait's score is the maximum similarity across its
    available definition terms. The highest-scoring trait becomes the
    predicted label.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    from personality import buckets, data, embeddings

    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    SPEAKERS = data.load_speakers()
    PROFILES = data.load_profiles()

    TRAIT_NAMES = list(buckets.TRAIT_NAMES)

    return (
        plt,
        np,
        pd,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        buckets,
        data,
        embeddings,
        SPEAKERS,
        PROFILES,
        TRAIT_NAMES,
    )


@app.cell
def _(SPEAKERS, PROFILES, buckets, pd):
    # Reference labels: each annotator's dominant personality trait.
    ACTUAL_TRAITS = buckets.dominant_traits(PROFILES)

    ACTUAL = pd.Series(
        ACTUAL_TRAITS,
        name="actual_trait",
        dtype="object",
    )
    ACTUAL.index.name = "annotator"

    # Use every valid, non-empty speaker text.
    VALID_SPEAKERS = SPEAKERS.dropna(
        subset=["annotator", "english_item"]
    ).copy()

    VALID_SPEAKERS["annotator"] = (
        VALID_SPEAKERS["annotator"].astype(str).str.strip()
    )
    VALID_SPEAKERS["english_item"] = (
        VALID_SPEAKERS["english_item"].astype(str).str.strip()
    )

    VALID_SPEAKERS = VALID_SPEAKERS[
        (VALID_SPEAKERS["annotator"] != "")
        & (VALID_SPEAKERS["english_item"] != "")
    ].reset_index(drop=True)

    DATA_SUMMARY = pd.DataFrame({
        "measure": [
            "All speaker rows",
            "Valid non-empty text rows",
            "Annotators with reference labels",
        ],
        "count": [
            len(SPEAKERS),
            len(VALID_SPEAKERS),
            len(ACTUAL),
        ],
    })

    DATA_SUMMARY
    return ACTUAL, VALID_SPEAKERS, DATA_SUMMARY


@app.cell
def _(mo):
    mo.md("""
    ## Training the Doc2Vec model

    The model is trained on all valid speaker texts. Personality
    definition terms are also included in the training corpus so their
    words are available for similarity scoring.

    The model uses 50-dimensional vectors, 10 training epochs, and a
    fixed random seed. The analysis is exploratory: word-vector
    similarities may not reliably represent the meaning of short
    personality-definition terms.
    """)
    return


@app.cell
def _(VALID_SPEAKERS, embeddings):
    DEFINITION_TERMS = [
        term
        for terms in embeddings.TRAIT_TERMS.values()
        for term in terms
    ]

    TRAIN_TEXTS = (
        VALID_SPEAKERS["english_item"].tolist()
        + DEFINITION_TERMS
    )

    MODEL = embeddings.train_doc2vec(
        TRAIN_TEXTS,
        vector_size=50,
        epochs=10,
        seed=42,
    )

    TRAINING_SUMMARY = {
        "valid speaker text rows": len(VALID_SPEAKERS),
        "definition terms added": len(DEFINITION_TERMS),
        "training documents": len(TRAIN_TEXTS),
        "vocabulary size": len(MODEL.wv),
        "vector size": MODEL.vector_size,
        "epochs": MODEL.epochs,
    }

    TRAINING_SUMMARY
    return MODEL, DEFINITION_TERMS


@app.cell
def _(mo):
    mo.md("""
    ## Similarity scores and predictions

    Each valid text is represented by an inferred document vector.
    Cosine similarity compares that vector with the available
    definition-term vectors for each trait.

    The maximum similarity is used as the trait score. Scores are
    similarity measures, not probabilities. Text-level scores are
    averaged by annotator before assigning the predicted trait.
    """)
    return


@app.cell
def _(MODEL, VALID_SPEAKERS, embeddings, pd):
    TRAIT_VECTORS = embeddings.score_traits(
        MODEL,
        embeddings.TRAIT_TERMS,
    )

    TEXT_SCORES = embeddings.predict_traits(
        MODEL,
        VALID_SPEAKERS["english_item"].tolist(),
        TRAIT_VECTORS,
    )

    TEXT_SCORES["annotator"] = (
        VALID_SPEAKERS["annotator"].to_numpy()
    )

    SPEAKER_SCORES = (
        TEXT_SCORES
        .groupby("annotator")[list(TRAIT_VECTORS.keys())]
        .mean()
    )

    PREDICTED = embeddings.predict_labels(SPEAKER_SCORES)

    RESULTS = pd.DataFrame({
        "actual_trait": ACTUAL,
        "predicted_trait": PREDICTED,
    })

    RESULTS = RESULTS.join(
        SPEAKER_SCORES.add_prefix("similarity_"),
        how="left",
    )
    RESULTS.index.name = "annotator"

    RESULTS
    return TRAIT_VECTORS, TEXT_SCORES, SPEAKER_SCORES, PREDICTED, RESULTS


@app.cell
def _(mo):
    mo.md("""
    ## Evaluation

    Predictions are compared with the dominant personality labels
    derived from the existing annotator profiles.

    - **Accuracy:** proportion of evaluated annotators classified correctly.
    - **Macro F1:** average F1 score across the five traits.
    - **Confusion matrix:** shows correct classifications and
      confusion between traits.

    These measures describe agreement with the existing labels; they
    do not establish that the model can reliably infer personality.
    """)
    return


@app.cell
def _(
    RESULTS,
    TRAIT_NAMES,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    pd,
):
    EVALUATION = RESULTS.dropna(
        subset=["actual_trait", "predicted_trait"]
    )

    Y_TRUE = EVALUATION["actual_trait"]
    Y_PRED = EVALUATION["predicted_trait"]

    if len(EVALUATION):
        ACCURACY = accuracy_score(Y_TRUE, Y_PRED)
        MACRO_F1 = f1_score(
            Y_TRUE,
            Y_PRED,
            labels=TRAIT_NAMES,
            average="macro",
            zero_division=0,
        )
    else:
        ACCURACY = float("nan")
        MACRO_F1 = float("nan")

    METRICS = pd.DataFrame({
        "metric": [
            "evaluated_annotators",
            "accuracy",
            "macro_f1",
        ],
        "value": [
            len(EVALUATION),
            ACCURACY,
            MACRO_F1,
        ],
    })

    CONFUSION = pd.DataFrame(
        confusion_matrix(Y_TRUE, Y_PRED, labels=TRAIT_NAMES),
        index=TRAIT_NAMES,
        columns=TRAIT_NAMES,
    )
    CONFUSION.index.name = "actual_trait"
    CONFUSION.columns.name = "predicted_trait"

    REPORT = pd.DataFrame(
        classification_report(
            Y_TRUE,
            Y_PRED,
            labels=TRAIT_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T

    METRICS
    return METRICS, CONFUSION, REPORT, EVALUATION


@app.cell
def _(mo):
    mo.md("""
    ## Mean similarity by trait

    The following table and figure summarize the average similarity
    score for each personality trait across annotators.

    Higher values indicate greater similarity to at least one
    available definition term under this model. They should not be
    interpreted as personality probabilities.
    """)
    return


@app.cell
def _(SPEAKER_SCORES, TRAIT_NAMES, pd):
    MEAN_SIMILARITY = pd.DataFrame({
        "trait": TRAIT_NAMES,
        "mean_similarity": [
            SPEAKER_SCORES[trait].mean()
            if trait in SPEAKER_SCORES.columns
            else float("nan")
            for trait in TRAIT_NAMES
        ],
        "std_similarity": [
            SPEAKER_SCORES[trait].std()
            if trait in SPEAKER_SCORES.columns
            else float("nan")
            for trait in TRAIT_NAMES
        ],
    })

    MEAN_SIMILARITY
    return (MEAN_SIMILARITY,)


@app.cell
def _(MEAN_SIMILARITY, plt):
    SIM_FIG, SIM_AX = plt.subplots(figsize=(9, 5))

    SIM_AX.bar(
        MEAN_SIMILARITY["trait"],
        MEAN_SIMILARITY["mean_similarity"],
    )
    SIM_AX.set_title("Mean similarity by personality trait")
    SIM_AX.set_xlabel("personality trait")
    SIM_AX.set_ylabel("mean cosine similarity")
    SIM_AX.tick_params(axis="x", rotation=25)

    SIM_FIG.tight_layout()
    SIM_FIG
    return (SIM_FIG,)


@app.cell
def _(CONFUSION, plt):
    CM_FIG, CM_AX = plt.subplots(figsize=(8, 6))

    IMAGE = CM_AX.imshow(CONFUSION.to_numpy())

    CM_AX.set_xticks(range(len(CONFUSION.columns)))
    CM_AX.set_xticklabels(
        CONFUSION.columns,
        rotation=35,
        ha="right",
    )
    CM_AX.set_yticks(range(len(CONFUSION.index)))
    CM_AX.set_yticklabels(CONFUSION.index)

    CM_AX.set_xlabel("predicted trait")
    CM_AX.set_ylabel("actual trait")
    CM_AX.set_title("Task 5 confusion matrix")

    for row in range(CONFUSION.shape[0]):
        for col in range(CONFUSION.shape[1]):
            CM_AX.text(
                col,
                row,
                str(CONFUSION.iloc[row, col]),
                ha="center",
                va="center",
            )

    CM_FIG.colorbar(IMAGE, ax=CM_AX, label="annotator count")
    CM_FIG.tight_layout()
    CM_FIG
    return (CM_FIG,)


@app.cell
def _(mo):
    mo.md("""
    ## Definition-term coverage

    This table checks whether each personality-definition term has
    at least one token represented in the model vocabulary. Terms with
    no known tokens cannot contribute to similarity scoring.

    Vocabulary coverage is not proof of semantic quality, but it helps
    identify missing terms that could affect the comparison.
    """)
    return


@app.cell
def _(MODEL, embeddings, pd):
    COVERAGE_ROWS = []

    for trait, terms in embeddings.TRAIT_TERMS.items():
        for term in terms:
            words = embeddings._tokens(term)
            known_words = [
                word for word in words if word in MODEL.wv
            ]

            COVERAGE_ROWS.append({
                "trait": trait,
                "definition_term": term,
                "token_count": len(words),
                "known_token_count": len(known_words),
                "known_tokens": ", ".join(known_words),
                "covered": bool(known_words),
            })

    TERM_COVERAGE = pd.DataFrame(COVERAGE_ROWS)

    COVERAGE_SUMMARY = (
        TERM_COVERAGE.groupby("trait", as_index=False)
        .agg(
            total_terms=("definition_term", "count"),
            covered_terms=("covered", "sum"),
        )
    )
    COVERAGE_SUMMARY["coverage_percent"] = (
        100 * COVERAGE_SUMMARY["covered_terms"]
        / COVERAGE_SUMMARY["total_terms"]
    )

    TERM_COVERAGE
    return TERM_COVERAGE, COVERAGE_SUMMARY


@app.cell
def _(
    RESULTS,
    TEXT_SCORES,
    METRICS,
    CONFUSION,
    REPORT,
    MEAN_SIMILARITY,
    TERM_COVERAGE,
    COVERAGE_SUMMARY,
    SIM_FIG,
    CM_FIG,
):
    # Save tables and figures for the project report.
    RESULTS.to_csv("outputs/tables/task5_predictions.csv")
    TEXT_SCORES.to_csv(
        "outputs/tables/task5_similarity_scores.csv",
        index=False,
    )
    METRICS.to_csv(
        "outputs/tables/task5_metrics.csv",
        index=False,
    )
    CONFUSION.to_csv("outputs/tables/task5_confusion_matrix.csv")
    REPORT.to_csv("outputs/tables/task5_classification_report.csv")
    MEAN_SIMILARITY.to_csv(
        "outputs/tables/task5_mean_similarity.csv",
        index=False,
    )
    TERM_COVERAGE.to_csv(
        "outputs/tables/task5_term_coverage.csv",
        index=False,
    )
    COVERAGE_SUMMARY.to_csv(
        "outputs/tables/task5_term_coverage_summary.csv",
        index=False,
    )

    SIM_FIG.savefig(
        "outputs/figures/task5_mean_similarity.png",
        dpi=150,
        bbox_inches="tight",
    )
    CM_FIG.savefig(
        "outputs/figures/task5_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    "Task 5 tables and figures saved."
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task 5 comment

    The meaning map explores whether the event texts are more similar
    to definitions of particular personality traits. The evaluation
    metrics and confusion matrix show how well these predictions agree
    with the existing dominant-trait labels.

    The results should be interpreted cautiously. The texts describe
    reactions to events rather than necessarily being authored by the
    annotators. Doc2Vec similarities may also be noisy, particularly
    for short definition terms. With only 36 annotators, this analysis
    is exploratory rather than conclusive.

    ## Outputs

    **Tables — `outputs/tables/`**

    - `task5_predictions.csv`
    - `task5_similarity_scores.csv`
    - `task5_metrics.csv`
    - `task5_confusion_matrix.csv`
    - `task5_classification_report.csv`
    - `task5_mean_similarity.csv`
    - `task5_term_coverage.csv`
    - `task5_term_coverage_summary.csv`

    **Figures — `outputs/figures/`**

    - `task5_mean_similarity.png`
    - `task5_confusion_matrix.png`
    """)
    return


if __name__ == "__main__":
    app.run()
