import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 05 - Personality meaning map (Task 5)

    Each text = a point on a map of meanings; close points = similar
    texts. That map = the meaning map. Doc2Vec draws the points;
    cosine similarity = the twin-score (1 = twins, 0 = strangers).

    One document per annotator (all their texts glued); its 4 pole-terms
    per trait = the landmarks; the trait score = the biggest
    twin-score; the highest trait = the prediction. Similarities are
    not probabilities.
    """)
    return


@app.cell
def _():
    # SETUP:
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

    T5_SPEAKERS = data.load_speakers()
    T5_PROFILES = data.load_profiles()
    return (
        T5_PROFILES,
        T5_SPEAKERS,
        accuracy_score,
        buckets,
        classification_report,
        confusion_matrix,
        embeddings,
        f1_score,
        pd,
        plt,
    )


@app.cell
def _(T5_PROFILES, T5_SPEAKERS, buckets, pd):
    # Reference labels: each annotator's dominant personality trait.
    T5_ACTUAL_TRAITS = buckets.dominant_traits(T5_PROFILES)

    T5_ACTUAL = pd.Series(
        T5_ACTUAL_TRAITS,
        name="actual_trait",
        dtype="object",
    )
    T5_ACTUAL.index.name = "annotator"

    # Use every valid, non-empty speaker text.
    T5_VALID_SPEAKERS = T5_SPEAKERS.dropna(
        subset=["annotator", "english_item"]
    ).copy()

    # force plain strings (some cells arrive as gaps); trim stray spaces
    T5_VALID_SPEAKERS["annotator"] = (
        T5_VALID_SPEAKERS["annotator"].astype(str).str.strip()
    )
    T5_VALID_SPEAKERS["english_item"] = (
        T5_VALID_SPEAKERS["english_item"].astype(str).str.strip()
    )

    # drop rows with an empty annotator or empty text; renumber after
    T5_VALID_SPEAKERS = T5_VALID_SPEAKERS[
        (T5_VALID_SPEAKERS["annotator"] != "")
        & (T5_VALID_SPEAKERS["english_item"] != "")
    ].reset_index(drop=True)

    # show the row count at each step: the cut's cost
    T5_DATA_SUMMARY = pd.DataFrame({
        "measure": [
            "All speaker rows",
            "Valid non-empty text rows",
            "Annotators with reference labels",
        ],
        "count": [
            len(T5_SPEAKERS),
            len(T5_VALID_SPEAKERS),
            len(T5_ACTUAL),
        ],
    })

    T5_DATA_SUMMARY
    return T5_ACTUAL, T5_VALID_SPEAKERS


@app.cell
def _(mo):
    mo.md("""
    ## Training + scoring (cached)

    The model = trained on the pile texts + the pole terms (so the model
    knows those words). The heavy part (train + per-speaker guesses)
    runs once; later runs load:
    `data/interim/task5_speaker_scores.parquet`. Rebuilding the cache
    moves the Spearman values ~0.02. The numbers = the cache's, and
    the cache = the record.

    An honest baseline warning: a 50-dim Doc2Vec vs short term-words =
    a weak probe. The next blocks say what it actually found.
    """)
    return


@app.cell
def _(T5_VALID_SPEAKERS, embeddings, pd):
    from pathlib import Path

    T5_SCORES_CACHE = Path("data/interim/task5_speaker_scores.parquet")
    T5_COVERAGE_CACHE = Path("data/interim/task5_term_coverage.parquet")

    # same numbers every run: load the saved scores, or train once and save
    if T5_SCORES_CACHE.exists():
        T5_SPEAKER_SCORES = pd.read_parquet(T5_SCORES_CACHE)
        T5_TERM_COVERAGE = pd.read_parquet(T5_COVERAGE_CACHE)
        T5_TRAINING_SUMMARY = {"loaded": "from data/interim caches"}
    else:
        T5_DEFINITION_TERMS = [
            term
            for terms in embeddings.TRAIT_TERMS.values()
            for term in terms
        ]

        T5_TRAIN_TEXTS = (
            T5_VALID_SPEAKERS["english_item"].tolist()
            + T5_DEFINITION_TERMS
        )

        T5_MODEL = embeddings.train_doc2vec(
            T5_TRAIN_TEXTS,
            vector_size=50,
            epochs=10,
            seed=42,
        )

        T5_TRAINING_SUMMARY = {
            "valid speaker text rows": len(T5_VALID_SPEAKERS),
            "definition terms added": len(T5_DEFINITION_TERMS),
            "training documents": len(T5_TRAIN_TEXTS),
            "vocabulary size": len(T5_MODEL.wv),
            "vector size": T5_MODEL.vector_size,
            "epochs": T5_MODEL.epochs,
        }

        T5_TERM_VECTORS = embeddings.score_traits(
            T5_MODEL,
            embeddings.TRAIT_TERMS,
        )

        T5_SPEAKER_TEXTS = (
            T5_VALID_SPEAKERS
            .groupby("annotator", sort=True)["english_item"]
            .agg(" ".join)
        )

        T5_SPEAKER_SCORES = embeddings.predict_traits(
            T5_MODEL,
            T5_SPEAKER_TEXTS.tolist(),
            T5_TERM_VECTORS,
        )
        T5_SPEAKER_SCORES.index = T5_SPEAKER_TEXTS.index

        T5_COVERAGE_ROWS = []
        for _trait, _terms in embeddings.TRAIT_TERMS.items():
            for _term in _terms:
                words = embeddings._tokens(_term)
                known_words = [
                    word for word in words if word in T5_MODEL.wv
                ]
                T5_COVERAGE_ROWS.append({
                    "trait": _trait,
                    "definition_term": _term,
                    "token_count": len(words),
                    "known_token_count": len(known_words),
                    "known_tokens": ", ".join(known_words),
                    "covered": bool(known_words),
                })
        T5_TERM_COVERAGE = pd.DataFrame(T5_COVERAGE_ROWS)

        T5_SPEAKER_SCORES.to_parquet(T5_SCORES_CACHE)
        T5_TERM_COVERAGE.to_parquet(T5_COVERAGE_CACHE)

    T5_TRAINING_SUMMARY
    return T5_SPEAKER_SCORES, T5_TERM_COVERAGE


@app.cell
def _(mo):
    mo.md("""
    ## Predictions

    The model's trait per annotator, next to their real dominant trait.
    """)
    return


@app.cell
def _(T5_ACTUAL, T5_SPEAKER_SCORES, embeddings, pd):
    T5_PREDICTED = embeddings.predict_labels(T5_SPEAKER_SCORES)

    T5_RESULTS = pd.DataFrame({
        "actual_trait": T5_ACTUAL,
        "predicted_trait": T5_PREDICTED,
    })

    T5_RESULTS = T5_RESULTS.join(
        T5_SPEAKER_SCORES.add_prefix("similarity_"),
        how="left",
    )
    T5_RESULTS.index.name = "annotator"

    T5_RESULTS
    return T5_SPEAKER_SCORES, T5_RESULTS


@app.cell
def _(mo):
    mo.md("""
    ### Spearman correlation analysis

    Does each trait's similarity track their real profile score?
    Spearman compares the two, per trait:

    - openness +0.104, conscientiousness -0.170, extraversion -0.197,
      agreeableness -0.035, neuroticism +0.221

    Weak. Direction, not proof (n = 36; untested significance).
    """)
    return


@app.cell
def _(T5_PROFILES, T5_SPEAKER_SCORES, pd):
    T5_BFI_COLUMNS = {
        "openness": "Open.",
        "conscientiousness": "Cons.",
        "extraversion": "Extra.",
        "agreeableness": "Agree.",
        "neuroticism": "Neuro.",
    }

    T5_BFI = T5_PROFILES.set_index("ID")
    T5_ALIGNED_BFI = T5_BFI.reindex(T5_SPEAKER_SCORES.index)

    T5_CORRELATION_ROWS = []

    for T5_TRAIT, T5_BFI_COLUMN in T5_BFI_COLUMNS.items():
        T5_COMPARISON = pd.DataFrame({
            "similarity": T5_SPEAKER_SCORES[T5_TRAIT],
            "bfi_score": T5_ALIGNED_BFI[T5_BFI_COLUMN],
        }).dropna()

        T5_RHO = T5_COMPARISON["similarity"].corr(
            T5_COMPARISON["bfi_score"],
            method="spearman",
        )

        T5_CORRELATION_ROWS.append({
            "trait": T5_TRAIT,
            "bfi_column": T5_BFI_COLUMN,
            "n": len(T5_COMPARISON),
            "spearman_rho": T5_RHO,
        })

    T5_SPEARMAN = pd.DataFrame(T5_CORRELATION_ROWS)

    T5_SPEARMAN
    return (T5_SPEARMAN,)


@app.cell
def _(mo):
    mo.md("""
    ## Evaluation

    Predictions are compared with the dominant personality labels
    derived from the existing annotator profiles.

    - **Accuracy:** how many guesses landed right.
    - **Macro F1:** the average score across the five traits; each
      trait counts the same.
    - **Uniform baseline:** expected accuracy of 20% when choosing uniformly
      among five traits.
    - **Majority-class baseline:** accuracy obtained by always predicting
      the most common actual trait in the evaluation set.
    - **Confusion matrix:** shows the actual and predicted trait counts.

    These metrics measure agreement with the existing reference labels;
    they do not establish that the model can reliably infer personality.
    Baselines are important because a model can appear to make predictions
    while performing worse than a simple majority-class rule.
    """)
    return


@app.cell
def _(
    T5_RESULTS,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    pd,
):
    T5_TRAIT_NAMES = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    T5_EVALUATION = T5_RESULTS.dropna(
        subset=["actual_trait", "predicted_trait"]
    )

    T5_Y_TRUE = T5_EVALUATION["actual_trait"]
    T5_Y_PRED = T5_EVALUATION["predicted_trait"]

    T5_ACCURACY = accuracy_score(T5_Y_TRUE, T5_Y_PRED)
    T5_MACRO_F1 = f1_score(
        T5_Y_TRUE,
        T5_Y_PRED,
        labels=T5_TRAIT_NAMES,
        average="macro",
        zero_division=0,
    )

    T5_MAJORITY_BASELINE = (
        T5_Y_TRUE.value_counts(normalize=True).max()
        if len(T5_Y_TRUE)
        else float("nan")
    )

    T5_UNIFORM_BASELINE = 1 / len(T5_TRAIT_NAMES)

    T5_METRICS = pd.DataFrame({
        "metric": [
            "evaluated_annotators",
            "accuracy",
            "macro_f1",
            "uniform_random_baseline",
            "majority_class_baseline",
        ],
        "value": [
            len(T5_EVALUATION),
            T5_ACCURACY,
            T5_MACRO_F1,
            T5_UNIFORM_BASELINE,
            T5_MAJORITY_BASELINE,
        ],
    })

    T5_CONFUSION = pd.DataFrame(
        confusion_matrix(
            T5_Y_TRUE,
            T5_Y_PRED,
            labels=T5_TRAIT_NAMES,
        ),
        index=T5_TRAIT_NAMES,
        columns=T5_TRAIT_NAMES,
    )
    T5_CONFUSION.index.name = "actual_trait"
    T5_CONFUSION.columns.name = "predicted_trait"

    T5_REPORT = pd.DataFrame(
        classification_report(
            T5_Y_TRUE,
            T5_Y_PRED,
            labels=T5_TRAIT_NAMES,
            output_dict=True,
            zero_division=0,
        )
    ).T

    T5_METRICS
    return T5_CONFUSION, T5_METRICS, T5_REPORT, T5_TRAIT_NAMES

@app.cell
def _(mo):
    mo.md("""
    **What this says:** accuracy 11.1% (4/36) - below BOTH baselines
    (uniform 20%, majority 52.78%). Predictions concentrate on
    neuroticism (33/36 annotators in one column): the model still
    distinguishes almost nothing, but the balanced pole lists removed
    the one-trait-takes-all bias.
    """)
    return
    
@app.cell
def _(mo):
    mo.md("""
    ## Mean similarity by trait

    Table + figure: the mean similarity per trait, and how much the
    scores wiggle across annotators. Not probabilities.
    """)
    return


@app.cell
def _(T5_SPEAKER_SCORES, T5_TRAIT_NAMES, pd):
    T5_MEAN_SIMILARITY = pd.DataFrame({
        "trait": T5_TRAIT_NAMES,
        "mean_similarity": [
            T5_SPEAKER_SCORES[trait].mean()
            for trait in T5_TRAIT_NAMES
        ],
        "std_similarity": [
            T5_SPEAKER_SCORES[trait].std()
            for trait in T5_TRAIT_NAMES
        ],
    })

    T5_MEAN_SIMILARITY
    return (T5_MEAN_SIMILARITY,)


@app.cell
def _(T5_MEAN_SIMILARITY, plt):
    T5_SIM_FIG, T5_SIM_AX = plt.subplots(figsize=(9, 5))

    T5_SIM_AX.bar(
        T5_MEAN_SIMILARITY["trait"],
        T5_MEAN_SIMILARITY["mean_similarity"],
    )
    T5_SIM_AX.set_title("Mean similarity by personality trait")
    T5_SIM_AX.set_xlabel("Personality trait")
    T5_SIM_AX.set_ylabel("Mean cosine similarity")
    T5_SIM_AX.tick_params(axis="x", rotation=25)

    T5_SIM_FIG.tight_layout()

    T5_SIM_FIG
    return (T5_SIM_FIG,)


@app.cell
def _(mo):
    mo.md("""
    **What this says:** the five bars sit within a few hundredths of each
    other - the cosine scores crowd together for every trait, which is
    why the biggest-similarity choice barely separates anyone (and why
    the winner can flip with small changes in the term lists).
    """)
    return


@app.cell
def _(T5_CONFUSION, plt):
    T5_CM_FIG, T5_CM_AX = plt.subplots(figsize=(8, 6))

    T5_CM_AX.imshow(T5_CONFUSION.values)
    T5_CM_AX.set_xticks(range(len(T5_CONFUSION.columns)))
    T5_CM_AX.set_xticklabels(T5_CONFUSION.columns, rotation=35, ha="right")
    T5_CM_AX.set_yticks(range(len(T5_CONFUSION.index)))
    T5_CM_AX.set_yticklabels(T5_CONFUSION.index)

    for row in range(len(T5_CONFUSION.index)):
        for col in range(len(T5_CONFUSION.columns)):
            T5_CM_AX.text(
                col,
                row,
                T5_CONFUSION.iloc[row, col],
                ha="center",
                va="center",
            )

    T5_CM_AX.set_xlabel("Predicted trait")
    T5_CM_AX.set_ylabel("Actual trait")
    T5_CM_AX.set_title("Personality trait confusion matrix")

    T5_CM_FIG.tight_layout()

    T5_CM_FIG
    return (T5_CM_FIG,)


@app.cell
def _(mo):
    mo.md("""
    **What this says:** the predicted column = one trait for (almost)
    every annotator; the diagonal = the correct hits. Compare with the
    52.78% majority-class baseline - the model distinguishes nothing.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Definition-term coverage

    Vocabulary check: does the model know at least one word of each
    pole-term? A model that knows nobody's words cannot guess anybody's
    trait. Coverage ≠ quality: knowing the words says nothing about the
    guesses' sense.
    """)
    return


@app.cell
def _(T5_TERM_COVERAGE):
    T5_COVERAGE_SUMMARY = (
        T5_TERM_COVERAGE.groupby("trait", as_index=False)
        .agg(
            total_terms=("definition_term", "count"),
            covered_terms=("covered", "sum"),
        )
    )

    T5_COVERAGE_SUMMARY["coverage_percent"] = (
        100
        * T5_COVERAGE_SUMMARY["covered_terms"]
        / T5_COVERAGE_SUMMARY["total_terms"]
    )

    T5_TERM_COVERAGE
    return T5_COVERAGE_SUMMARY, T5_TERM_COVERAGE


@app.cell
def _(mo):
    mo.md("""
    **What this says:** 20/20 pole-terms = known. The failure = the
    scoring, not the vocabulary.
    """)
    return


@app.cell
def _(
    Path,
    T5_CM_FIG,
    T5_CONFUSION,
    T5_COVERAGE_SUMMARY,
    T5_MEAN_SIMILARITY,
    T5_METRICS,
    T5_REPORT,
    T5_RESULTS,
    T5_SIM_FIG,
    T5_SPEAKER_SCORES,
    T5_SPEARMAN,
    T5_TERM_COVERAGE,
):
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    T5_RESULTS.to_csv("outputs/tables/task5_predictions.csv")
    T5_METRICS.to_csv("outputs/tables/task5_metrics.csv", index=False)
    T5_CONFUSION.to_csv("outputs/tables/task5_confusion_matrix.csv")
    T5_REPORT.to_csv("outputs/tables/task5_classification_report.csv")
    T5_MEAN_SIMILARITY.to_csv(
        "outputs/tables/task5_mean_similarity.csv", index=False
    )
    T5_SPEAKER_SCORES.to_csv(
        "outputs/tables/task5_speaker_similarity.csv"
    )
    T5_SPEARMAN.to_csv(
    "outputs/tables/task5_spearman_correlations.csv",
    index=False,
    )
    T5_TERM_COVERAGE.to_csv(
        "outputs/tables/task5_term_coverage.csv", index=False
    )
    T5_COVERAGE_SUMMARY.to_csv(
        "outputs/tables/task5_term_coverage_summary.csv", index=False
    )

    T5_SIM_FIG.savefig(
        "outputs/figures/task5_mean_similarity.png",
        dpi=150,
        bbox_inches="tight",
    )
    T5_CM_FIG.savefig(
        "outputs/figures/task5_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    "Task 5 outputs saved."
    return



@app.cell
def _(mo):
    mo.md("""
    ## Task 5 comment

    **Setup.** The meaning map, one document per annotator, scored
    against 4 pole-terms per trait; the biggest twin-score = the trait.
    Model: a 50-dim Doc2Vec on our own event texts.

    **Scores.** accuracy 11.1% (4/36); macro F1 0.076. Baselines:
    uniform 20%, majority 52.8%. Spearman vs their real profiles: max
    +0.22 (neuroticism).

    **What this says.**

    - The prediction = almost one trait for everyone (neuroticism,
      33/36). The balanced 4-pole lists removed the
      one-trait-takes-all bias, but the crowding remains.
    - The similarity bars = crowded (all within a few hundredths):
      Doc2Vec's guesses sit near the pile-average; the biggest-similarity
      wins barely move.
    - The Spearman story = weak in every direction. Nothing here
      tracks their real profiles.

    **Verdict.** Meaning map fails as configured: an honest negative.
    The paper's stronger setup (big models, 22-31% first-guesses) says
    personality effects in language = small; a Doc2Vec probe this thin
    never had a chance, and the vocab is complete (20/20). The loss =
    the probe, not the words.

    **Weak spots.** The texts = the events' words (the Rule A framing).
    n = 36. The term lists' balance = itself a modeling choice. Full
    detail lives in the blocks above; the report carries the
    limitations.
    """)
    return



if __name__ == "__main__":
    app.run()
