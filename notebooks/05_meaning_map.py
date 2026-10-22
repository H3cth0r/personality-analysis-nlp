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

    This task explores whether event-related texts are more similar to
    definition terms associated with the five Big Five personality traits.

    A Doc2Vec model represents text and definition terms as numerical
    vectors. Cosine similarity is used to score each trait, taking the
    maximum similarity across that trait's definition terms.

    Each annotator's valid texts are combined into one document, producing
    one set of trait similarity scores and one predicted trait per
    annotator. These scores are similarities, not probabilities or
    validated measurements of personality.
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

    T5_VALID_SPEAKERS["annotator"] = (
        T5_VALID_SPEAKERS["annotator"].astype(str).str.strip()
    )
    T5_VALID_SPEAKERS["english_item"] = (
        T5_VALID_SPEAKERS["english_item"].astype(str).str.strip()
    )

    T5_VALID_SPEAKERS = T5_VALID_SPEAKERS[
        (T5_VALID_SPEAKERS["annotator"] != "")
        & (T5_VALID_SPEAKERS["english_item"] != "")
    ].reset_index(drop=True)

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
    ## Training the Doc2Vec model

    The model is trained on the valid speaker texts, with the personality
    definition terms added to the training corpus so their words can be
    represented in the embedding space.

    The model uses 50-dimensional vectors, 10 training epochs, and a
    fixed training seed. Document-vector inference may still vary unless
    its randomness is controlled separately.

    This is an exploratory baseline. Doc2Vec similarity to short
    definition terms does not necessarily capture the intended
    personality meaning.
    """)
    return


@app.cell
def _(T5_VALID_SPEAKERS, embeddings):
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

    T5_TRAINING_SUMMARY
    return (T5_MODEL,)


@app.cell
def _(mo):
    mo.md("""
    ## Similarity scores and predictions

    Valid texts are grouped by annotator and combined into one document
    per annotator. The model infers one document vector for each combined
    text and compares it with the definition-term vectors for each trait.

    Each trait score is the maximum cosine similarity across its
    definition terms. The highest-scoring trait is assigned as the
    predicted label.

    Similarity scores are not probabilities. Because multiple texts are
    combined before inference, each annotator contributes one score per
    trait rather than an average of separate text-level scores.
    """)
    return


@app.cell
def _(T5_ACTUAL, T5_MODEL, T5_VALID_SPEAKERS, embeddings, pd):
    T5_TRAIT_VECTORS = embeddings.score_traits(
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
        T5_TRAIT_VECTORS,
    )

    T5_SPEAKER_SCORES.index = T5_SPEAKER_TEXTS.index

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
    return T5_RESULTS, T5_SPEAKER_SCORES


@app.cell
def _():
    return



@app.cell
def _(mo):
    mo.md("""
    ### Spearman correlation analysis

    Spearman correlations were calculated between each personality trait's
    similarity scores and the corresponding raw BFI scores for 36 annotators.

    The correlations were weak across all five traits, ranging from
    approximately -0.168 to 0.194. Neuroticism showed the strongest positive
    association (ρ = 0.194), while extraversion showed the strongest negative
    association (ρ = -0.168). Agreeableness showed almost no association
    (ρ = 0.029).

    Overall, the results suggest limited correspondence between the
    similarity scores and the corresponding BFI measurements. These are
    descriptive correlations; statistical significance has not been tested.
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

    - **Accuracy:** proportion of evaluated annotators classified correctly.
    - **Macro F1:** unweighted average of the F1 scores across the five traits.
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
    **What this says:** The model achieved 8.33% accuracy, below both
    the 20% uniform baseline and the 52.78% majority-class baseline.
    This indicates poor classification performance on the 36 annotators
    evaluated.
    """)
    return
    
@app.cell
def _(mo):
    mo.md("""
    ## Mean similarity by trait

    The table and figure summarize the mean and standard deviation of
    each trait's similarity scores across annotators.

    Higher similarity means that a speaker's combined text is closer
    to at least one definition-term vector for that trait in this model.
    Similarity values are not probabilities, and a higher value alone
    does not show that the annotator has that personality trait.
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
    **What this says**

    The diagonal shows correct predictions; off-diagonal cells show
    disagreements between the reference and predicted traits. Concentration
    in a single predicted column indicates that the model tends to assign
    the same trait to many annotators rather than distinguish the five traits.
    Interpret the matrix alongside the accuracy and majority-class baseline.
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
    ## Definition-term coverage

    This table checks whether each personality-definition term has at
    least one token represented in the Doc2Vec vocabulary. A term with
    no known tokens cannot contribute a term vector to similarity scoring.

    Coverage is a vocabulary check, not a measure of semantic quality.
    A term may be present in the vocabulary without its vector accurately
    representing the intended personality meaning.
    """)
    return


@app.cell
def _(T5_MODEL, embeddings, pd):
    T5_COVERAGE_ROWS = []

    for trait, terms in embeddings.TRAIT_TERMS.items():
        for term in terms:
            words = embeddings._tokens(term)
            known_words = [
                word for word in words if word in T5_MODEL.wv
            ]

            T5_COVERAGE_ROWS.append({
                "trait": trait,
                "definition_term": term,
                "token_count": len(words),
                "known_token_count": len(known_words),
                "known_tokens": ", ".join(known_words),
                "covered": bool(known_words),
            })

    T5_TERM_COVERAGE = pd.DataFrame(T5_COVERAGE_ROWS)

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
    **What this says**

    The coverage summary identifies definition terms that may be missing
    from the learned vocabulary. Low coverage can limit trait scoring,
    while high coverage alone does not establish that the embeddings
    represent the definitions correctly.
    """)
    return


@app.cell
def _(
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
    from pathlib import Path

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

    This task explores whether Doc2Vec similarity to personality-definition
    terms can distinguish annotators' dominant personality labels from
    their event-related texts.

    Each annotator is represented by one combined text document. The
    predicted trait is the trait with the highest maximum cosine similarity
    to its definition-term vectors.

    ### Evaluation results

    The model was evaluated on 36 annotators and achieved an accuracy of
    8.33% and a macro F1 score of approximately 0.042. Accuracy was below
    both the uniform five-trait baseline of 20% and the majority-class
    baseline of 52.78%.

    Predictions were heavily concentrated in neuroticism: 35 of the 36
    annotators were assigned this label. The confusion matrix therefore
    indicates poor discrimination between the five reference labels.

    ### Spearman correlations

    Spearman correlations were calculated between each trait's similarity
    scores and its corresponding raw BFI score for all 36 annotators.
    The correlations were weak, ranging from approximately -0.168 to
    0.194. Neuroticism had the strongest positive association (ρ = 0.194),
    while extraversion had the strongest negative association (ρ = -0.168).
    Agreeableness showed almost no association (ρ = 0.029).

    These are descriptive correlations; statistical significance was not
    tested. They provide little evidence that the similarity scores closely
    track the corresponding BFI measurements.

    ### Definition-term coverage

    All 21 definition terms had at least one token represented in the
    Doc2Vec vocabulary. This confirms vocabulary coverage, but does not
    establish that the learned vectors accurately represent the intended
    personality meanings.

    ### Limitations

    The results indicate weak performance in this exploratory evaluation.
    Similarity scores are not probabilities or validated personality
    measurements. The findings do not establish that event-related text
    can reliably identify personality. Results may be affected by the
    definition terms, the limited training corpus, and the properties
    of Doc2Vec embeddings.

    ## Outputs

    **Tables — `outputs/tables/`**

    - `task5_predictions.csv`
    - `task5_metrics.csv`
    - `task5_confusion_matrix.csv`
    - `task5_classification_report.csv`
    - `task5_mean_similarity.csv`
    - `task5_speaker_similarity.csv`
    - `task5_spearman_correlations.csv`
    - `task5_term_coverage.csv`
    - `task5_term_coverage_summary.csv`

    **Figures — `outputs/figures/`**

    - `task5_mean_similarity.png`
    - `task5_confusion_matrix.png`
    """)
    return



if __name__ == "__main__":
    app.run()
