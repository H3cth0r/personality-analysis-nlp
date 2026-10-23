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
    # 06 - AI explanations (Task 6)

    We give the AI measured numbers (similarity scores, personality
    profile values and word statistics) and ask it to explain the choice.
    It must not guess from text alone.

    We select one case for each actual dominant trait, preferring wrong
    predictions because they show where the method can mislead. These
    are diagnostic examples chosen to show errors, not a fair sample.

    The texts describe events that annotators reacted to; they did not
    write the event texts themselves.
    """)
    return


@app.cell
def _():
    # SETUP:
    import hashlib
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import pandas as pd

    from personality import buckets, data
    from personality.llm import explain_trait, PROMPT_VERSION

    T6_PROFILES = data.load_profiles()
    T6_SPEAKERS = data.load_speakers()

    T6_TASK1 = pd.read_csv(
        "outputs/tables/task1_speaker_stats.csv",
        index_col=0,
    )
    T6_TASK3 = pd.read_csv(
        "outputs/tables/task3_speaker_vad.csv",
        index_col=0,
    )
    T6_TASK4 = pd.read_csv(
        "outputs/tables/task4_bucket_stats.csv",
        index_col=0,
    )
    T6_PREDICTIONS = pd.read_csv(
        "outputs/tables/task5_predictions.csv"
    )
    T6_SIMILARITY = pd.read_csv(
        "outputs/tables/task5_speaker_similarity.csv",
        index_col=0,
    )

    return (
        Path,
        buckets,
        data,
        explain_trait,
        hashlib,
        json,
        pd,
        plt,
        PROMPT_VERSION,
        T6_PROFILES,
        T6_SPEAKERS,
        T6_TASK1,
        T6_TASK3,
        T6_TASK4,
        T6_PREDICTIONS,
        T6_SIMILARITY,
    )


@app.cell
def _(
    T6_PROFILES,
    T6_TASK1,
    T6_TASK3,
    T6_TASK4,
    T6_PREDICTIONS,
    T6_SIMILARITY,
    buckets,
    pd,
):
    # Join the existing measurements by anonymized annotator ID.
    # Normalize the loaded table indexes before joining them.
    for T6_FRAME in (
        T6_TASK1,
        T6_TASK3,
        T6_TASK4,
        T6_SIMILARITY,
    ):
        T6_FRAME.index = T6_FRAME.index.astype(str).str.strip()

    T6_PREDICTION_DATA = T6_PREDICTIONS.set_index("annotator")
    T6_PREDICTION_DATA.index = (
        T6_PREDICTION_DATA.index.astype(str).str.strip()
    )

    T6_TRAITS = list(buckets.TRAIT_NAMES)
    T6_SIM_COLUMNS = [f"similarity_{trait}" for trait in T6_TRAITS]

    T6_ACTUAL = pd.Series(
        buckets.dominant_traits(T6_PROFILES),
        name="actual_trait",
    )
    T6_ACTUAL.index = T6_ACTUAL.index.astype(str).str.strip()

    T6_PROFILE_RENAME = {
        "Open.": "profile_openness",
        "Cons.": "profile_conscientiousness",
        "Extra.": "profile_extraversion",
        "Agree.": "profile_agreeableness",
        "Neuro.": "profile_neuroticism",
    }
    T6_BFI = (
        T6_PROFILES.set_index("ID")[list(T6_PROFILE_RENAME)]
        .rename(columns=T6_PROFILE_RENAME)
    )
    T6_BFI.index = T6_BFI.index.astype(str).str.strip()

    T6_CASES = (
        T6_PREDICTION_DATA[["actual_trait", "predicted_trait"]]
        .join(T6_SIMILARITY.add_prefix("similarity_"), how="inner")
        .join(T6_TASK1, how="left")
        .join(T6_TASK3, how="left")
        .join(T6_BFI, how="left")
    )

    T6_CASES["prediction_correct"] = (
        T6_CASES["actual_trait"] == T6_CASES["predicted_trait"]
    )

    def T6_score_summary(row):
        T6_RANKED = row[T6_SIM_COLUMNS].astype(float).sort_values(
            ascending=False
        )
        return pd.Series({
            "top_similarity_trait": T6_RANKED.index[0].replace(
                "similarity_", ""
            ),
            "top_similarity": T6_RANKED.iloc[0],
            "second_similarity": T6_RANKED.iloc[1],
            "score_margin": T6_RANKED.iloc[0] - T6_RANKED.iloc[1],
        })

    T6_CASES = T6_CASES.join(
        T6_CASES.apply(T6_score_summary, axis=1)
    )

    # Normalize count measures so buckets of different sizes are comparable.
    T6_BUCKET_RATES = T6_TASK4.copy()
    T6_BUCKET_RATES["repetitions_per_1000"] = (
        1000 * T6_BUCKET_RATES["repetitions"] / T6_BUCKET_RATES["tokens"]
    )
    T6_BUCKET_RATES["confirmations_per_1000"] = (
        1000 * T6_BUCKET_RATES["yes_words"] / T6_BUCKET_RATES["tokens"]
    )
    T6_BUCKET_RATES["negations_per_1000"] = (
        1000 * T6_BUCKET_RATES["no_words"] / T6_BUCKET_RATES["tokens"]
    )
    T6_BUCKET_RATES = T6_BUCKET_RATES.reindex(T6_TRAITS)

    # Select one diagnostic case per actual trait. Prefer a wrong prediction,
    # then a larger top-versus-second similarity margin.
    T6_SELECTED_IDS = []
    for T6_TRAIT in T6_TRAITS:
        T6_GROUP = T6_CASES[T6_CASES["actual_trait"] == T6_TRAIT]
        if T6_GROUP.empty:
            continue

        T6_WRONG = T6_GROUP[~T6_GROUP["prediction_correct"]]
        T6_CANDIDATES = T6_WRONG if not T6_WRONG.empty else T6_GROUP
        T6_CHOSEN = T6_CANDIDATES.sort_values(
            "score_margin", ascending=False
        ).head(1)
        T6_SELECTED_IDS.extend(T6_CHOSEN.index.tolist())

    T6_SELECTED = T6_CASES.loc[T6_SELECTED_IDS].copy()

    T6_SELECTED[
        [
            "actual_trait", "predicted_trait",
            "top_similarity_trait", "top_similarity",
            "second_similarity", "score_margin",
            "prediction_correct",
        ]
    ].round(3).reset_index()

    return (
    T6_ACTUAL,
    T6_CASES,
    T6_SELECTED,
    T6_BUCKET_RATES,
    T6_TRAITS,
    T6_SIM_COLUMNS,
)


@app.cell
def _(
    T6_SELECTED,
    T6_TASK4,
    T6_SPEAKERS,
    T6_BUCKET_RATES,
    T6_TRAITS,
    T6_SIM_COLUMNS,
    pd,
):
    def T6_number(value):
        return round(float(value), 4) if pd.notna(value) else None

    def T6_group_rates(trait):
        if trait not in T6_BUCKET_RATES.index:
            return {}
        return {
            name: T6_number(T6_BUCKET_RATES.loc[trait, name])
            for name in (
                "repetitions_per_1000",
                "confirmations_per_1000",
                "negations_per_1000",
            )
        }

    T6_EVIDENCE_BY_ID = {}

    for T6_EVIDENCE_ANNOTATOR, T6_ROW in T6_SELECTED.iterrows():
        T6_PROFILE = {
            trait: T6_number(T6_ROW.get(f"profile_{trait}"))
            for trait in T6_TRAITS
        }
        T6_SIMILARITIES = {
            trait: T6_number(T6_ROW[f"similarity_{trait}"])
            for trait in T6_TRAITS
        }

        T6_PILE = T6_SPEAKERS.loc[
            T6_SPEAKERS["annotator"].astype(str) == str(T6_EVIDENCE_ANNOTATOR),
            "english_item",
        ]

        T6_EXAMPLES = []
        for T6_TEXT in T6_PILE.dropna().astype(str).drop_duplicates():
            T6_TEXT = " ".join(T6_TEXT.split())
            if T6_TEXT:
                T6_EXAMPLES.append(T6_TEXT[:220])
            if len(T6_EXAMPLES) == 2:
                break

        T6_EVIDENCE_BY_ID[str(T6_EVIDENCE_ANNOTATOR)] = {
            "annotator_id": str(T6_EVIDENCE_ANNOTATOR),
            "actual_dominant_trait": T6_ROW["actual_trait"],
            "task5_predicted_trait": T6_ROW["predicted_trait"],
            "highest_similarity_trait": T6_ROW["top_similarity_trait"],
            "top_similarity": T6_number(T6_ROW["top_similarity"]),
            "second_highest_similarity": T6_number(
                T6_ROW["second_similarity"]
            ),
            "top_score_margin": T6_number(T6_ROW["score_margin"]),
            "similarity_scores_not_probabilities": T6_SIMILARITIES,
            "big_five_profile_scores": T6_PROFILE,
            "task1_word_statistics": {
                "tokens": T6_number(T6_ROW.get("tokens")),
                "vocabulary_size": T6_number(T6_ROW.get("vocabulary_size")),
                "repetitions": T6_number(T6_ROW.get("repetitions")),
                "repetitions_per_1000": T6_number(
                    1000 * T6_ROW["repetitions"] / T6_ROW["tokens"]
                ),
                "confirmations_per_1000": T6_number(
                    1000 * T6_ROW["yes_words"] / T6_ROW["tokens"]
                ),
                "negations_per_1000": T6_number(
                    1000 * T6_ROW["no_words"] / T6_ROW["tokens"]
                ),
            },
            "task3_vad_statistics": {
                name: T6_number(T6_ROW.get(name))
                for name in (
                    "valence_mean", "valence_std",
                    "arousal_mean", "arousal_std",
                    "dominance_mean", "dominance_std",
                )
            },
            "task4_bucket_rates_per_1000_tokens": {
                "actual_trait_group": T6_group_rates(
                    T6_ROW["actual_trait"]
                ),
                "predicted_trait_group": T6_group_rates(
                    T6_ROW["predicted_trait"]
                ),
                "all_trait_groups": {
                    trait: T6_group_rates(trait)
                    for trait in T6_TRAITS
                },
            },
            "event_text_examples_not_authored_by_annotator": T6_EXAMPLES,
        }

    T6_EVIDENCE_TABLE = T6_SELECTED[
    [
        "actual_trait",
        "predicted_trait",
        "top_similarity_trait",
        "top_similarity",
        "second_similarity",
        "score_margin",
        "prediction_correct",
        "tokens",
        "vocabulary_size",
        "valence_mean",
        "arousal_mean",
        "dominance_mean",
    ]
    ].copy()

    # Add normalized per-annotator count rates from Task 1.
    T6_EVIDENCE_TABLE["repetitions_per_1000"] = (
        1000 * T6_SELECTED["repetitions"] / T6_SELECTED["tokens"]
    )
    T6_EVIDENCE_TABLE["confirmations_per_1000"] = (
        1000 * T6_SELECTED["yes_words"] / T6_SELECTED["tokens"]
    )
    T6_EVIDENCE_TABLE["negations_per_1000"] = (
        1000 * T6_SELECTED["no_words"] / T6_SELECTED["tokens"]
    )
    T6_EVIDENCE_TABLE = T6_EVIDENCE_TABLE.round(3).reset_index()

    T6_EVIDENCE_TABLE
    return T6_EVIDENCE_BY_ID, T6_EVIDENCE_TABLE



@app.cell
def _(mo):
    mo.md("""
    **What the evidence table shows**

    The five selected cases cover all five actual dominant-trait
    categories. Four were misclassified by Task 5, while E17 was
    classified correctly.

    All five cases received neuroticism as the highest-similarity
    prediction. The margin between the top two similarity scores ranges
    from 0.0353 to 0.0726. E9 has the smallest margin, while E22 has
    the largest.

    Actual traits come from the highest Big Five profile score, while
    predicted traits come from Doc2Vec similarity scores. Similarity
    scores are not probabilities, and a larger margin does not guarantee
    a correct prediction.

    These are diagnostic examples chosen to show errors, not a fair
    sample for estimating the model's overall performance.
    """)
    return

@app.cell
def _(mo):
    mo.md("""
    **Re-runs reuse cached explanations.** Each answer is keyed by a hash of
    (its evidence + the prompt version). Same inputs = the stored text.
    New evidence or an edited prompt (version bumped) = fresh API calls.

    The word count is recorded for each explanation as a soft check.
    Responses are not rejected solely for exceeding 250 words.
    """)
    return

@app.cell
def _(
    Path,
    explain_trait,
    hashlib,
    json,
    pd,
    PROMPT_VERSION,
    T6_EVIDENCE_BY_ID,
):
    # Create the output directory next to the file write.
    T6_OUTPUT_TABLES = Path("outputs/tables")
    T6_OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    T6_CACHE_PATH = T6_OUTPUT_TABLES / "task6_llm_explanations.csv"

    # Cache successful responses so reruns do not repeat API requests.
    T6_CACHE_BY_ID = {}

    if T6_CACHE_PATH.exists():
        T6_OLD_CACHE = pd.read_csv(T6_CACHE_PATH).fillna("")
        T6_CACHE_BY_ID = {
            str(row["annotator"]): row.to_dict()
            for _, row in T6_OLD_CACHE.iterrows()
        }

    T6_CURRENT_RESULTS = []

    for T6_ANNOTATOR, T6_EVIDENCE in T6_EVIDENCE_BY_ID.items():
        T6_EVIDENCE_HASH = hashlib.sha256(
            json.dumps(
                T6_EVIDENCE,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        T6_OLD = T6_CACHE_BY_ID.get(T6_ANNOTATOR, {})
        T6_CACHE_VALID = (
            T6_OLD.get("prompt_version") == PROMPT_VERSION
            and T6_OLD.get("evidence_hash") == T6_EVIDENCE_HASH
            and T6_OLD.get("status") == "success"
            and bool(T6_OLD.get("explanation"))
        )

        if T6_CACHE_VALID:
            # Reuse the explanation and update its word count.
            T6_RESULT = T6_OLD.copy()
            T6_RESULT["word_count"] = len(
                str(T6_RESULT["explanation"]).split()
            )
            T6_CACHE_BY_ID[T6_ANNOTATOR] = T6_RESULT

        else:
            try:
                T6_RESPONSE = explain_trait(T6_EVIDENCE)
                T6_EXPLANATION = T6_RESPONSE["explanation"]

                T6_RESULT = {
                    "annotator": T6_ANNOTATOR,
                    "actual_trait": T6_EVIDENCE["actual_dominant_trait"],
                    "predicted_trait": T6_EVIDENCE["task5_predicted_trait"],
                    "score_margin": T6_EVIDENCE["top_score_margin"],
                    "model": T6_RESPONSE["model"],
                    "prompt_version": T6_RESPONSE["prompt_version"],
                    "evidence_hash": T6_EVIDENCE_HASH,
                    "explanation": T6_EXPLANATION,
                    "word_count": len(T6_EXPLANATION.split()),
                    "status": "success",
                    "error": "",
                }

                # Save each successful response immediately.
                T6_CACHE_BY_ID[T6_ANNOTATOR] = T6_RESULT
                pd.DataFrame(T6_CACHE_BY_ID.values()).to_csv(
                    T6_CACHE_PATH,
                    index=False,
                )

            except RuntimeError as T6_ERROR:
                T6_RESULT = {
                    "annotator": T6_ANNOTATOR,
                    "actual_trait": T6_EVIDENCE["actual_dominant_trait"],
                    "predicted_trait": T6_EVIDENCE["task5_predicted_trait"],
                    "score_margin": T6_EVIDENCE["top_score_margin"],
                    "model": "",
                    "prompt_version": PROMPT_VERSION,
                    "evidence_hash": T6_EVIDENCE_HASH,
                    "explanation": "",
                    "word_count": 0,
                    "status": "error",
                    "error": str(T6_ERROR),
                }

        T6_CURRENT_RESULTS.append(T6_RESULT)

    # Persist word counts even if all explanations came from the cache.
    pd.DataFrame(T6_CACHE_BY_ID.values()).to_csv(
        T6_CACHE_PATH,
        index=False,
    )

    T6_EXPLANATIONS = pd.DataFrame(T6_CURRENT_RESULTS)

    T6_EXPLANATIONS[
        [
            "annotator",
            "actual_trait",
            "predicted_trait",
            "score_margin",
            "model",
            "prompt_version",
            "evidence_hash",
            "word_count",
            "status",
            "explanation",
            "error",
        ]
    ]

    return (T6_EXPLANATIONS,)


@app.cell
def _(mo, T6_EXPLANATIONS):
    T6_SUCCEEDED = (
        int((T6_EXPLANATIONS["status"] == "success").sum())
        if not T6_EXPLANATIONS.empty else 0
    )

    mo.md(f"""
    **LLM generation status:** {T6_SUCCEEDED} of
    {len(T6_EXPLANATIONS)} selected explanations were generated successfully.

    **Word-count check:** The prompt requests no more than 250 words.
    The saved counts are E3: 284, E9: 217, E10: 225, E17: 233, and
    E22: 217 words. E3 exceeds the requested limit. Counts are recorded
    in the CSV as a soft check; longer responses are retained rather
    than automatically rejected.

    **What the LLM explanations show**

    The explanations use the supplied similarity scores, Big Five
    profiles, word statistics and VAD measurements to discuss supporting
    evidence, counter-arguments and alternative interpretations.

    All five cases received neuroticism as the predicted trait. Only E17
    has neuroticism as its actual dominant trait.

    The explanations provide a structured interpretation, but their
    claims must be checked against the supplied measurements. A plausible
    narrative is not evidence that a prediction is correct.

    These explanations come after the prediction: they describe the
    result, not what caused it. The LLM cannot inspect Doc2Vec's internal
    computations or establish the causal reason for an individual prediction.
    """)
    return


@app.cell
def _(Path, T6_SELECTED, T6_SIM_COLUMNS, T6_TRAITS, plt):
    T6_HEAT_FIG, T6_HEAT_AX = plt.subplots(
        figsize=(9, max(3.5, 0.7 * len(T6_SELECTED)))
    )
    T6_HEAT_DATA = T6_SELECTED[T6_SIM_COLUMNS].astype(float).to_numpy()

    T6_HEAT_IMAGE = T6_HEAT_AX.imshow(
        T6_HEAT_DATA, aspect="auto"
    )
    T6_HEAT_AX.set_xticks(range(len(T6_TRAITS)))
    T6_HEAT_AX.set_xticklabels(T6_TRAITS, rotation=30, ha="right")
    T6_HEAT_AX.set_yticks(range(len(T6_SELECTED)))
    T6_HEAT_AX.set_yticklabels([
        f"{annotator}: {row.actual_trait} → {row.predicted_trait}"
        for annotator, row in T6_SELECTED.iterrows()
    ])
    T6_HEAT_AX.set_title("Task 5 trait similarities for selected cases")
    T6_HEAT_FIG.colorbar(
        T6_HEAT_IMAGE, ax=T6_HEAT_AX, label="Cosine similarity"
    )
    T6_HEAT_FIG.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    T6_HEAT_FIG.savefig(
        Path("outputs/figures") / "task6_similarity_heatmap.png",
        dpi=150, bbox_inches="tight",
    )
    T6_HEAT_FIG
    return T6_HEAT_FIG,


@app.cell
def _(mo):
    mo.md("""
    **What the similarity heatmap shows**

    Cosine similarity compares vector directions. Values closer to 1
    indicate more similar directions, while values near 0 indicate
    little alignment. Similarity scores are not probabilities.

    Neuroticism has the highest similarity score for all five selected
    cases. However, only E17 has neuroticism as the actual dominant trait.
    The other four cases have different actual traits despite receiving
    the same prediction.

    E22 has the largest top-versus-second margin, approximately 0.073.
    E9 has the smallest, approximately 0.035. These margins show how far
    the leading score is above the runner-up, not the probability that
    the prediction is correct.

    The heatmap shows that the Task 5 method favors neuroticism across
    cases with different actual traits. Individual similarity rankings
    should therefore be interpreted cautiously.
    """)
    return


@app.cell
def _(Path, T6_BUCKET_RATES, plt):
    T6_RATE_COLUMNS = [
        ("repetitions_per_1000", "Repetitions per 1,000 tokens"),
        ("confirmations_per_1000", "Confirmations per 1,000 tokens"),
        ("negations_per_1000", "Negations per 1,000 tokens"),
    ]

    T6_RATE_FIG, T6_RATE_AXES = plt.subplots(1, 3, figsize=(14, 4))

    for T6_AX, (T6_COLUMN, T6_TITLE) in zip(
        T6_RATE_AXES, T6_RATE_COLUMNS
    ):
        T6_AX.bar(
            T6_BUCKET_RATES.index, T6_BUCKET_RATES[T6_COLUMN]
        )
        T6_AX.set_title(T6_TITLE)
        T6_AX.tick_params(axis="x", rotation=70)
        T6_AX.set_ylabel("Count per 1,000 tokens")

    T6_RATE_FIG.suptitle("Task 4 personality-bucket word rates")
    T6_RATE_FIG.tight_layout()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    T6_RATE_FIG.savefig(
        Path("outputs/figures") / "task6_bucket_rates.png",
        dpi=150, bbox_inches="tight",
    )
    T6_RATE_FIG
    return T6_RATE_FIG,


@app.cell
def _(mo):
    mo.md("""
    **What the word-rate charts show**

    Repetition rates range from 325.993 to 332.587 per 1,000 tokens.
    Confirmation rates range from 1.418 to 1.476, and negation rates
    range from 13.682 to 14.214.

    The differences between the five personality-trait groups are small.
    These word-count features provide limited descriptive evidence for
    distinguishing the traits. Normalizing by token count improves
    comparability, but does not establish statistical significance.
    """)
    return


@app.cell
def _(Path, T6_ACTUAL, T6_TASK3, T6_TRAITS, plt):
    T6_VAD_COLUMNS = [
        "valence_mean",
        "arousal_mean",
        "dominance_mean",
    ]

    # Use all annotators, not just the five selected examples.
    T6_VAD_DATA = T6_TASK3.join(T6_ACTUAL, how="inner")

    T6_VAD_BY_TRAIT = (
        T6_VAD_DATA.groupby("actual_trait")[T6_VAD_COLUMNS]
        .mean()
        .reindex(T6_TRAITS)
    )

    # Show the number of annotators in each trait group.
    T6_VAD_COUNTS = (
        T6_VAD_DATA["actual_trait"]
        .value_counts()
        .reindex(T6_TRAITS, fill_value=0)
    )

    T6_VAD_FIG, T6_VAD_AX = plt.subplots(figsize=(10, 5))

    T6_VAD_BY_TRAIT.rename(columns={
        "valence_mean": "Valence",
        "arousal_mean": "Arousal",
        "dominance_mean": "Dominance",
    }).plot(kind="bar", ax=T6_VAD_AX)

    T6_VAD_AX.set_title("Mean VAD scores by actual dominant trait")
    T6_VAD_AX.set_xlabel("Actual dominant trait")
    T6_VAD_AX.set_ylabel("Mean lexicon score")
    T6_VAD_AX.set_xticklabels(
        [
            f"{trait}\n(n={T6_VAD_COUNTS[trait]})"
            for trait in T6_TRAITS
        ],
        rotation=25,
        ha="right",
    )
    T6_VAD_AX.legend(title="VAD dimension")
    T6_VAD_FIG.tight_layout()

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    T6_VAD_FIG.savefig(
        Path("outputs/figures") / "task6_vad_by_trait.png",
        dpi=150,
        bbox_inches="tight",
    )

    T6_VAD_FIG
    return T6_VAD_FIG, T6_VAD_BY_TRAIT


@app.cell
def _(mo):
    mo.md("""
    **What the VAD chart shows**

    Using all 36 annotators, mean valence ranges from 0.6027 to 0.6050,
    arousal from 0.4396 to 0.4410, and dominance from 0.5218 to 0.5241.
    The averages are very close, with no clear visual separation between
    the five personality-trait groups.

    Group sizes are uneven, with 19 annotators in conscientiousness,
    10 in agreeableness, 3 in openness, and only 2 each in extraversion
    and neuroticism. These results provide little descriptive evidence
    that aggregate VAD measures distinguish the personality traits.

    VAD scores describe the emotional characteristics of words in the
    event texts. They do not directly measure annotators' emotional
    states or establish their personality traits. Annotators reacted
    to these event texts rather than writing them themselves.
    """)
    return

@app.cell
def _(mo):
    mo.md("""
    ## Task 6 comment

    The LLM was prompted to explain the Doc2Vec personality predictions
    using similarity scores, Big Five profiles, word statistics and VAD
    measurements. Five cases were selected, one for each actual dominant
    personality trait.

    Task 5 predicted neuroticism for all five cases, but only E17 had
    neuroticism as its actual dominant trait. Across all 36 annotators,
    the underlying method achieved 11.1% accuracy, below the uniform
    baseline of 20% and majority-class baseline of 52.8%.

    The word-rate charts show only small differences between trait groups.
    VAD averages across all 36 annotators are also very close: valence
    ranges from 0.6027 to 0.6050, arousal from 0.4396 to 0.4410, and
    dominance from 0.5218 to 0.5241. Uneven group sizes warrant caution
    when interpreting these descriptive comparisons.

    All five LLM explanations were generated successfully. Four are
    within the requested 250-word limit, while E3 returned 284 words.
    The word counts are recorded in the output CSV, and the longer
    response is retained rather than silently discarded.

    The explanations help communicate and critically examine the findings,
    but they describe predictions after the fact. They do not establish
    what caused Doc2Vec to make a particular prediction or prove that
    the method reliably distinguishes personality traits.

    **Conclusion:** LLM prompting makes the findings easier to interpret,
    but it does not validate the underlying personality predictions.
    """)
    return

@app.cell
def _(mo):
    mo.md("""
    ## Outputs

    - `outputs/tables/task6_llm_explanations.csv` — five explanations,
      including the model, prompt version, evidence hash, status and word count.
    - `outputs/figures/task6_similarity_heatmap.png` — similarity scores
      for the selected cases.
    - `outputs/figures/task6_bucket_rates.png` — normalized word rates
      by personality-trait group.
    - `outputs/figures/task6_vad_by_trait.png` — average VAD scores by
      actual dominant trait, using all annotators with available measurements.
    """)
    return

if __name__ == "__main__":
    app.run()
